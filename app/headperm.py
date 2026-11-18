"""目标模型注意力头重排导出与核对（设备端指定头顺序）。

部署场景：把目标 Transformer 某一个块的 4 个注意力头按指定顺序重排，
产出可重新加载的权重文件。重排是保函数变换：

  - Q/K/V 投影（in_proj_weight / in_proj_bias）按头整段重排**行**；
    in_proj 布局为 [q_proj; k_proj; v_proj]，每段 d_model 行内
    第 h 个头占 [h*head_dim, (h+1)*head_dim)；
  - 输出投影 out_proj.weight 按头整段重排**列**（其输入是各头输出的拼接，
    第 h 个头占输入维 [h*head_dim, (h+1)*head_dim)）；
    out_proj.bias 是逐输出维的偏置，与头的顺序无关，保持不变；
  - 排列语义：order[i] = 新位置 i 上放置的原头编号。

数学上重排前后逐点等价（out_proj 只是换序求和），float32 下仅有
~1e-6 级归约顺序误差，核对阈值取 1e-3。

文件与绑定：
  - torch.save 单文件，含 state_dict 与元数据（块号、排列、结构参数）；
  - 共享词嵌入与 LM head 的权重绑定写入文件，重载后显式重新绑定并校验；

隔离与并发：
  - 每次导出构建全新模型实例，进程内 /api/decode 使用的单例旧权重
    完全不被触碰；
  - 每次导出分配独立 export_id 与独立临时文件，先核对、通过后原子改名
    落盘；报告与文件按 export_id 一一对应，并发导出互不串扰；
  - 核对任一失败即删除临时文件，不提供下载。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import uuid
from datetime import datetime, timezone

import torch

from . import tokenizer
from .model import (
    D_MODEL,
    MAX_POS,
    TARGET_HEADS,
    TARGET_LAYERS,
    VOCAB_SIZE,
    CausalTransformer,
    _build_pair,
    greedy_argmax,
)

# 核对阈值：无缓存 / KV 增量 logits 的最大绝对误差上限
TOLERANCE = 1e-3

# 权重文件格式版本（重载时校验，避免误载不兼容文件）
FILE_FORMAT = "headperm-v1"

# 导出文件目录（可用环境变量覆盖；Docker 容器内为 /srv/exports）
EXPORT_DIR = os.environ.get(
    "EXPORT_DIR", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "exports"))
)

# 内存报告注册表：export_id -> report（另在磁盘保留 JSON 副本，重启后可查）
_REGISTRY: dict[str, dict] = {}
_REGISTRY_LOCK = threading.Lock()

_EXPORT_ID_RE = re.compile(r"[0-9a-f]{32}")


# ---------------------------------------------------------------- 参数校验


def validate_request(block: int, order: list[int], prompts: list[str],
                     max_new: int) -> None:
    """校验导出请求，不合法时抛 ValueError（由路由层转成 400）。"""
    if not isinstance(block, int) or not 0 <= block < TARGET_LAYERS:
        raise ValueError(f"块号必须是 0~{TARGET_LAYERS - 1} 的整数。")
    if (not isinstance(order, list) or len(order) != TARGET_HEADS
            or sorted(order) != list(range(TARGET_HEADS))):
        raise ValueError(
            f"头排列必须是 0~{TARGET_HEADS - 1} 的一个排列（长度 {TARGET_HEADS}，不重复）。")
    if list(order) == list(range(TARGET_HEADS)):
        raise ValueError("头排列必须是非原顺序（恒等排列不需要重排导出）。")
    if not isinstance(prompts, list) or not 1 <= len(prompts) <= 3:
        raise ValueError("提示词需要 1~3 条。")
    for p in prompts:
        if not isinstance(p, str) or not 1 <= len(p) <= 32:
            raise ValueError("每条提示词须为 1~32 个字符。")
        if any(ord(c) < 0x20 or ord(c) > 0x7E for c in p):
            raise ValueError("提示词只能包含可打印 ASCII 字符（0x20~0x7E）。")
    if not isinstance(max_new, int) or not 1 <= max_new <= 8:
        raise ValueError("续写长度 max_new 必须是 1~8 的整数。")


# ---------------------------------------------------------------- 头重排


def permute_block_heads_(model: CausalTransformer, block: int,
                         order: list[int]) -> None:
    """就地重排 model 第 block 块的注意力头（Q/K/V 行 + 输出投影列）。

    order[i] = 新位置 i 上放置的原头编号。调用方需保证 order 已通过校验。
    """
    layer = model.blocks[block]
    attn = layer.self_attn
    dm, hd = model.d_model, model.head_dim

    with torch.no_grad():
        w = attn.in_proj_weight   # (3*d_model, d_model)，[q; k; v] 三段
        b = attn.in_proj_bias     # (3*d_model,)
        new_w = torch.empty_like(w)
        new_b = torch.empty_like(b)
        for pos, src in enumerate(order):
            for s in range(3):  # 0=q, 1=k, 2=v，三段同样重排
                dst = slice(s * dm + pos * hd, s * dm + (pos + 1) * hd)
                src_sl = slice(s * dm + src * hd, s * dm + (src + 1) * hd)
                new_w[dst] = w[src_sl]
                new_b[dst] = b[src_sl]
        w.copy_(new_w)
        b.copy_(new_b)

        ow = attn.out_proj.weight  # (d_model, d_model)，列对应拼接后的各头
        new_ow = torch.empty_like(ow)
        for pos, src in enumerate(order):
            new_ow[:, pos * hd:(pos + 1) * hd] = ow[:, src * hd:(src + 1) * hd]
        ow.copy_(new_ow)
        # out_proj.bias 逐输出维，与头顺序无关，保持不变


def build_permuted_target(block: int, order: list[int]) -> CausalTransformer:
    """构建一个全新的、与进程单例权重一致的目标模型，并就地重排指定块。

    权重来自确定性构建（与单例逐位相同），不触碰单例本身。
    """
    _, target = _build_pair()
    permute_block_heads_(target, block, list(order))
    return target


# ---------------------------------------------------------------- 文件存取


def _new_target_shell() -> CausalTransformer:
    """构建目标模型空壳（结构 + 权重绑定），参数值随后由 load_state_dict 覆盖。

    直接走 _build_pair（内部持构建锁）：nn 模块初始化会消耗全局 RNG，
    若不加锁地与并发的 _build_pair 交叉，会破坏对方的确定性权重流。
    """
    return _build_pair()[1]


def save_weights_file(model: CausalTransformer, path: str, meta: dict) -> None:
    """把模型权重与元数据写入单文件（tied 张量在文件内只存一份）。"""
    payload = {
        "format": FILE_FORMAT,
        "meta": meta,
        "state_dict": model.state_dict(),
    }
    torch.save(payload, path)


def load_weights_file(path: str) -> tuple[CausalTransformer, dict]:
    """从文件重载目标模型，恢复共享嵌入/LM head 绑定，返回 (model, meta)。"""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format") != FILE_FORMAT:
        raise ValueError(f"不支持的权重文件格式: {payload.get('format')!r}")
    meta = payload["meta"]
    if (meta.get("n_layers") != TARGET_LAYERS or meta.get("n_heads") != TARGET_HEADS
            or meta.get("d_model") != D_MODEL or meta.get("vocab_size") != VOCAB_SIZE
            or meta.get("max_pos") != MAX_POS):
        raise ValueError("权重文件结构参数与当前模型不一致。")

    model = _new_target_shell()
    model.load_state_dict(payload["state_dict"], strict=True)
    # 显式恢复权重绑定（load_state_dict 是逐参数 copy_，这里把绑定关系坐实）
    model.lm_head.weight = model.tok_emb.weight
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, meta


# ---------------------------------------------------------------- 核对


def _max_abs_err(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max().item())


def _nocache_err(m1: CausalTransformer, m2: CausalTransformer,
                 ids: list[int]) -> float:
    """无缓存整序列前向，逐位置 logits 的最大绝对误差。"""
    return _max_abs_err(m1.forward(ids, use_cache=False),
                        m2.forward(ids, use_cache=False))


def _kv_incremental_err(m1: CausalTransformer, m2: CausalTransformer,
                        ids: list[int]) -> float:
    """KV 缓存逐 token 增量前向，每步 logits 的最大绝对误差。"""
    m1.init_cache()
    m2.init_cache()
    err = 0.0
    for t in ids:
        err = max(err, _max_abs_err(m1.forward([t], use_cache=True),
                                    m2.forward([t], use_cache=True)))
    return err


def _greedy_ids(model: CausalTransformer, prompt_ids: list[int], max_new: int,
                use_cache: bool) -> list[int]:
    """贪心续写（EOS 计入输出并停止，平局取小 id，与 decoding.py 语义一致）。"""
    out: list[int] = []
    if use_cache:
        model.init_cache()
        logits = model.forward(prompt_ids, use_cache=True)[-1]
        for _ in range(max_new):
            nxt = greedy_argmax(logits)
            out.append(nxt)
            if nxt == tokenizer.EOS_ID:
                break
            logits = model.forward([nxt], use_cache=True)[-1]
    else:
        for _ in range(max_new):
            logits = model.forward(list(prompt_ids) + out, use_cache=False)
            nxt = greedy_argmax(logits[-1])
            out.append(nxt)
            if nxt == tokenizer.EOS_ID:
                break
    return out


def _verify(original: CausalTransformer, reloaded: CausalTransformer,
            prompts: list[str], max_new: int) -> dict:
    """逐提示词核对原模型与重载模型，返回报告的 prompts 部分与汇总。"""
    per_prompt: list[dict] = []
    err_nc_all = 0.0
    err_kv_all = 0.0
    greedy_all_match = True
    for text in prompts:
        ids = tokenizer.encode(text)
        err_nc = _nocache_err(original, reloaded, ids)
        err_kv = _kv_incremental_err(original, reloaded, ids)
        g_orig = _greedy_ids(original, ids, max_new, use_cache=False)
        g_reload = _greedy_ids(reloaded, ids, max_new, use_cache=False)
        g_reload_kv = _greedy_ids(reloaded, ids, max_new, use_cache=True)
        match = g_orig == g_reload == g_reload_kv
        err_nc_all = max(err_nc_all, err_nc)
        err_kv_all = max(err_kv_all, err_kv)
        greedy_all_match = greedy_all_match and match
        per_prompt.append({
            "prompt": text,
            "prompt_ids": ids,
            "max_abs_err_nocache": err_nc,
            "max_abs_err_kv": err_kv,
            "greedy_ids_original": g_orig,
            "greedy_ids_reloaded": g_reload,
            "greedy_ids_reloaded_kv": g_reload_kv,
            "greedy_tokens_view": [
                {"id": t, "text": tokenizer.render_token(t)} for t in g_reload
            ],
            "greedy_match": match,
        })
    return {
        "prompts": per_prompt,
        "max_abs_err_nocache": err_nc_all,
        "max_abs_err_kv": err_kv_all,
        "greedy_all_match": greedy_all_match,
    }


# ---------------------------------------------------------------- 导出流程


def _export_file_path(export_id: str) -> str:
    return os.path.join(EXPORT_DIR, f"headperm_{export_id}.pt")


def _report_file_path(export_id: str) -> str:
    return os.path.join(EXPORT_DIR, f"headperm_{export_id}.json")


def _sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def export_head_permutation(block: int, order: list[int], prompts: list[str],
                            max_new: int) -> dict:
    """完整导出流程：重排 → 写临时文件 → 从文件重载 → 核对 → 通过才落盘。

    返回报告 dict；核对失败时报告 status=FAIL 且不含下载信息（文件已删除）。
    全程只使用本次调用新建的模型实例，进程单例（旧权重）不受影响。
    """
    validate_request(block, order, prompts, max_new)
    order = list(order)

    export_id = uuid.uuid4().hex
    os.makedirs(EXPORT_DIR, exist_ok=True)
    tmp_path = os.path.join(EXPORT_DIR, f".tmp_{export_id}.pt")
    created_at = datetime.now(timezone.utc).isoformat()

    meta = {
        "block": block,
        "order": order,
        "n_layers": TARGET_LAYERS,
        "n_heads": TARGET_HEADS,
        "d_model": D_MODEL,
        "vocab_size": VOCAB_SIZE,
        "max_pos": MAX_POS,
        "tied_weights": ["tok_emb.weight", "lm_head.weight"],
        "created_at": created_at,
    }

    report: dict = {
        "export_id": export_id,
        "status": "FAIL",
        "block": block,
        "order": order,
        "max_new": max_new,
        "tolerance": TOLERANCE,
        "n_layers": TARGET_LAYERS,
        "n_heads": TARGET_HEADS,
        "created_at": created_at,
        "fail_reasons": [],
    }

    try:
        # 1. 全新实例上重排（单例旧权重不受影响），写临时文件
        permuted = build_permuted_target(block, order)
        save_weights_file(permuted, tmp_path, meta)

        # 2. 从文件重载，核对权重绑定
        reloaded, _ = load_weights_file(tmp_path)
        tied = (reloaded.lm_head.weight is reloaded.tok_emb.weight
                or reloaded.lm_head.weight.data_ptr()
                == reloaded.tok_emb.weight.data_ptr())
        report["tied_weights_preserved"] = tied

        # 3. 全新的原模型（确定性权重 == 单例旧权重），逐提示词核对
        _, original = _build_pair()
        report.update(_verify(original, reloaded, prompts, max_new))

        # 4. 汇总判定
        if not tied:
            report["fail_reasons"].append("重载后共享嵌入与 LM head 的权重绑定丢失。")
        if report["max_abs_err_nocache"] > TOLERANCE:
            report["fail_reasons"].append(
                f"无缓存 logits 最大绝对误差 {report['max_abs_err_nocache']:.3e} "
                f"超过阈值 {TOLERANCE:.0e}。")
        if report["max_abs_err_kv"] > TOLERANCE:
            report["fail_reasons"].append(
                f"KV 增量 logits 最大绝对误差 {report['max_abs_err_kv']:.3e} "
                f"超过阈值 {TOLERANCE:.0e}。")
        if not report["greedy_all_match"]:
            report["fail_reasons"].append("贪心续写序列不一致。")

        if report["fail_reasons"]:
            os.remove(tmp_path)  # 核对失败：不提供文件
        else:
            # 5. 核对通过：原子落盘，报告与文件按 export_id 一一对应
            final_path = _export_file_path(export_id)
            os.replace(tmp_path, final_path)
            report["status"] = "PASS"
            report["file_name"] = os.path.basename(final_path)
            report["file_bytes"] = os.path.getsize(final_path)
            report["sha256"] = _sha256_of(final_path)
            report["download_url"] = f"/api/headperm/{export_id}/download"
            report["download_name"] = (
                f"headperm_block{block}_perm{''.join(map(str, order))}"
                f"_{export_id[:8]}.pt")
            # 磁盘留一份报告副本（先写临时文件再原子改名）
            tmp_json = os.path.join(EXPORT_DIR, f".tmp_{export_id}.json")
            with open(tmp_json, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            os.replace(tmp_json, _report_file_path(export_id))
    finally:
        # 任何异常路径都不留临时文件
        for suffix in (".pt", ".json"):
            p = os.path.join(EXPORT_DIR, f".tmp_{export_id}{suffix}")
            if os.path.exists(p):
                os.remove(p)

    with _REGISTRY_LOCK:
        _REGISTRY[export_id] = report
    return report


def get_report(export_id: str) -> dict | None:
    """按 export_id 取报告（先内存注册表，后磁盘副本）。"""
    if not _EXPORT_ID_RE.fullmatch(export_id):
        return None
    with _REGISTRY_LOCK:
        report = _REGISTRY.get(export_id)
    if report is not None:
        return report
    path = _report_file_path(export_id)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


def resolve_export_file(export_id: str) -> str | None:
    """按 export_id 解析下载文件路径；不存在（含核对失败）时返回 None。"""
    if not _EXPORT_ID_RE.fullmatch(export_id):
        return None
    path = _export_file_path(export_id)
    return path if os.path.exists(path) else None
