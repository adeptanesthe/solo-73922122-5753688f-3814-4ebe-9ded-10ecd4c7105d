"""注意力头重排导出的关键功能测试：

  1. 重排是保函数变换：各块/各非恒等排列下，重排模型与原模型 logits 误差 ≤1e-3；
  2. Q/K/V 行与 out_proj 列确实按指定排列改排，out_proj.bias 与其他块不动；
  3. 非法参数（恒等排列、重复、越界、坏块号、坏提示词）一律拒绝；
  4. 权重文件可重载，且共享嵌入与 LM head 的绑定在重载后保留；
  5. 导出核对全通过才给文件；失败（阈值被强制压到负数）不留文件、不能下载；
  6. 导出后进程单例（旧权重）逐位不变，原解码结果不变；
  7. 并发导出不同排列：每份报告的 sha256/排列与对应文件严格一致，互不串扰。
"""

from __future__ import annotations

import hashlib
import os
import sys
import tempfile
import threading

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app import headperm, tokenizer
from app.decoding import compare
from app.model import TARGET_HEADS, TARGET_LAYERS, _build_pair, get_models

torch.set_num_threads(1)

# 隔离的导出目录（不影响仓库内文件）
_TMP_EXPORT = tempfile.mkdtemp(prefix="headperm_test_")
headperm.EXPORT_DIR = _TMP_EXPORT

PROMPTS = ["the quick brown fox ", "hello world! ", "0 1 1 2 3 5 8 "]


def _fresh_target() -> torch.nn.Module:
    return _build_pair()[1]


def test_permutation_preserves_function() -> None:
    """每个块 × 若干非恒等排列：无缓存与 KV 增量 logits 误差均 ≤1e-3。"""
    orders = [[1, 0, 2, 3], [3, 2, 1, 0], [2, 0, 3, 1], [1, 2, 3, 0]]
    ids = tokenizer.encode("the quick brown fox jumps ")
    worst = 0.0
    for block in range(TARGET_LAYERS):
        original = _fresh_target()
        for order in orders:
            permuted = headperm.build_permuted_target(block, order)
            err_nc = headperm._nocache_err(original, permuted, ids)
            err_kv = headperm._kv_incremental_err(original, permuted, ids)
            worst = max(worst, err_nc, err_kv)
            assert err_nc <= headperm.TOLERANCE, (block, order, err_nc)
            assert err_kv <= headperm.TOLERANCE, (block, order, err_kv)
    print(f"ok: {TARGET_LAYERS} 块 × {len(orders)} 排列重排后 logits 误差 "
          f"≤1e-3（实测最大 {worst:.2e}）")


def test_weights_actually_permuted() -> None:
    """Q/K/V 行与 out_proj 列按排列改排；bias 与其他块保持原值。"""
    block, order = 1, [2, 0, 3, 1]
    original = _fresh_target()
    permuted = _fresh_target()
    headperm.permute_block_heads_(permuted, block, order)

    o_attn = original.blocks[block].self_attn
    p_attn = permuted.blocks[block].self_attn
    hd = original.head_dim
    dm = original.d_model
    for pos, src in enumerate(order):
        for s in range(3):
            dst = slice(s * dm + pos * hd, s * dm + (pos + 1) * hd)
            src_sl = slice(s * dm + src * hd, s * dm + (src + 1) * hd)
            assert torch.equal(p_attn.in_proj_weight[dst], o_attn.in_proj_weight[src_sl])
            assert torch.equal(p_attn.in_proj_bias[dst], o_attn.in_proj_bias[src_sl])
        assert torch.equal(p_attn.out_proj.weight[:, pos * hd:(pos + 1) * hd],
                           o_attn.out_proj.weight[:, src * hd:(src + 1) * hd])
    # out_proj.bias 不变；其他块完全不动；共享嵌入不动
    assert torch.equal(p_attn.out_proj.bias, o_attn.out_proj.bias)
    for b in range(TARGET_LAYERS):
        if b == block:
            continue
        for p1, p2 in zip(original.blocks[b].parameters(),
                          permuted.blocks[b].parameters()):
            assert torch.equal(p1, p2)
    assert torch.equal(permuted.tok_emb.weight, original.tok_emb.weight)
    # 且重排确实改变了该块权重（非恒等排列）
    assert not torch.equal(p_attn.in_proj_weight, o_attn.in_proj_weight)
    print("ok: Q/K/V 行与 out_proj 列按排列改排，bias/其他块/嵌入不动")


def test_invalid_requests_rejected() -> None:
    bad = [
        dict(block=0, order=[0, 1, 2, 3], prompts=["a"], max_new=8),      # 恒等排列
        dict(block=0, order=[0, 0, 1, 2], prompts=["a"], max_new=8),      # 重复
        dict(block=0, order=[0, 1, 2, 4], prompts=["a"], max_new=8),      # 越界
        dict(block=0, order=[0, 1, 2], prompts=["a"], max_new=8),         # 长度不足
        dict(block=4, order=[1, 0, 2, 3], prompts=["a"], max_new=8),      # 坏块号
        dict(block=-1, order=[1, 0, 2, 3], prompts=["a"], max_new=8),     # 负块号
        dict(block=0, order=[1, 0, 2, 3], prompts=[], max_new=8),         # 无提示词
        dict(block=0, order=[1, 0, 2, 3], prompts=["a"] * 4, max_new=8),  # 超 3 条
        dict(block=0, order=[1, 0, 2, 3], prompts=[""], max_new=8),       # 空提示词
        dict(block=0, order=[1, 0, 2, 3], prompts=["a" * 33], max_new=8),  # 超 32 字符
        dict(block=0, order=[1, 0, 2, 3], prompts=["中文"], max_new=8),    # 非 ASCII
        dict(block=0, order=[1, 0, 2, 3], prompts=["a\nb"], max_new=8),   # 控制字符
        dict(block=0, order=[1, 0, 2, 3], prompts=["a"], max_new=0),      # 续写越界
        dict(block=0, order=[1, 0, 2, 3], prompts=["a"], max_new=9),
    ]
    for kw in bad:
        try:
            headperm.export_head_permutation(**kw)
        except ValueError:
            continue
        raise AssertionError(f"非法请求未被拒绝: {kw}")
    # 合法请求不抛异常
    headperm.validate_request(0, [1, 0, 2, 3], ["a"], 1)
    print(f"ok: {len(bad)} 组非法请求全部被拒绝")


def test_save_reload_roundtrip_and_tie() -> None:
    """文件可重载：权重逐位一致，绑定保留，元数据完整。"""
    block, order = 2, [3, 1, 0, 2]
    permuted = headperm.build_permuted_target(block, order)
    path = os.path.join(_TMP_EXPORT, "roundtrip.pt")
    meta = {"block": block, "order": order, "n_layers": TARGET_LAYERS,
            "n_heads": TARGET_HEADS, "d_model": 128, "vocab_size": 98,
            "max_pos": 256, "tied_weights": ["tok_emb.weight", "lm_head.weight"],
            "created_at": "test"}
    headperm.save_weights_file(permuted, path, meta)

    reloaded, loaded_meta = headperm.load_weights_file(path)
    assert loaded_meta["block"] == block and loaded_meta["order"] == order
    # 重载后绑定保留：lm_head 与 tok_emb 是同一份存储
    assert reloaded.lm_head.weight.data_ptr() == reloaded.tok_emb.weight.data_ptr()
    # 权重逐位一致
    for (n1, p1), (n2, p2) in zip(permuted.state_dict().items(),
                                  reloaded.state_dict().items()):
        assert n1 == n2 and torch.equal(p1, p2), n1
    # 改嵌入即改输出层（绑定是活的）
    with torch.no_grad():
        reloaded.tok_emb.weight[0, 0] += 1.0
    assert reloaded.lm_head.weight[0, 0] == reloaded.tok_emb.weight[0, 0]
    print("ok: 权重文件可重载，共享嵌入 ↔ LM head 绑定保留且为活绑定")


def test_export_pass_report_and_file() -> None:
    """完整导出：PASS、报告字段齐全、sha256 与文件一致、贪心序列一致。"""
    report = headperm.export_head_permutation(
        block=2, order=[2, 0, 3, 1], prompts=PROMPTS[:2], max_new=8)
    assert report["status"] == "PASS", report["fail_reasons"]
    assert report["tied_weights_preserved"] is True
    assert report["max_abs_err_nocache"] <= 1e-3
    assert report["max_abs_err_kv"] <= 1e-3
    assert report["greedy_all_match"] is True
    assert len(report["prompts"]) == 2
    for p in report["prompts"]:
        assert p["greedy_ids_original"] == p["greedy_ids_reloaded"]
        assert p["greedy_ids_original"] == p["greedy_ids_reloaded_kv"]
        assert 1 <= len(p["greedy_ids_original"]) <= 8

    # 文件存在且 sha256 与报告一致（报告与文件对应同一次提交）
    path = headperm.resolve_export_file(report["export_id"])
    assert path is not None and os.path.exists(path)
    with open(path, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    assert digest == report["sha256"]
    assert report["file_bytes"] == os.path.getsize(path)
    # 文件内元数据记录的排列与请求一致
    _, meta = headperm.load_weights_file(path)
    assert meta["block"] == 2 and meta["order"] == [2, 0, 3, 1]
    # 报告可按 export_id 取回且内容一致
    again = headperm.get_report(report["export_id"])
    assert again is not None and again["sha256"] == report["sha256"]
    print(f"ok: 导出 PASS，文件 {report['file_bytes']} 字节，"
          f"sha256={report['sha256'][:16]}…，报告↔文件一致")


def test_failed_verification_gives_no_file() -> None:
    """核对失败（阈值强制为负）→ status=FAIL、无下载信息、文件不存在。"""
    real_tol = headperm.TOLERANCE
    headperm.TOLERANCE = -1.0  # 任何误差都超限，必然失败
    try:
        report = headperm.export_head_permutation(
            block=0, order=[1, 0, 2, 3], prompts=["abc"], max_new=4)
    finally:
        headperm.TOLERANCE = real_tol
    assert report["status"] == "FAIL"
    assert report["fail_reasons"]
    assert "download_url" not in report and "sha256" not in report
    assert headperm.resolve_export_file(report["export_id"]) is None
    # 导出目录中没有该 id 的 .pt 残留
    leftover = [f for f in os.listdir(_TMP_EXPORT)
                if report["export_id"] in f and f.endswith(".pt")]
    assert leftover == []
    print("ok: 核对失败不提供文件（无下载字段、无残留文件）")


def test_original_singleton_untouched() -> None:
    """导出后进程单例权重逐位不变，原解码（旧权重）结果不变。"""
    draft, target = get_models()
    before = {n: p.clone() for n, p in target.state_dict().items()}
    ids = tokenizer.encode("the quick brown fox ")
    ref = compare(draft, target, ids, 3, 16)

    headperm.export_head_permutation(1, [3, 2, 1, 0], PROMPTS[:1], 8)
    headperm.export_head_permutation(3, [1, 2, 3, 0], PROMPTS, 8)

    after = target.state_dict()
    for n, p in before.items():
        assert torch.equal(after[n], p), n
    again = compare(draft, target, ids, 3, 16)
    assert again.baseline.new_tokens == ref.baseline.new_tokens
    assert again.match
    print("ok: 导出未触碰单例旧权重，原解码结果不变")


def test_concurrent_exports_are_consistent() -> None:
    """8 个线程并发导出不同排列：报告、文件、元数据严格一一对应。"""
    cases = [
        (0, [1, 0, 2, 3]), (1, [2, 0, 3, 1]), (2, [3, 2, 1, 0]),
        (3, [1, 2, 3, 0]), (0, [3, 0, 2, 1]), (1, [0, 2, 1, 3]),
        (2, [2, 3, 0, 1]), (3, [1, 3, 2, 0]),
    ]
    reports: list[dict] = []
    errors: list[BaseException] = []

    def run(case):
        try:
            block, order = case
            reports.append(headperm.export_head_permutation(
                block, order, PROMPTS[:2], 8))
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=run, args=(c,)) for c in cases]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, errors
    assert len(reports) == len(cases)
    by_case = {(r["block"], tuple(r["order"])): r for r in reports}
    assert len(by_case) == len(cases), "export_id/排列应一一对应"
    for case, r in by_case.items():
        assert r["status"] == "PASS", r["fail_reasons"]
        path = headperm.resolve_export_file(r["export_id"])
        assert path is not None
        with open(path, "rb") as f:
            assert hashlib.sha256(f.read()).hexdigest() == r["sha256"]
        _, meta = headperm.load_weights_file(path)
        assert (meta["block"], tuple(meta["order"])) == case
        # 文件内该块权重确实按本次提交的排列改排（与全新重排实例逐位一致）
        expect = headperm.build_permuted_target(*case)
        got, _ = headperm.load_weights_file(path)
        for p1, p2 in zip(expect.state_dict().values(), got.state_dict().values()):
            assert torch.equal(p1, p2)
    print(f"ok: {len(cases)} 路并发导出，报告/文件/元数据全部一一对应")


if __name__ == "__main__":
    test_invalid_requests_rejected()
    test_weights_actually_permuted()
    test_save_reload_roundtrip_and_tie()
    test_permutation_preserves_function()
    test_export_pass_report_and_file()
    test_failed_verification_gives_no_file()
    test_original_singleton_untouched()
    test_concurrent_exports_are_consistent()
    print("\n头重排导出全部测试通过 ✔")
