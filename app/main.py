"""FastAPI 后端：推测解码 vs 无缓存贪心基线的对照服务。"""

from __future__ import annotations

import os
from dataclasses import asdict

from fastapi import FastAPI, Response
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import tokenizer
from .decoding import compare
from .headperm import export_head_permutation, get_report, resolve_export_file
from .model import TARGET_HEADS, TARGET_LAYERS, get_models

app = FastAPI(title="Speculative Decoding Verifier")

STATIC_DIR = os.path.join(os.path.dirname(__file__), "..", "static")

# 候选提示词（纯 ASCII，固定词表全部可编码）；启动时扫描出
# 分别触发"拒绝"和"整块接受"的两条预设。
_CANDIDATE_PROMPTS = [
    "the quick brown fox jumps over the lazy dog while ",
    "hello world, this is a tiny transformer demo of ",
    "speculative decoding uses a small draft model to ",
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa ",
    "in the beginning there was only silence and then ",
    "print(\"sum =\")  # ",
    "the answer to life the universe and everything is ",
    "one two three four five six seven eight nine ten ",
    "python java rust go c plus plus ruby swift scala ",
    "a stitch in time saves nine, and early birds get ",
    "the rain in spain stays mainly on the plain while ",
    "to be or not to be that is the question whether ",
    "x = 1\ny = 2\nz = x + y\nprint(z)\n# output is ",
    "all work and no play makes jack a dull boy, so ",
    "once upon a time in a small village lived an old ",
    "the capital of france is paris and the capital of ",
    "0 1 1 2 3 5 8 13 21 34 55 89 144 233 next number ",
    "she sells sea shells by the sea shore, the shells ",
    "machine learning models can be compressed with ",
    "alpha beta gamma delta epsilon zeta eta theta iota ",
]

# 固定预设：经实测与测试套件锁定，分别稳定触发"拒绝"与"整块接受"。
# 拒绝预设：草稿与目标在该上下文上分歧，首轮接受后持续被拒，
# 用于验证"被拒草稿不会污染后续生成"；
# 整块接受预设：每轮草稿 token 全部通过，展示目标前向节省的上限。
PRESET_REJECT = "python java rust go c plus plus ruby swift scala "
PRESET_ALL_ACCEPTED = "the quick brown fox jumps over the lazy dog while "


class DecodeRequest(BaseModel):
    prompt: str = Field(default="", max_length=200)
    k: int = Field(default=3, ge=1, le=4)
    max_new: int = Field(default=32, ge=1, le=32)


class HeadPermExportRequest(BaseModel):
    """注意力头重排导出请求：块号 + 非原顺序的 4 头排列 + 1~3 条提示词。"""
    block: int = Field(ge=0, le=TARGET_LAYERS - 1)
    order: list[int] = Field(min_length=TARGET_HEADS, max_length=TARGET_HEADS)
    prompts: list[str] = Field(min_length=1, max_length=3)
    max_new: int = Field(default=8, ge=1, le=8)


def _token_view(ids: list[int]) -> list[dict]:
    return [{"id": int(t), "text": tokenizer.render_token(int(t))} for t in ids]


def _serialize(cmp_result, prompt_text: str) -> dict:
    base, spec = cmp_result.baseline, cmp_result.speculative
    return {
        "prompt": prompt_text,
        "prompt_ids": base.prompt_ids,
        "match": cmp_result.match,
        "status": "PASS" if cmp_result.match else "FAIL",
        "first_mismatch_index": cmp_result.first_mismatch_index,
        "baseline": {
            "new_tokens": _token_view(base.new_tokens),
            "text": tokenizer.decode(base.new_tokens),
            "target_forward_count": base.target_forward_count,
            "stopped_by_eos": base.stopped_by_eos,
        },
        "speculative": {
            "new_tokens": _token_view(spec.new_tokens),
            "text": tokenizer.decode(spec.new_tokens),
            "target_forward_count": spec.target_forward_count,
            "draft_forward_count": spec.draft_forward_count,
            "stopped_by_eos": spec.stopped_by_eos,
            "rounds": [asdict(r) | {
                "draft_tokens_view": _token_view(r.draft_tokens),
                "target_predictions_view": _token_view(r.target_predictions),
                "target_tokens_view": _token_view(r.target_tokens),
            } for r in spec.rounds],
        },
        "savings": {
            "baseline_forwards": base.target_forward_count,
            "spec_forwards": spec.target_forward_count,
            "saved": base.target_forward_count - spec.target_forward_count,
        },
    }


@app.get("/api/health")
def health() -> dict:
    draft, target = get_models()
    return {
        "ok": True,
        "vocab_size": tokenizer.VOCAB_SIZE,
        "target_layers": target.n_layers,
        "draft_layers": draft.n_layers,
    }


@app.post("/api/decode")
def decode(req: DecodeRequest) -> JSONResponse:
    prompt_ids = tokenizer.encode(req.prompt)
    if not prompt_ids:
        return JSONResponse(
            status_code=400,
            content={"error": "提示词编码后为空，请输入可打印 ASCII 字符。"},
        )
    draft, target = get_models()
    result = compare(draft, target, prompt_ids, req.k, req.max_new)
    return JSONResponse(_serialize(result, req.prompt))


@app.get("/api/presets")
def presets() -> dict:
    """返回固定的两条预设（由测试套件锁定其判定类型，避免请求时全量扫描）。"""
    return {"reject": PRESET_REJECT, "all_accepted": PRESET_ALL_ACCEPTED}


# ---- 注意力头重排导出（设备端指定头顺序）----


@app.post("/api/headperm/export")
def headperm_export(req: HeadPermExportRequest) -> JSONResponse:
    """重排指定块的注意力头并产出可重载权重文件。

    核对（原模型 vs 从文件重载的模型：无缓存与 KV 增量 logits 误差 ≤1e-3、
    贪心续写逐 token 一致、权重绑定保留）全部通过才提供下载；
    失败仅返回报告，不留文件。
    """
    try:
        report = export_head_permutation(req.block, req.order, req.prompts, req.max_new)
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    return JSONResponse(report)


@app.get("/api/headperm/{export_id}")
def headperm_report(export_id: str) -> JSONResponse:
    """按 export_id 取回某次提交的报告（与下载文件一一对应）。"""
    report = get_report(export_id)
    if report is None:
        return JSONResponse(status_code=404, content={"error": "报告不存在。"})
    return JSONResponse(report)


@app.get("/api/headperm/{export_id}/download")
def headperm_download(export_id: str) -> Response:
    """下载某次核对通过的权重文件（核对失败的提交没有文件，返回 404）。"""
    path = resolve_export_file(export_id)
    if path is None:
        return JSONResponse(status_code=404,
                            content={"error": "文件不存在（核对未通过或已清理）。"})
    report = get_report(export_id) or {}
    filename = report.get("download_name", os.path.basename(path))
    return FileResponse(path, media_type="application/octet-stream", filename=filename)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
