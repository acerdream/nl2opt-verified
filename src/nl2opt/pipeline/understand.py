"""理解环节：自然语言 → IR。调用 LLM（DeepSeek）抽取结构化中间表示。

提示词**分层**：常驻核心 + 按题面命中的领域卡片（见 `prompts.py` 的分层理由）。
选中的卡片会写入 `meta["cards"]`，使失败可归因到「没选中卡片」还是「选中了也没做对」。
"""
from __future__ import annotations

from ..ir import IR, ir_from_dict
from ..llm import LLMMessage, LLMProvider, complete_structured, extract_json
from .prompts import SYSTEM_PROMPT, build_system_prompt, prompt_stats  # noqa: F401

__all__ = ["understand", "SYSTEM_PROMPT", "build_system_prompt", "prompt_stats"]




def understand(text: str, llm: LLMProvider, meta: dict | None = None) -> IR:
    """自然语言 → IR（调用 LLM）。

    提示词按题面分层组装（核心 + 命中的领域卡片）。传入 `meta` 时写入
    `meta["cards"]` / `meta["prompt_chars"]`，便于把失败归因到卡片选择。

    容错策略：
    - 输出被 max_tokens 截断（finish_reason=length）或 content 为空 → 降 reasoning_effort
      重试一次（思考型模型常见陷阱：reasoning 吃光输出预算，正文 JSON 被截断）；
    - 仍被截断 → 抛带指引的错误。**不**把截断断片送去解析：extract_json 的宽松兜底
      会把 `{'name': 'y', ...}` 这类碎片解析成畸形 dict，把「预算不足」误报成「IR 校验失败」。

    截断/空检测统一走 `complete_structured`（共享响应保护层），与 rewrite/reflect 一致。
    """
    stats = prompt_stats(text)
    if meta is not None:
        meta.update(stats)
    messages = [
        LLMMessage(role="system", content=build_system_prompt(text)),
        LLMMessage(role="user", content=text),
    ]
    resp = complete_structured(llm, messages, stage="understand")

    data = extract_json(resp.text)
    # 解析器层容错：归一化 IR dict（== → = 等）
    data = _normalize_ir_dict(data)
    return ir_from_dict(data)


def _normalize_ir_dict(data: dict) -> dict:
    """对 LLM 输出的 IR dict 做健壮性归一化，避免因细微格式问题失败。

    处理：
    - 约束 sense 的 "==" → "="
    - 目标 sense 的 "max" / "min" → "maximize" / "minimize"
    """
    # 约束 sense 归一化
    for con in data.get("constraints", []):
        if isinstance(con, dict):
            sense = con.get("sense")
            if sense == "==":
                con["sense"] = "="
    # 目标 sense 归一化
    obj = data.get("objective")
    if isinstance(obj, dict):
        sense = obj.get("sense")
        if sense in ("max", "maximum"):
            obj["sense"] = "maximize"
        elif sense in ("min", "minimum"):
            obj["sense"] = "minimize"
    return data
