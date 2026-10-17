"""重写环节：依据反思建议，把上一轮 IR 修正为新 IR。

输入：原始自然语言 + 上一轮 IR + 验证失败原因 + 反思建议
输出：修正后的 IR（只改出错处，尽量保持结构稳定）
"""
from __future__ import annotations

import json

from ..ir import IR, ir_from_dict
from ..llm import LLMMessage, LLMProvider, complete_structured, extract_json

REWRITE_PROMPT = """你是一名数学建模专家。下面是一个自然语言线性/整数规划问题、上一轮建模得到的 IR（JSON）、以及验证失败的原因和诊断建议。

请**修正 IR**，使其满足问题要求。规则：
1. 只输出修正后的完整 IR JSON，不要输出解释文字。
2. 尽量保持变量名与约束结构稳定，只改出错的地方。
3. 严格遵循 IR Schema（variables / objective / constraints）。
4. 变量类型：continuous / integer / binary；sense 用 minimize / maximize（目标）与 "<=" / ">=" / "="（约束）。

## 自然语言问题
{nl_text}

## 上一轮 IR（JSON）
{ir_json}

## 验证失败原因
{errors}

## 诊断建议
{advice}

现在请输出修正后的 IR JSON："""


def rewrite(
    nl_text: str,
    ir: IR,
    errors: str,
    advice: str,
    llm: LLMProvider,
) -> IR:
    """调用 LLM 重写 IR。失败时抛异常，由编排器决定终止或继续。"""
    prompt = REWRITE_PROMPT.format(
        nl_text=nl_text,
        ir_json=json.dumps(ir.model_dump(), ensure_ascii=False),
        errors=errors,
        advice=advice,
    )
    messages = [
        LLMMessage(role="system", content="你是数学建模专家，负责修正优化模型。"),
        LLMMessage(role="user", content=prompt),
    ]
    # 截断/空输出经共享响应保护层拦截（与 understand 一致），不再把断片送去解析
    resp = complete_structured(llm, messages, stage="rewrite")
    data = extract_json(resp.text)
    return ir_from_dict(data)
