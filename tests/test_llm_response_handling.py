"""统一 LLM 响应保护层与答案防泄露的单元测试。

对应 scripts/run_tests.py 里的截断/空输出/解析失败保护、答案泄露剥离断言。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest  # noqa: E402

from nl2opt.llm import LLMError, LLMMessage, LLMProvider, LLMResponse  # noqa: E402
from nl2opt.pipeline.reflect import ReflectionParseError, _parse_reflection  # noqa: E402
from nl2opt.pipeline.rewrite import rewrite  # noqa: E402
from nl2opt.pipeline.understand import understand  # noqa: E402


class _FixedLLM(LLMProvider):
    """可控假 Provider：可指定 finish_reason（截断）与固定文本。"""

    def __init__(self, text="", finish_reason=None):
        self._text = text
        self._finish = finish_reason

    def complete(self, messages, **kwargs):
        return LLMResponse(text=self._text, finish_reason=self._finish)


def test_understand_rejects_truncated_response():
    class TruncatingProvider(LLMProvider):
        def complete(self, messages, **kwargs):
            return LLMResponse(
                text='{"variables": [{"name": "y", "var_type": "integer"',
                usage={}, finish_reason="length",
            )

    with pytest.raises(LLMError, match="截断"):
        understand("测试截断", TruncatingProvider())


def test_rewrite_rejects_truncated_response():
    from nl2opt.ir import IR, LinearExpr, Objective, Sense
    ir = IR(variables=[], objective=Objective(sense=Sense.minimize, expr=LinearExpr()))
    with pytest.raises(LLMError, match="rewrite.*截断"):
        rewrite("测试题", ir, "err", "advice", _FixedLLM(text='{"x": ', finish_reason="length"))


def test_reflect_rejects_empty_response():
    from nl2opt.pipeline.reflect import reflect
    with pytest.raises(LLMError, match="reflect.*空"):
        reflect("测试题", "{}", "err", _FixedLLM(text="   ", finish_reason="stop"))


def test_reflect_parse_failure_does_not_default_to_l2():
    # 响应完整但 JSON 非法且无关键词 → ReflectionParseError，而非默认 L2
    with pytest.raises(ReflectionParseError):
        _parse_reflection("这段文字没有任何层级信号")


def test_reflect_keyword_fallback_still_works():
    r = _parse_reflection("这是建模错误：rhs 应为 100")
    assert r.category == "L2"


def test_reflect_json_parses():
    r = _parse_reflection('{"layer": "L3", "advice": "漏了约束"}')
    assert r.category == "L3" and r.advice == "漏了约束"


def test_no_answer_value_leak_into_prompt():
    """check_optimality 失败 message 不得含参考答案数值（污染 reflect/rewrite prompt）。"""
    from nl2opt import verifier
    from nl2opt.solver import SolverResult
    sol = SolverResult(status="optimal", objective_value=221.0, variables={})
    r = verifier.check_optimality(sol, 220.0, 1e-4)
    assert r.ok is False
    assert "220" not in r.message  # message 不含期望值
    assert r.detail.get("expected") == 220.0  # 数值保留在 detail，不进 prompt
