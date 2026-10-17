"""闭环编排路由的集成测试：重写修复 / L3 重新理解 / L4 转人工 / 连续错误 / 疑似参考噪声分层。

对应 scripts/run_tests.py 里的 orchestrator 路由断言（用 Fake LLM 不联网）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest  # noqa: E402

from nl2opt.config import load_config  # noqa: E402
from nl2opt.dataset import Sample  # noqa: E402
from nl2opt.gold_ir import get_gold_ir  # noqa: E402
from nl2opt.llm.fake import ScriptedProvider  # noqa: E402
from nl2opt.pipeline import run_sample  # noqa: E402


def _wrong_ir(mutate):
    """knapsack-01 的 IR dict，施加 mutate 变换后返回。"""
    correct = get_gold_ir("knapsack-01").model_dump()
    return mutate(correct)


def _sample(sid, nl_text="测试", expected=220.0):
    return Sample(id=sid, nl_text=nl_text, problem_type="milp",
                  expected={"objective_value": expected})


def test_rewrite_loop_repairs_ir():
    correct = get_gold_ir("knapsack-01").model_dump()
    wrong = json.loads(json.dumps(correct))
    for t in wrong["objective"]["expr"]["terms"]:
        if t["var"] == "x2":
            t["coef"] = 1000.0
    llm = ScriptedProvider([
        json.dumps(wrong, ensure_ascii=False),
        "建模错误：系数错误",
        json.dumps(correct, ensure_ascii=False),
    ])
    result = run_sample(_sample("test-rewrite"), llm=llm, cfg=load_config())
    assert result.success is True, result.errors
    assert result.rounds == 2
    assert result.objective_value == 220.0


def test_l3_route_reunderstands():
    correct = get_gold_ir("knapsack-01").model_dump()
    wrong = json.loads(json.dumps(correct))
    wrong["constraints"] = []  # 漏约束
    llm = ScriptedProvider([
        json.dumps(wrong, ensure_ascii=False),
        json.dumps({"layer": "L3", "advice": "漏了容量约束"}),
        json.dumps(correct, ensure_ascii=False),
    ])
    result = run_sample(_sample("test-L3"), llm=llm, cfg=load_config())
    assert result.success is True, result.errors
    assert any("L3" in (t.get("route", "")) for t in result.trajectory)


def test_l4_route_manual():
    correct = get_gold_ir("knapsack-01").model_dump()
    wrong = json.loads(json.dumps(correct))
    wrong["constraints"] = []
    llm = ScriptedProvider([
        json.dumps(wrong, ensure_ascii=False),
        json.dumps({"layer": "L4", "advice": "问题信息不足"}),
    ])
    result = run_sample(_sample("test-L4"), llm=llm, cfg=load_config())
    assert result.success is False
    assert result.status == "manual_required"
    assert any("L4 转人工" in e for e in result.errors)


def test_repeated_error_terminates_early():
    wrong = {
        "variables": [{"name": "x1", "var_type": "binary"},
                      {"name": "x2", "var_type": "binary"},
                      {"name": "x3", "var_type": "binary"}],
        "objective": {"sense": "maximize",
                      "expr": {"terms": [{"var": "x1", "coef": 0.0}], "constant": 0.0}},
        "constraints": [],
    }
    llm = ScriptedProvider([
        json.dumps(wrong, ensure_ascii=False),
        json.dumps({"layer": "L2", "advice": "改系数"}),
        json.dumps(wrong, ensure_ascii=False),
        json.dumps({"layer": "L2", "advice": "改系数"}),
        json.dumps(wrong, ensure_ascii=False),
        json.dumps({"layer": "L2", "advice": "改系数"}),
        json.dumps(wrong, ensure_ascii=False),
    ])
    result = run_sample(_sample("test-repeated"), llm=llm, cfg=load_config())
    assert result.success is False
    assert result.status == "repeated_error"
    assert result.rounds == 3  # 第 3 轮终止（不跑满 max_rounds）
    assert any("连续相同错误终止" in e for e in result.errors)


def test_suspicious_reference_flags_but_still_repairs():
    """untrusted + 目标不符：仍走一次重写，但绝不升级为通过（安全不变量）。"""
    from nl2opt import verifier
    from nl2opt.solver import solve
    ir = get_gold_ir("knapsack-01")
    sol = solve(ir)
    checks = verifier.verify(ir, sol, {"objective_value": 999.0})
    assert verifier.all_pass(checks) is False
    assert verifier.suspicious_reference(checks) is True


def test_suspicious_outcome_tiers():
    """converged 计通过（suspicious_resolved）；unresolved 不计通过（suspicious）。"""
    # 目标里加常数 +1：可行但目标值不符（用于触发 suspicious 判定）
    correct = get_gold_ir("knapsack-01").model_dump()
    wrong = json.loads(json.dumps(correct))
    expr = wrong["objective"]["expr"]
    expr["constant"] = float(expr.get("constant") or 0.0) + 1.0
    # 首轮错 → 反思 L2 → 重写回正确 → converged
    llm = ScriptedProvider([
        json.dumps(wrong, ensure_ascii=False),
        json.dumps({"layer": "L2", "advice": "去掉目标常数"}),
        json.dumps(correct, ensure_ascii=False),
    ])
    sample = Sample(id="test-converged", nl_text="测试 converged", problem_type="milp",
                    expected={"objective_value": 220.0}, trusted_reference=False)
    result = run_sample(sample, llm=llm, cfg=load_config())
    # converged：重写后达到参考值 → 计通过，status=suspicious_resolved
    assert result.status == "suspicious_resolved"
    assert result.success is True


def test_suspicious_unresolved_reports_final_round_solution():
    """unresolved 终止时，objective_value 必须反映**最终轮**的模型（与 final_ir 一致）。

    回归：旧实现在 `if suspicion is None:` 里只设置第 1 轮的 sol，后续轮次不更新，
    导致报告的 objective_value 停留在第 1 轮、与 final_ir（最后一轮）不一致。
    实测影响 6 题（NL4OPT 5 + IndustryOR 1，全部 rounds=2、status=suspicious）：
    例 industryor-005 报告 18.9566 而 final_ir 实为 18.9857。
    """
    correct = get_gold_ir("knapsack-01").model_dump()

    def with_offset(delta: float) -> dict:
        d = json.loads(json.dumps(correct))
        expr = d["objective"]["expr"]
        expr["constant"] = float(expr.get("constant") or 0.0) + delta
        return d

    llm = ScriptedProvider([
        json.dumps(with_offset(1.0), ensure_ascii=False),    # 第 1 轮：目标 221
        json.dumps({"layer": "L2", "advice": "去掉目标常数"}),
        json.dumps(with_offset(500.0), ensure_ascii=False),  # 第 2 轮：目标 720
    ])
    sample = Sample(id="test-unresolved", nl_text="测试 unresolved", problem_type="milp",
                    expected={"objective_value": 220.0}, trusted_reference=False)
    result = run_sample(sample, llm=llm, cfg=load_config())

    assert result.success is False
    assert result.status == "suspicious"
    assert result.rounds == 2
    # 关键：报告值是最后一轮的 720，而不是第 1 轮的 221
    assert result.objective_value == 720.0
    # 与 final_ir 重解一致
    from nl2opt.ir import IR
    from nl2opt.solver import solve
    assert solve(IR.model_validate(result.final_ir)).objective_value == result.objective_value
