"""验证器解向量完整性与数值/目标校验的单元测试。

对应 scripts/run_tests.py 里的 verifier 断言（缺/多变量、NaN/Inf、容差解耦、
目标退化、系数依据分级）。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest  # noqa: E402

from nl2opt import verifier  # noqa: E402
from nl2opt.config import load_config  # noqa: E402
from nl2opt.gold_ir import get_gold_ir  # noqa: E402
from nl2opt.ir import (  # noqa: E402
    IR, Constraint, LinearExpr, Objective, Sense, Term, VarType, Variable,
)
from nl2opt.solver import SolverResult, solve  # noqa: E402


def _mk_ir():
    """两个变量：x（下界 1）、y（0/1）；约束 x + y <= 3。"""
    return IR(
        variables=[
            Variable(name="x", var_type=VarType.continuous, lower=1.0),
            Variable(name="y", var_type=VarType.binary),
        ],
        objective=Objective(sense=Sense.maximize, expr=LinearExpr(terms=[
            Term(var="x", coef=1.0), Term(var="y", coef=1.0)])),
        constraints=[Constraint(name="c1", sense=Sense.le, lhs=LinearExpr(terms=[
            Term(var="x", coef=1.0), Term(var="y", coef=1.0)]), rhs=3.0)],
    )


def test_verifier_rejects_missing_declared_variable():
    ir = _mk_ir()
    sol = SolverResult(status="optimal", objective_value=0.0, variables={"y": 0.0})
    r = verifier.check_feasibility(ir, sol)
    assert r.ok is False and "缺少" in r.message


def test_verifier_rejects_nan_solution_value():
    ir = _mk_ir()
    sol = SolverResult(status="optimal", objective_value=0.0,
                       variables={"x": 1.0, "y": float("nan")})
    r = verifier.check_feasibility(ir, sol)
    assert r.ok is False and "非有限" in r.message


def test_verifier_rejects_inf_solution_value():
    ir = _mk_ir()
    sol = SolverResult(status="optimal", objective_value=0.0,
                       variables={"x": 1.0, "y": float("inf")})
    r = verifier.check_feasibility(ir, sol)
    assert r.ok is False and "非有限" in r.message


def test_verifier_rejects_extra_solution_variable():
    ir = _mk_ir()
    sol = SolverResult(status="optimal", objective_value=0.0,
                       variables={"x": 1.0, "y": 0.0, "z": 5.0})
    r = verifier.check_feasibility(ir, sol)
    assert r.ok is False and "未声明" in r.message


def test_verifier_accepts_complete_solution():
    ir = _mk_ir()
    sol = SolverResult(status="optimal", objective_value=1.0, variables={"x": 1.0, "y": 0.0})
    assert verifier.check_feasibility(ir, sol).ok is True


def test_verifier_detects_violation():
    ir = get_gold_ir("knapsack-01")
    sol = solve(ir)
    sol.variables["x1"] = 1.0
    r = verifier.check_feasibility(ir, sol)
    assert r.ok is False and "capacity" in r.message


def test_objective_tolerance_decoupled_and_relative():
    cfg = load_config()
    obj_tol = float(cfg.pipeline.get("objective_tolerance", 1e-4))
    flaky = SolverResult(status="optimal", objective_value=1.9999989999999999, variables={})
    assert verifier.check_optimality(flaky, 2.0, obj_tol).ok is True  # 浮点抖动不误判
    assert verifier.check_optimality(flaky, 3.0, obj_tol).ok is False  # 真实错误仍拦下

    big = SolverResult(status="optimal", objective_value=1_000_000.0001, variables={})
    assert verifier.check_optimality(big, 1_000_000.0, obj_tol).ok is True


def test_degenerate_objective_rejected():
    def mk(terms):
        return IR(
            variables=[Variable(name="t", var_type=VarType.continuous, lower=0)],
            objective=Objective(sense=Sense.minimize, expr=LinearExpr(terms=terms)),
            constraints=[Constraint(name="c1", sense=Sense.ge,
                                    lhs=LinearExpr(terms=[Term(var="t", coef=1)]), rhs=1)],
        )

    assert verifier.check_objective_defined(mk([]), {}).ok is False       # 空目标
    assert verifier.check_objective_defined(mk([Term(var="t", coef=0.0)]), {}).ok is False  # 系数全 0
    assert verifier.check_objective_defined(mk([Term(var="t", coef=3.0)]), {}).ok is True


def _mk_weighted_ir(coefs):
    return IR(
        variables=[Variable(name=f"t{i}", var_type=VarType.continuous, lower=0)
                   for i in range(len(coefs))],
        objective=Objective(sense=Sense.minimize, expr=LinearExpr(
            terms=[Term(var=f"t{i}", coef=c) for i, c in enumerate(coefs)])),
        constraints=[Constraint(name="c1", sense=Sense.ge,
                                lhs=LinearExpr(terms=[Term(var="t0", coef=1)]), rhs=0)],
    )


def test_coefficient_grounding_strong_evidence_hard_fail():
    from nl2opt.verifier import check_coefficient_grounding, structured_weight_map
    real = ('{"jobs": [{"job_id": "J1", "duration": 5, "weight": 3}, '
            '{"job_id": "J2", "duration": 2, "weight": 1}]}')
    assert structured_weight_map(real) == {"J1": 3.0, "J2": 1.0}
    r = check_coefficient_grounding(_mk_weighted_ir([1.0, 1.0]), real, {})
    assert r.ok is False and r.blocking is True
    assert "强证据" in r.message
    # 系数有区分 → 通过
    assert check_coefficient_grounding(_mk_weighted_ir([3.0, 1.0]), real, {}).ok is True


def test_coefficient_grounding_weak_evidence_warns_only():
    from nl2opt.verifier import check_coefficient_grounding, structured_weight_map
    weak = ("Each job has a priority: the first job's weight matters most, "
            "priority 5 for urgent orders, priority 2 otherwise. "
            "Minimize total weighted completion.")
    assert structured_weight_map(weak) is None
    r = check_coefficient_grounding(_mk_weighted_ir([1.0, 1.0]), weak, {})
    assert r.ok is False and r.blocking is False  # 仅告警，不阻断
    assert "弱证据" in r.message


def test_coefficient_grounding_exempt_no_weights():
    from nl2opt.verifier import check_coefficient_grounding
    fake = ("A shop has machines M1. Jobs: J1 dur 3 rel 2 due 7, J2 dur 1 rel 0 due 9. "
            "Minimize weighted tardiness. Do not overlap a machine.")
    assert check_coefficient_grounding(_mk_weighted_ir([1.0, 1.0]), fake, {}).ok is True


# ============ objective_consistency：由 IR 重算目标值，闭合「执行→验证」边界 ============


def _simple_ir():
    """max x + 2y，s.t. x <= 4, y <= 1；x、y 非负连续。"""
    return IR(
        variables=[
            Variable(name="x", var_type=VarType.continuous, lower=0),
            Variable(name="y", var_type=VarType.continuous, lower=0),
        ],
        objective=Objective(sense=Sense.maximize, expr=LinearExpr(terms=[
            Term(var="x", coef=1.0), Term(var="y", coef=2.0)], constant=0.0)),
        constraints=[
            Constraint(name="c1", sense=Sense.le, lhs=LinearExpr(terms=[Term(var="x", coef=1.0)]), rhs=4.0),
            Constraint(name="c2", sense=Sense.le, lhs=LinearExpr(terms=[Term(var="y", coef=1.0)]), rhs=1.0),
        ],
    )


def test_objective_consistency_rejects_forged_objective():
    """可行解 + 伪造 objective（报告值 ≠ IR 重算值）必须失败。"""
    from nl2opt.verifier import check_objective_consistency
    ir = _simple_ir()
    # 真实解：x=4, y=1 → 目标 = 4 + 2*1 = 6；伪造报告成 999（恰好等于期望）
    sol = SolverResult(status="optimal", objective_value=999.0, variables={"x": 4.0, "y": 1.0})
    r = check_objective_consistency(ir, sol)
    assert r.ok is False and r.blocking is True
    assert "重算" in r.message or "不符" in r.message


def test_objective_consistency_accepts_correct_objective():
    from nl2opt.verifier import check_objective_consistency
    ir = _simple_ir()
    sol = SolverResult(status="optimal", objective_value=6.0, variables={"x": 4.0, "y": 1.0})
    assert check_objective_consistency(ir, sol).ok is True


def test_objective_consistency_rejects_nan_objective():
    from nl2opt.verifier import check_objective_consistency
    ir = _simple_ir()
    sol = SolverResult(status="optimal", objective_value=float("nan"), variables={"x": 4.0, "y": 1.0})
    r = check_objective_consistency(ir, sol)
    assert r.ok is False and "非有限" in r.message


def test_objective_consistency_rejects_inf_objective():
    from nl2opt.verifier import check_objective_consistency
    ir = _simple_ir()
    sol = SolverResult(status="optimal", objective_value=float("inf"), variables={"x": 4.0, "y": 1.0})
    r = check_objective_consistency(ir, sol)
    assert r.ok is False and "非有限" in r.message


def test_optimality_rejects_nan_expected():
    """expected=NaN 时，旧实现 abs(nan-nan)>tol 为 False 会误判一致；现须失败。"""
    sol = SolverResult(status="optimal", objective_value=220.0, variables={})
    r = verifier.check_optimality(sol, float("nan"), 1e-4)
    assert r.ok is False and "非有限" in r.message


def test_optimality_rejects_nan_objective_value():
    sol = SolverResult(status="optimal", objective_value=float("nan"), variables={})
    r = verifier.check_optimality(sol, 220.0, 1e-4)
    assert r.ok is False and "非有限" in r.message


def test_optimality_uses_answer_variable_when_specified():
    """目标规划题：参考答案是偏差变量的值，不是目标函数值。

    IndustryOR-029：objective=10000*d2_minus+d3_minus=10,003,000，而参考答案 2500
    是 d3_minus 的值。指定 answer_variable 后应比较该变量。
    """
    sol = SolverResult(status="optimal", objective_value=10_003_000.0,
                       variables={"d2_minus": 1000.0, "d3_minus": 2500.0})
    # 不指定 → 拿目标函数值比，失败
    assert verifier.check_optimality(sol, 2500.0, 1e-4).ok is False
    # 指定 → 拿 d3_minus=2500 比，通过
    r = verifier.check_optimality(sol, 2500.0, 1e-4, answer_var="d3_minus")
    assert r.ok is True, r.message
    assert "d3_minus" in r.message
    # 值不符仍要拦下
    sol_bad = SolverResult(status="optimal", objective_value=10_003_500.0,
                           variables={"d2_minus": 1000.0, "d3_minus": 3000.0})
    r2 = verifier.check_optimality(sol_bad, 2500.0, 1e-4, answer_var="d3_minus")
    assert r2.ok is False
    assert r2.detail["answer_variable"] == "d3_minus"
    assert r2.detail["actual"] == 3000.0
    # message 里不得出现参考值（防答案泄漏进重写 prompt）
    assert "2500" not in r2.message


def test_optimality_missing_answer_variable_reports_clearly():
    """解里没有答案变量时，给出可操作的错误（提示偏差变量命名约定）。"""
    sol = SolverResult(status="optimal", objective_value=10003000.0,
                       variables={"p3_unmet": 3000.0})
    r = verifier.check_optimality(sol, 2500.0, 1e-4, answer_var="d3_minus")
    assert r.ok is False
    assert "不在求解结果中" in r.message
    assert "d3_minus" in r.message
    assert "d{k}_minus" in r.message          # 指向命名约定
    assert r.detail["answer_variable"] == "d3_minus"


def test_consistency_applies_when_reference_none():
    """参考答案为 None 时，一致性检查仍须生效（不依赖参考答案）。"""
    from nl2opt.verifier import check_objective_consistency
    ir = _simple_ir()
    # 伪造 objective，且 expected 无 objective_value → 一致性检查必须拦下
    sol = SolverResult(status="optimal", objective_value=999.0, variables={"x": 4.0, "y": 1.0})
    r = check_objective_consistency(ir, sol)
    assert r.ok is False and r.blocking is True
    # verify() 接线：expected={}（无 objective_value）时，all_pass 仍因一致性失败
    checks = verifier.verify(ir, sol, {})
    assert verifier.all_pass(checks) is False
    assert any(c.kind == "objective_consistency" and not c.ok for c in checks)


def test_verify_forged_envelope_fails_end_to_end():
    """端到端：可行解 + 伪造 objective，即使等于期望，verify() 也必须失败。"""
    ir = _simple_ir()
    sol = SolverResult(status="optimal", objective_value=6.0, variables={"x": 4.0, "y": 1.0})
    # 先把真实解跑通：期望 6.0 应通过
    assert verifier.all_pass(verifier.verify(ir, sol, {"objective_value": 6.0})) is True
    # 伪造 objective=6.0 但 variables 实际对应 4+2*0=4（把 y 改成 0）
    forged = SolverResult(status="optimal", objective_value=6.0, variables={"x": 4.0, "y": 0.0})
    checks = verifier.verify(ir, forged, {"objective_value": 6.0})
    assert verifier.all_pass(checks) is False
    assert any(c.kind == "objective_consistency" and not c.ok for c in checks)


# ============ 方案 A：诊断上下文（紧约束 + 卡边界，进 detail 不进 message） ============


def test_optimality_diagnostic_in_detail_not_message():
    """诊断（紧约束）应进 detail 落盘，但**不进 message**（实测对 LLM 修复无效 + 噪声）。"""
    ir = get_gold_ir("lp-01")
    sol = solve(ir)  # 真实最优 x1=42, x2=16，两条约束都紧
    forged = SolverResult(status="optimal", objective_value=999.0, variables=sol.variables)
    r = verifier.check_optimality(forged, 158.0, 1e-4, ir)
    assert r.ok is False
    # message 保持简洁（不污染 reflect/rewrite prompt），且不泄露期望值
    assert "诊断" not in r.message
    assert "158" not in r.message
    # 诊断在 detail 里（落盘/人复核可见）
    assert "diagnostic" in r.detail
    assert "紧约束" in r.detail["diagnostic"] and "c1" in r.detail["diagnostic"] and "c2" in r.detail["diagnostic"]


def test_diagnostic_bound_variable_included_for_continuous():
    """连续/整数变量卡边界应被诊断；binary 卡 0/1 应被排除（平凡事实）。"""
    from nl2opt.verifier import _diagnostic_context
    from nl2opt.ir import IR, Constraint, LinearExpr, Objective, Sense, Term, VarType, Variable
    ir = IR(
        variables=[
            Variable(name="x", var_type=VarType.continuous, lower=0, upper=10),
            Variable(name="b", var_type=VarType.binary),
        ],
        objective=Objective(sense=Sense.maximize, expr=LinearExpr(terms=[
            Term(var="x", coef=1.0), Term(var="b", coef=1.0)])),
        constraints=[Constraint(name="c1", sense=Sense.le, lhs=LinearExpr(terms=[
            Term(var="x", coef=1.0), Term(var="b", coef=1.0)]), rhs=10.0)],
    )
    sol = SolverResult(status="optimal", objective_value=1.0,
                       variables={"x": 0.0, "b": 1.0})
    ctx = _diagnostic_context(ir, sol)
    # x 卡在下界 0 → 应诊断；b 是 binary（=1 卡上界）→ 应排除
    assert "x=下界0" in ctx
    assert "b" not in ctx
