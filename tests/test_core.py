"""IR 与求解器、验证器、黄金 IR 的单元/集成测试。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest  # noqa: E402

from nl2opt.gold_ir import get_gold_ir  # noqa: E402
from nl2opt.ir import IR, Constraint, LinearExpr, Objective, Sense, Term, VarType, Variable  # noqa: E402
from nl2opt.solver import solve  # noqa: E402
from nl2opt import verifier  # noqa: E402


def _t(var, coef):
    return Term(var=var, coef=coef)


def _expr(terms, constant=0.0):
    return LinearExpr(terms=terms, constant=constant)


def test_ir_duplicate_variable_rejected():
    with pytest.raises(Exception):
        IR(
            variables=[
                Variable(name="x", var_type=VarType.continuous),
                Variable(name="x", var_type=VarType.continuous),
            ],
            objective=Objective(sense=Sense.maximize, expr=_expr([_t("x", 1)])),
            constraints=[],
        )


def test_ir_undeclared_variable_rejected():
    with pytest.raises(Exception):
        IR(
            variables=[Variable(name="x", var_type=VarType.continuous)],
            objective=Objective(sense=Sense.maximize, expr=_expr([_t("y", 1)])),
            constraints=[],
        )


@pytest.mark.parametrize(
    "sid,expected_obj",
    [
        ("lp-01", 158.0),
        ("knapsack-01", 220.0),
        ("knapsack-02", 90.0),
        ("transport-01", 420.0),
        ("assignment-01", 5.0),
        ("integer-01", 40.0),
        ("diet-01", 15.6),
    ],
)
def test_gold_ir_solves_to_expected(sid, expected_obj):
    ir = get_gold_ir(sid)
    assert ir is not None
    sol = solve(ir)
    assert sol.success, sol.message
    assert sol.objective_value == pytest.approx(expected_obj, abs=1e-4)


def test_infeasible():
    ir = get_gold_ir("infeasible-01")
    sol = solve(ir)
    assert sol.status == "infeasible"


def test_unbounded():
    ir = get_gold_ir("unbounded-01")
    sol = solve(ir)
    assert sol.status == "unbounded"


def test_verifier_detects_violation():
    ir = get_gold_ir("knapsack-01")
    sol = solve(ir)
    # 人为制造违反：把 x1 设为 1（重量 10+20+30=60>50 违反容量）
    sol.variables["x1"] = 1.0
    res = verifier.check_feasibility(ir, sol)
    assert res.ok is False
    assert "capacity" in res.message
