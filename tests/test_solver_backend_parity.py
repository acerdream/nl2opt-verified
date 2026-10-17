"""求解后端对拍与 codegen 正确性的单元测试（双后端语义一致性 + 变量名安全）。

对应 scripts/run_tests.py 里的后端对拍 / codegen 断言。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest  # noqa: E402

from nl2opt import codegen  # noqa: E402
from nl2opt.gold_ir import get_gold_ir  # noqa: E402
from nl2opt.ir import (  # noqa: E402
    IR, Constraint, LinearExpr, Objective, Sense, Term, VarType, Variable,
)
from nl2opt.solver import solve  # noqa: E402


def test_objective_constant_counted():
    """目标常数必须计入本地求解（与 codegen/docker 一致）。"""
    ir = IR(
        variables=[Variable(name="x", var_type=VarType.continuous, lower=0, upper=10)],
        objective=Objective(sense=Sense.maximize, expr=LinearExpr(
            terms=[Term(var="x", coef=1.0)], constant=7.0)),
        constraints=[Constraint(name="c1", sense=Sense.le,
                                lhs=LinearExpr(terms=[Term(var="x", coef=1.0)]), rhs=10.0)],
    )
    sol = solve(ir)
    assert sol.success
    # max x=10 → 目标 = 10 + 7 = 17（丢弃常数会得 10）
    assert sol.objective_value == pytest.approx(17.0)


def test_codegen_name_safety():
    """非法 Python 标识符变量名也能编译，结果名映射回原始 IR 名。"""
    for name in ("x-y", "class", "1x", "销量"):
        ir = IR(
            variables=[Variable(name=name, var_type=VarType.continuous, lower=0)],
            objective=Objective(sense=Sense.maximize,
                                expr=LinearExpr(terms=[Term(var=name, coef=1.0)])),
            constraints=[Constraint(name="c1", sense=Sense.le,
                                    lhs=LinearExpr(terms=[Term(var=name, coef=1.0)]),
                                    rhs=10.0)],
        )
        code = codegen.generate_code(ir)
        compile(code, "<codegen-name-safety>", "exec")  # 不抛 SyntaxError 即通过
        assert f"= LpVariable('{name}'" in code, f"{name} 的显示名丢失"
        assert f"{name} =" not in code, f"{name} 仍被当 Python 名使用"


def test_codegen_result_maps_back_to_ir_names():
    """结果回填显式用「原始实例名 → 安全表达式」，不依赖 PuLP 可能改写的 name。"""
    ir = IR(
        variables=[
            Variable(name="x-y", var_type=VarType.continuous, lower=0),
            Variable(name="start", var_type=VarType.integer, index_set=["J1"], lower=0),
        ],
        objective=Objective(sense=Sense.maximize, expr=LinearExpr(terms=[
            Term(var="x-y", coef=1.0), Term(var="start_J1", coef=1.0)])),
    )
    code = codegen.generate_code(ir)
    assert "'x-y'" in code and "'start_J1'" in code
    assert "_v0" in code and "_v1['J1']" in code
    assert "prob.variables()" not in code  # 旧实现依赖 PuLP 可能改写的 name


def test_codegen_emits_executable_code():
    code = codegen.generate_code(get_gold_ir("knapsack-01"))
    compile(code, "<codegen>", "exec")
    assert "from pulp import" in code
    assert "__NL2OPT_RESULT__" in code
    assert "LpVariable('x1'" in code


def test_codegen_index_map():
    """索引变量族（整数索引）→ dict 字符串键下标访问。"""
    ir = IR(
        variables=[Variable(name="y", var_type=VarType.integer, index_set=[0, 1])],
        objective=Objective(
            sense=Sense.maximize,
            expr=LinearExpr(terms=[Term(var="y_0", coef=2), Term(var="y_1", coef=3)]),
        ),
        constraints=[
            Constraint(
                name="c1", sense=Sense.le,
                lhs=LinearExpr(terms=[Term(var="y_0", coef=1), Term(var="y_1", coef=1)]),
                rhs=4,
            )
        ],
    )
    code = codegen.generate_code(ir)
    assert "_v0['0']" in code and "_v0['1']" in code
