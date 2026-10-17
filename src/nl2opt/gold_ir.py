"""黄金 IR 基准：为每道内置样例提供手工标注的“正确答案”IR。

作用：
1. 无 LLM / 无网络时，确定性链路（建模→生成→执行→验证）能完整跑通；
2. 作为评测基准，将来对比 LLM 理解的正确率（变量名可不同，语义需一致）。

注意：黄金 IR 的变量实例名与 dataset 中 expected.variables 的键保持一致。
"""
from __future__ import annotations

from .ir import Constraint, IR, LinearExpr, Objective, Sense, Term, VarType, Variable


def _t(var: str, coef: float) -> Term:
    return Term(var=var, coef=coef)


def _expr(terms: list[Term], constant: float = 0.0) -> LinearExpr:
    return LinearExpr(terms=terms, constant=constant)


def _var(name: str, vt: VarType, lower: float | None = 0.0, upper: float | None = None) -> Variable:
    return Variable(name=name, var_type=vt, lower=lower, upper=upper)


GOLD_IR: dict[str, IR] = {
    # 连续 LP：生产计划
    "lp-01": IR(
        variables=[_var("x1", VarType.continuous), _var("x2", VarType.continuous)],
        objective=Objective(sense=Sense.maximize, expr=_expr([_t("x1", 3), _t("x2", 2)])),
        constraints=[
            Constraint(name="c1", sense=Sense.le, lhs=_expr([_t("x1", 2), _t("x2", 1)]), rhs=100),
            Constraint(name="c2", sense=Sense.le, lhs=_expr([_t("x1", 1), _t("x2", 3)]), rhs=90),
        ],
    ),
    # 0-1 背包
    "knapsack-01": IR(
        variables=[_var("x1", VarType.binary), _var("x2", VarType.binary), _var("x3", VarType.binary)],
        objective=Objective(sense=Sense.maximize, expr=_expr([_t("x1", 60), _t("x2", 100), _t("x3", 120)])),
        constraints=[
            Constraint(name="capacity", sense=Sense.le, lhs=_expr([_t("x1", 10), _t("x2", 20), _t("x3", 30)]), rhs=50),
        ],
    ),
    "knapsack-02": IR(
        variables=[_var("x1", VarType.binary), _var("x2", VarType.binary), _var("x3", VarType.binary), _var("x4", VarType.binary)],
        objective=Objective(sense=Sense.maximize, expr=_expr([_t("x1", 10), _t("x2", 40), _t("x3", 30), _t("x4", 50)])),
        constraints=[
            Constraint(name="capacity", sense=Sense.le, lhs=_expr([_t("x1", 5), _t("x2", 4), _t("x3", 6), _t("x4", 3)]), rhs=10),
        ],
    ),
    # 运输问题（连续 LP）
    "transport-01": IR(
        variables=[
            _var("x_A_X", VarType.continuous), _var("x_A_Y", VarType.continuous), _var("x_A_Z", VarType.continuous),
            _var("x_B_X", VarType.continuous), _var("x_B_Y", VarType.continuous), _var("x_B_Z", VarType.continuous),
        ],
        objective=Objective(
            sense=Sense.minimize,
            expr=_expr([
                _t("x_A_X", 8), _t("x_A_Y", 6), _t("x_A_Z", 10),
                _t("x_B_X", 9), _t("x_B_Y", 12), _t("x_B_Z", 13),
            ]),
        ),
        constraints=[
            # 供应（A 出 30，B 出 20）
            Constraint(name="supply_A", sense=Sense.eq, lhs=_expr([_t("x_A_X", 1), _t("x_A_Y", 1), _t("x_A_Z", 1)]), rhs=30),
            Constraint(name="supply_B", sense=Sense.eq, lhs=_expr([_t("x_B_X", 1), _t("x_B_Y", 1), _t("x_B_Z", 1)]), rhs=20),
            # 需求（X=20, Y=15, Z=15）
            Constraint(name="demand_X", sense=Sense.eq, lhs=_expr([_t("x_A_X", 1), _t("x_B_X", 1)]), rhs=20),
            Constraint(name="demand_Y", sense=Sense.eq, lhs=_expr([_t("x_A_Y", 1), _t("x_B_Y", 1)]), rhs=15),
            Constraint(name="demand_Z", sense=Sense.eq, lhs=_expr([_t("x_A_Z", 1), _t("x_B_Z", 1)]), rhs=15),
        ],
    ),
    # 指派问题（0-1）
    "assignment-01": IR(
        variables=[_var(f"x_{i}_{j}", VarType.binary) for i in range(3) for j in range(3)],
        objective=Objective(
            sense=Sense.minimize,
            expr=_expr([_t("x_0_0", 4), _t("x_0_1", 1), _t("x_0_2", 3),
                        _t("x_1_0", 2), _t("x_1_1", 0), _t("x_1_2", 5),
                        _t("x_2_0", 3), _t("x_2_1", 2), _t("x_2_2", 2)]),
        ),
        constraints=[
            *[Constraint(name=f"worker_{i}", sense=Sense.eq, lhs=_expr([_t(f"x_{i}_{j}", 1) for j in range(3)]), rhs=1) for i in range(3)],
            *[Constraint(name=f"task_{j}", sense=Sense.eq, lhs=_expr([_t(f"x_{i}_{j}", 1) for i in range(3)]), rhs=1) for j in range(3)],
        ],
    ),
    # 整数规划
    "integer-01": IR(
        variables=[_var("x", VarType.integer), _var("y", VarType.integer)],
        objective=Objective(sense=Sense.maximize, expr=_expr([_t("x", 5), _t("y", 8)])),
        constraints=[
            Constraint(name="c1", sense=Sense.le, lhs=_expr([_t("x", 1), _t("y", 1)]), rhs=6),
            Constraint(name="c2", sense=Sense.le, lhs=_expr([_t("x", 5), _t("y", 9)]), rhs=45),
        ],
    ),
    # 配料（连续 LP，>= 约束）
    "diet-01": IR(
        variables=[_var("x_A", VarType.continuous), _var("x_B", VarType.continuous)],
        objective=Objective(sense=Sense.minimize, expr=_expr([_t("x_A", 2), _t("x_B", 3)])),
        constraints=[
            Constraint(name="protein", sense=Sense.ge, lhs=_expr([_t("x_A", 3), _t("x_B", 1)]), rhs=15),
            Constraint(name="fat", sense=Sense.ge, lhs=_expr([_t("x_A", 1), _t("x_B", 2)]), rhs=9),
        ],
    ),
    # 不可行
    "infeasible-01": IR(
        variables=[_var("x", VarType.continuous)],
        objective=Objective(sense=Sense.maximize, expr=_expr([_t("x", 1)])),
        constraints=[
            Constraint(name="c1", sense=Sense.ge, lhs=_expr([_t("x", 1)]), rhs=5),
            Constraint(name="c2", sense=Sense.le, lhs=_expr([_t("x", 1)]), rhs=2),
        ],
    ),
    # 无界
    "unbounded-01": IR(
        variables=[_var("x", VarType.continuous), _var("y", VarType.continuous)],
        objective=Objective(sense=Sense.maximize, expr=_expr([_t("x", 1), _t("y", 1)])),
        constraints=[
            Constraint(name="c1", sense=Sense.le, lhs=_expr([_t("x", 1), _t("y", -1)]), rhs=10),
        ],
    ),
    # 多索引生产（2 台机器 × 3 种产品，连续 LP）
    "multi-index-01": IR(
        variables=[
            _var("x_A_1", VarType.continuous), _var("x_A_2", VarType.continuous), _var("x_A_3", VarType.continuous),
            _var("x_B_1", VarType.continuous), _var("x_B_2", VarType.continuous), _var("x_B_3", VarType.continuous),
        ],
        objective=Objective(sense=Sense.maximize, expr=_expr([
            _t("x_A_1", 5), _t("x_A_2", 7), _t("x_A_3", 4),
            _t("x_B_1", 6), _t("x_B_2", 5), _t("x_B_3", 8),
        ])),
        constraints=[
            # 机器 A 总工时 <= 120
            Constraint(name="machine_A", sense=Sense.le, lhs=_expr([_t("x_A_1", 2), _t("x_A_2", 3), _t("x_A_3", 1)]), rhs=120),
            # 机器 B 总工时 <= 90
            Constraint(name="machine_B", sense=Sense.le, lhs=_expr([_t("x_B_1", 1), _t("x_B_2", 2), _t("x_B_3", 3)]), rhs=90),
        ],
    ),
    # 选址（3 候选仓库 × 4 客户，0-1 固定成本 + 运输成本）
    "location-01": IR(
        variables=[
            _var("y_1", VarType.binary), _var("y_2", VarType.binary), _var("y_3", VarType.binary),
            _var("x_1_1", VarType.continuous), _var("x_1_2", VarType.continuous), _var("x_1_3", VarType.continuous), _var("x_1_4", VarType.continuous),
            _var("x_2_1", VarType.continuous), _var("x_2_2", VarType.continuous), _var("x_2_3", VarType.continuous), _var("x_2_4", VarType.continuous),
            _var("x_3_1", VarType.continuous), _var("x_3_2", VarType.continuous), _var("x_3_3", VarType.continuous), _var("x_3_4", VarType.continuous),
        ],
        objective=Objective(sense=Sense.minimize, expr=_expr([
            _t("y_1", 100), _t("y_2", 150), _t("y_3", 120),
            _t("x_1_1", 2), _t("x_1_2", 4), _t("x_1_3", 5), _t("x_1_4", 3),
            _t("x_2_1", 3), _t("x_2_2", 1), _t("x_2_3", 2), _t("x_2_4", 6),
            _t("x_3_1", 5), _t("x_3_2", 3), _t("x_3_3", 1), _t("x_3_4", 2),
        ])),
        constraints=[
            # 每个客户需求 = 1
            Constraint(name="d1", sense=Sense.eq, lhs=_expr([_t("x_1_1", 1), _t("x_2_1", 1), _t("x_3_1", 1)]), rhs=1),
            Constraint(name="d2", sense=Sense.eq, lhs=_expr([_t("x_1_2", 1), _t("x_2_2", 1), _t("x_3_2", 1)]), rhs=1),
            Constraint(name="d3", sense=Sense.eq, lhs=_expr([_t("x_1_3", 1), _t("x_2_3", 1), _t("x_3_3", 1)]), rhs=1),
            Constraint(name="d4", sense=Sense.eq, lhs=_expr([_t("x_1_4", 1), _t("x_2_4", 1), _t("x_3_4", 1)]), rhs=1),
            # 每个仓库若开放（y=1）才能供货，容量约束（y*M 型）
            Constraint(name="cap1", sense=Sense.le, lhs=_expr([_t("x_1_1", 1), _t("x_1_2", 1), _t("x_1_3", 1), _t("x_1_4", 1), _t("y_1", -4)]), rhs=0),
            Constraint(name="cap2", sense=Sense.le, lhs=_expr([_t("x_2_1", 1), _t("x_2_2", 1), _t("x_2_3", 1), _t("x_2_4", 1), _t("y_2", -4)]), rhs=0),
            Constraint(name="cap3", sense=Sense.le, lhs=_expr([_t("x_3_1", 1), _t("x_3_2", 1), _t("x_3_3", 1), _t("x_3_4", 1), _t("y_3", -4)]), rhs=0),
        ],
    ),
    # 最短路径（5 节点，源 s 到汇 t，0-1 边选择）
    "shortest-path-01": IR(
        variables=[
            _var("x_s1", VarType.binary), _var("x_s2", VarType.binary),
            _var("x_12", VarType.binary), _var("x_13", VarType.binary), _var("x_23", VarType.binary),
            _var("x_1t", VarType.binary), _var("x_2t", VarType.binary), _var("x_3t", VarType.binary),
        ],
        objective=Objective(sense=Sense.minimize, expr=_expr([
            _t("x_s1", 2), _t("x_s2", 4),
            _t("x_12", 1), _t("x_13", 5), _t("x_23", 1),
            _t("x_1t", 3), _t("x_2t", 2), _t("x_3t", 1),
        ])),
        constraints=[
            # 源 s：出流 - 入流 = 1
            Constraint(name="src", sense=Sense.eq, lhs=_expr([_t("x_s1", 1), _t("x_s2", 1)]), rhs=1),
            # 汇 t：入流 - 出流 = 1
            Constraint(name="sink", sense=Sense.eq, lhs=_expr([_t("x_1t", 1), _t("x_2t", 1), _t("x_3t", 1)]), rhs=1),
            # 中间节点 1：入 - 出 = 0（入 s1；出 12,13,1t）
            Constraint(name="n1", sense=Sense.eq, lhs=_expr([_t("x_s1", 1), _t("x_12", -1), _t("x_13", -1), _t("x_1t", -1)]), rhs=0),
            # 中间节点 2：入 - 出 = 0（入 s2,12；出 23,2t）
            Constraint(name="n2", sense=Sense.eq, lhs=_expr([_t("x_s2", 1), _t("x_12", 1), _t("x_23", -1), _t("x_2t", -1)]), rhs=0),
            # 中间节点 3：入 - 出 = 0（入 13,23；出 3t）
            Constraint(name="n3", sense=Sense.eq, lhs=_expr([_t("x_13", 1), _t("x_23", 1), _t("x_3t", -1)]), rhs=0),
        ],
    ),
    # 混合配料（3 原料，连续 LP）
    "blending-01": IR(
        variables=[_var("x1", VarType.continuous), _var("x2", VarType.continuous), _var("x3", VarType.continuous)],
        objective=Objective(sense=Sense.minimize, expr=_expr([_t("x1", 4), _t("x2", 5), _t("x3", 7)])),
        constraints=[
            Constraint(name="total", sense=Sense.eq, lhs=_expr([_t("x1", 1), _t("x2", 1), _t("x3", 1)]), rhs=100),
            Constraint(name="nutrient_min", sense=Sense.ge, lhs=_expr([_t("x1", 3), _t("x2", 2), _t("x3", 5)]), rhs=250),
            Constraint(name="x1_min", sense=Sense.ge, lhs=_expr([_t("x1", 1)]), rhs=20),
        ],
    ),
    # 排班（整数：每周每天最少人数）
    "scheduling-01": IR(
        variables=[_var("x1", VarType.integer), _var("x2", VarType.integer), _var("x3", VarType.integer),
                   _var("x4", VarType.integer), _var("x5", VarType.integer), _var("x6", VarType.integer), _var("x7", VarType.integer)],
        objective=Objective(sense=Sense.minimize, expr=_expr([_t("x1", 1), _t("x2", 1), _t("x3", 1), _t("x4", 1), _t("x5", 1), _t("x6", 1), _t("x7", 1)])),
        constraints=[
            Constraint(name="day1", sense=Sense.ge, lhs=_expr([_t("x1", 1), _t("x4", 1), _t("x5", 1), _t("x6", 1), _t("x7", 1)]), rhs=14),
            Constraint(name="day2", sense=Sense.ge, lhs=_expr([_t("x1", 1), _t("x2", 1), _t("x5", 1), _t("x6", 1), _t("x7", 1)]), rhs=11),
            Constraint(name="day3", sense=Sense.ge, lhs=_expr([_t("x1", 1), _t("x2", 1), _t("x3", 1), _t("x6", 1), _t("x7", 1)]), rhs=12),
            Constraint(name="day4", sense=Sense.ge, lhs=_expr([_t("x1", 1), _t("x2", 1), _t("x3", 1), _t("x4", 1), _t("x7", 1)]), rhs=15),
            Constraint(name="day5", sense=Sense.ge, lhs=_expr([_t("x1", 1), _t("x2", 1), _t("x3", 1), _t("x4", 1), _t("x5", 1)]), rhs=13),
            Constraint(name="day6", sense=Sense.ge, lhs=_expr([_t("x2", 1), _t("x3", 1), _t("x4", 1), _t("x5", 1), _t("x6", 1)]), rhs=10),
            Constraint(name="day7", sense=Sense.ge, lhs=_expr([_t("x3", 1), _t("x4", 1), _t("x5", 1), _t("x6", 1), _t("x7", 1)]), rhs=9),
        ],
    ),
    # 投资组合（连续 LP）
    "portfolio-01": IR(
        variables=[_var("x1", VarType.continuous), _var("x2", VarType.continuous), _var("x3", VarType.continuous)],
        objective=Objective(sense=Sense.maximize, expr=_expr([_t("x1", 0.08), _t("x2", 0.12), _t("x3", 0.10)])),
        constraints=[
            Constraint(name="budget", sense=Sense.eq, lhs=_expr([_t("x1", 1), _t("x2", 1), _t("x3", 1)]), rhs=100000),
            Constraint(name="risk", sense=Sense.le, lhs=_expr([_t("x1", 0.1), _t("x2", 0.3), _t("x3", 0.2)]), rhs=20000),
            Constraint(name="x3_max", sense=Sense.le, lhs=_expr([_t("x3", 1)]), rhs=30000),
        ],
    ),
    # 背包（多约束：容量 + 体积）
    "knapsack-03": IR(
        variables=[_var("x1", VarType.binary), _var("x2", VarType.binary), _var("x3", VarType.binary), _var("x4", VarType.binary)],
        objective=Objective(sense=Sense.maximize, expr=_expr([_t("x1", 30), _t("x2", 50), _t("x3", 60), _t("x4", 40)])),
        constraints=[
            Constraint(name="weight", sense=Sense.le, lhs=_expr([_t("x1", 4), _t("x2", 6), _t("x3", 8), _t("x4", 5)]), rhs=12),
            Constraint(name="volume", sense=Sense.le, lhs=_expr([_t("x1", 3), _t("x2", 5), _t("x3", 7), _t("x4", 4)]), rhs=10),
        ],
    ),
    # 生产（整数）
    "production-02": IR(
        variables=[_var("x1", VarType.integer), _var("x2", VarType.integer)],
        objective=Objective(sense=Sense.maximize, expr=_expr([_t("x1", 10), _t("x2", 15)])),
        constraints=[
            Constraint(name="labor", sense=Sense.le, lhs=_expr([_t("x1", 2), _t("x2", 3)]), rhs=100),
            Constraint(name="material", sense=Sense.le, lhs=_expr([_t("x1", 4), _t("x2", 2)]), rhs=120),
        ],
    ),
    # 指派 4×4
    "assignment-02": IR(
        variables=[_var(f"x_{i}_{j}", VarType.binary) for i in range(4) for j in range(4)],
        objective=Objective(sense=Sense.minimize, expr=_expr([
            _t("x_0_0", 9), _t("x_0_1", 2), _t("x_0_2", 7), _t("x_0_3", 8),
            _t("x_1_0", 6), _t("x_1_1", 4), _t("x_1_2", 3), _t("x_1_3", 7),
            _t("x_2_0", 5), _t("x_2_1", 8), _t("x_2_2", 1), _t("x_2_3", 8),
            _t("x_3_0", 7), _t("x_3_1", 6), _t("x_3_2", 9), _t("x_3_3", 4),
        ])),
        constraints=[
            *[Constraint(name=f"w{i}", sense=Sense.eq, lhs=_expr([_t(f"x_{i}_{j}", 1) for j in range(4)]), rhs=1) for i in range(4)],
            *[Constraint(name=f"t{j}", sense=Sense.eq, lhs=_expr([_t(f"x_{i}_{j}", 1) for i in range(4)]), rhs=1) for j in range(4)],
        ],
    ),
    # 带容量的选址（2 仓库 × 3 客户）
    "location-02": IR(
        variables=[
            _var("y1", VarType.binary), _var("y2", VarType.binary),
            _var("x_1_1", VarType.continuous), _var("x_1_2", VarType.continuous), _var("x_1_3", VarType.continuous),
            _var("x_2_1", VarType.continuous), _var("x_2_2", VarType.continuous), _var("x_2_3", VarType.continuous),
        ],
        objective=Objective(sense=Sense.minimize, expr=_expr([
            _t("y1", 80), _t("y2", 60),
            _t("x_1_1", 4), _t("x_1_2", 6), _t("x_1_3", 8),
            _t("x_2_1", 7), _t("x_2_2", 3), _t("x_2_3", 5),
        ])),
        constraints=[
            Constraint(name="d1", sense=Sense.eq, lhs=_expr([_t("x_1_1", 1), _t("x_2_1", 1)]), rhs=30),
            Constraint(name="d2", sense=Sense.eq, lhs=_expr([_t("x_1_2", 1), _t("x_2_2", 1)]), rhs=40),
            Constraint(name="d3", sense=Sense.eq, lhs=_expr([_t("x_1_3", 1), _t("x_2_3", 1)]), rhs=20),
            Constraint(name="cap1", sense=Sense.le, lhs=_expr([_t("x_1_1", 1), _t("x_1_2", 1), _t("x_1_3", 1), _t("y1", -100)]), rhs=0),
            Constraint(name="cap2", sense=Sense.le, lhs=_expr([_t("x_2_1", 1), _t("x_2_2", 1), _t("x_2_3", 1), _t("y2", -100)]), rhs=0),
        ],
    ),
    # 自由变量（可负）
    "free-var-01": IR(
        variables=[_var("x", VarType.continuous, lower=None), _var("y", VarType.continuous, lower=None)],
        objective=Objective(sense=Sense.minimize, expr=_expr([_t("x", 1), _t("y", 1)])),
        constraints=[
            Constraint(name="c1", sense=Sense.eq, lhs=_expr([_t("x", 1), _t("y", 1)]), rhs=5),
            Constraint(name="c2", sense=Sense.ge, lhs=_expr([_t("x", 2), _t("y", 1)]), rhs=8),
        ],
    ),
}


def get_gold_ir(sample_id: str) -> IR | None:
    """按样例 id 取黄金 IR（仅内置 `GOLD_IR` 20 题）。

    ⚠️ 不要给外部数据集（OR-Bench/NL4OPT/IndustryOR）接黄金 IR：
    `run_sample` 在「无内嵌 IR 且 get_gold_ir 命中」时会**直接用黄金 IR、跳过 LLM 理解**，
    这会让外部 NL 评测静默作弊。外部数据集的语义覆盖是**另一回事**，须走
    `force_llm_understand` 的显式路径，绝不能混进 get_gold_ir 的取值分支。
    """
    return GOLD_IR.get(sample_id)
