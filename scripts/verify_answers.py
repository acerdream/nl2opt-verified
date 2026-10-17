"""批量独立验证：对全量评测的 31 道待核验题，按题目语义独立建模求解。

每题输出：连续LP解、整数MILP解、参考答案、LLM解，供人工判定：
- 参考答案错（连续解 == LLM 解 != 参考答案）
- LLM 建模错（连续解 == 参考答案 != LLM 解）

用法：python scripts/verify_answers.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nl2opt.ir import IR, Constraint, LinearExpr, Objective, Sense, Term, VarType, Variable  # noqa: E402
from nl2opt.solver import solve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "nl4opt.jsonl"


def T(v, c):
    return Term(var=v, coef=c)


def E(ts, k=0.0):
    return LinearExpr(terms=ts, constant=k)


def V(n, vt, lb=0.0):
    return Variable(name=n, var_type=vt, lower=lb)


def C(n, s, l, r):
    return Constraint(name=n, sense=s, lhs=l, rhs=r)


def build(vars_, obj_sense, obj_terms, cons, var_types):
    """var_types: dict name -> 'cont'|'int'"""
    variables = []
    for n in vars_:
        vt = VarType.integer if var_types.get(n) == "int" else VarType.continuous
        variables.append(V(n, vt))
    # 把约束元组转成 Constraint 对象
    constraints = [C(name, sn, E(terms), rhs) for (name, sn, terms, rhs) in cons]
    ir = IR(
        variables=variables,
        objective=Objective(sense=obj_sense, expr=E(obj_terms)),
        constraints=constraints,
    )
    return solve(ir)


# 每道题：(id, 变量名列表, sense, 目标项, 约束列表, 变量类型dict)
# 约束列表每项 = (name, sense_str, lhs_terms, rhs)
PROBLEMS = []

P = PROBLEMS.append

# 021 面包房
P(("nl4opt-021", ["b", "c"], Sense.maximize, [T("b", 20), T("c", 40)],
   [("oven", Sense.le, [T("b", 2), T("c", 1)], 70), ("chef", Sense.le, [T("b", 0.25), T("c", 2)], 32)],
   {"b": "cont", "c": "cont"}))

# 022 催化剂
P(("nl4opt-022", ["x", "y"], Sense.maximize, [T("x", 5), T("y", 4)],
   [("pt", Sense.le, [T("x", 15), T("y", 20)], 450), ("pd", Sense.le, [T("x", 25), T("y", 14)], 390)],
   {"x": "cont", "y": "cont"}))

# 025 药丸
P(("nl4opt-025", ["z", "y"], Sense.minimize, [T("z", 1), T("y", 3)],
   [("z1", Sense.ge, [T("z", 1.3), T("y", 1.2)], 5), ("d3", Sense.ge, [T("z", 1.5), T("y", 5)], 10)],
   {"z": "cont", "y": "cont"}))

# 028 渔业
P(("nl4opt-028", ["d", "t"], Sense.maximize, [T("d", 100), T("t", 300)],
   [("budget", Sense.le, [T("d", 50), T("t", 100)], 1000), ("d_lt_t", Sense.le, [T("d", 1), T("t", -1)], -1)],
   {"d": "int", "t": "int"}))

# 032 洗发水
P(("nl4opt-032", ["s", "g"], Sense.minimize, [T("s", 0.5), T("g", 0.75)],
   [("s_min", Sense.ge, [T("s", 1)], 100), ("total", Sense.eq, [T("s", 1), T("g", 1)], 400), ("s_le_2g", Sense.le, [T("s", 1), T("g", -2)], 0)],
   {"s": "cont", "g": "cont"}))

# 034 运输设备
P(("nl4opt-034", ["c", "t"], Sense.minimize, [T("c", 2), T("t", 4)],
   [("rate", Sense.ge, [T("c", 5), T("t", 7)], 100), ("t_min", Sense.ge, [T("t", 1)], 12), ("t_ratio", Sense.le, [T("c", -0.4), T("t", 0.6)], 0)],
   {"c": "int", "t": "int"}))

# 036 回收箱
P(("nl4opt-036", ["s", "l"], Sense.maximize, [T("s", 25), T("l", 60)],
   [("workers", Sense.le, [T("s", 2), T("l", 5)], 100), ("s_eq_3l", Sense.eq, [T("s", 1), T("l", -3)], 0), ("s_min", Sense.ge, [T("s", 1)], 10), ("l_min", Sense.ge, [T("l", 1)], 4)],
   {"s": "int", "l": "int"}))

# 051 纸张
P(("nl4opt-051", ["g", "m"], Sense.maximize, [T("g", 4), T("m", 2.5)],
   [("print", Sense.le, [T("g", 3), T("m", 1.5)], 350), ("scan", Sense.le, [T("g", 5.5), T("m", 3)], 350)],
   {"g": "cont", "m": "cont"}))

# 068 药厂
P(("nl4opt-068", ["f1", "f2"], Sense.minimize, [T("f1", 1), T("f2", 1)],
   [("allergy", Sense.ge, [T("f1", 20), T("f2", 10)], 700), ("fever", Sense.ge, [T("f1", 15), T("f2", 30)], 600), ("compound", Sense.le, [T("f1", 20), T("f2", 30)], 1000)],
   {"f1": "cont", "f2": "cont"}))

# 071 糖尿病药
P(("nl4opt-071", ["A", "B"], Sense.maximize, [T("A", 12), T("B", 8)],
   [("imported", Sense.le, [T("A", 30), T("B", 40)], 300), ("mrna", Sense.le, [T("A", 50), T("B", 30)], 400), ("A_max", Sense.le, [T("A", 1)], 5), ("B_gt_A", Sense.le, [T("A", 1), T("B", -1)], -1)],
   {"A": "int", "B": "int"}))

# 073 投资
P(("nl4opt-073", ["t", "h"], Sense.maximize, [T("t", 0.03), T("h", 0.01)],
   [("total", Sense.le, [T("t", 1), T("h", 1)], 100000), ("t_ge_3h", Sense.ge, [T("t", 1), T("h", -3)], 0), ("t_max", Sense.le, [T("t", 1)], 70000)],
   {"t": "cont", "h": "cont"}))

# 085 原油
P(("nl4opt-085", ["l", "n", "h"], Sense.maximize, [T("l", 550), T("n", 750), T("h", 950)],
   [("A", Sense.le, [T("l", 3), T("n", 6), T("h", 9)], 250), ("B", Sense.le, [T("l", 3), T("n", 2), T("h", 3)], 150)],
   {"l": "cont", "n": "cont", "h": "cont"}))

# 105 肉类
P(("nl4opt-105", ["h", "r"], Sense.maximize, [T("h", 150), T("r", 300)],
   [("slicer", Sense.le, [T("h", 4), T("r", 2)], 4000), ("packer", Sense.le, [T("h", 2.5), T("r", 3.5)], 4000)],
   {"h": "cont", "r": "cont"}))

# 123 疫苗
P(("nl4opt-123", ["c", "a"], Sense.minimize, [T("c", 50), T("a", 75)],
   [("mrna", Sense.le, [T("c", 50), T("a", 75)], 20000), ("adult_70", Sense.le, [T("c", 0.7), T("a", -0.3)], 0), ("c_min", Sense.ge, [T("c", 1)], 50)],
   {"c": "int", "a": "int"}))

# 125 大猩猩
P(("nl4opt-125", ["b", "m"], Sense.minimize, [T("b", 10), T("m", 8)],
   [("cal", Sense.ge, [T("b", 80), T("m", 100)], 4000), ("k", Sense.ge, [T("b", 20), T("m", 15)], 150), ("m_33", Sense.le, [T("b", -0.33), T("m", 0.67)], 0)],
   {"b": "cont", "m": "cont"}))

# 131 糖浆
P(("nl4opt-131", ["x", "y"], Sense.minimize, [T("x", 0.5), T("y", 0.3)],
   [("throat_max", Sense.le, [T("x", 0.5), T("y", 0.2)], 5), ("lung_min", Sense.ge, [T("x", 0.4), T("y", 0.5)], 4)],
   {"x": "cont", "y": "cont"}))

# 138 减肥餐
P(("nl4opt-138", ["s", "b"], Sense.maximize, [T("s", 2), T("b", 7)],
   [("cal", Sense.le, [T("s", 300), T("b", 250)], 2000), ("b_eq_2s", Sense.eq, [T("s", -2), T("b", 1)], 0)],
   {"s": "int", "b": "int"}))

# 143 沙漠运输
P(("nl4opt-143", ["c", "t"], Sense.minimize, [T("c", 12), T("t", 5)],
   [("deliver", Sense.ge, [T("c", 50), T("t", 150)], 1500), ("c_gt_t", Sense.le, [T("c", -1), T("t", 1)], -1)],
   {"c": "int", "t": "int"}))

# 155 婴儿食品
P(("nl4opt-155", ["a", "c"], Sense.maximize, [T("a", 2), T("c", 4)],
   [("a_eq_3c", Sense.eq, [T("a", 1), T("c", -3)], 0), ("c_min", Sense.ge, [T("c", 1)], 2), ("folate", Sense.le, [T("a", 5), T("c", 3)], 100)],
   {"a": "int", "c": "int"}))

# 163 餐厅
P(("nl4opt-163", ["c", "h"], Sense.minimize, [T("c", 5), T("h", 20)],
   [("interaction", Sense.ge, [T("c", 70), T("h", 85)], 4000), ("cart_70", Sense.ge, [T("c", 0.3), T("h", -0.7)], 0), ("h_min", Sense.ge, [T("h", 1)], 3)],
   {"c": "int", "h": "int"}))

# 167 员工餐
P(("nl4opt-167", ["h", "w"], Sense.minimize, [T("h", 6.5), T("w", 4)],
   [("cal", Sense.ge, [T("h", 800), T("w", 450)], 2200), ("protein", Sense.ge, [T("h", 19), T("w", 12)], 50), ("carb", Sense.ge, [T("h", 20), T("w", 10)], 70)],
   {"h": "cont", "w": "cont"}))

# 174 体温计
P(("nl4opt-174", ["e", "r"], Sense.maximize, [T("e", 1), T("r", 1)],
   [("time", Sense.le, [T("e", 3), T("r", 2)], 15000), ("e_ge_2r", Sense.ge, [T("e", 1), T("r", -2)], 0), ("r_min", Sense.ge, [T("r", 1)], 50)],
   {"e": "int", "r": "int"}))

# 180 水果
P(("nl4opt-180", ["o", "g"], Sense.minimize, [T("o", 5), T("g", 6)],
   [("vc", Sense.ge, [T("o", 5), T("g", 7)], 80), ("va", Sense.ge, [T("o", 3), T("g", 5)], 70), ("o_ge_2g", Sense.ge, [T("o", 1), T("g", -2)], 0)],
   {"o": "cont", "g": "cont"}))

# 187 诊所
P(("nl4opt-187", ["s", "p"], Sense.maximize, [T("s", 10), T("p", 7)],
   [("vc", Sense.le, [T("s", 30), T("p", 50)], 1200), ("vd", Sense.le, [T("s", 40), T("p", 30)], 1500), ("s_max", Sense.le, [T("s", 1)], 10), ("p_gt_s", Sense.le, [T("s", 1), T("p", -1)], -1)],
   {"s": "int", "p": "int"}))

# 192 赌博
P(("nl4opt-192", ["b", "h", "s"], Sense.maximize, [T("b", 1.2), T("h", 0.5), T("s", 0.1)],
   [("loss", Sense.le, [T("b", 0.2), T("h", -0.05), T("s", -0.2)], 0), ("total", Sense.le, [T("b", 1), T("h", 1), T("s", 1)], 100000)],
   {"b": "cont", "h": "cont", "s": "cont"}))

# 197 疫苗
P(("nl4opt-197", ["f", "s"], Sense.minimize, [T("f", 20), T("s", 60)],
   [("ab", Sense.le, [T("f", 30), T("s", 65)], 35000), ("f_gt_s", Sense.le, [T("f", -1), T("s", 1)], -1), ("s_min", Sense.ge, [T("s", 1)], 40)],
   {"f": "int", "s": "int"}))

# 202 农田
P(("nl4opt-202", ["t", "p"], Sense.maximize, [T("t", 350), T("p", 600)],
   [("land", Sense.le, [T("t", 1), T("p", 1)], 140), ("t_ge_p", Sense.ge, [T("t", 1), T("p", -1)], 0), ("t_le_2p", Sense.le, [T("t", 1), T("p", -2)], 0), ("t_min", Sense.ge, [T("t", 1)], 20), ("p_min", Sense.ge, [T("p", 1)], 30)],
   {"t": "cont", "p": "cont"}))

# 206 补充剂
P(("nl4opt-206", ["A", "B"], Sense.minimize, [T("A", 14), T("B", 25)],
   [("ca", Sense.ge, [T("A", 30), T("B", 60)], 400), ("mg", Sense.ge, [T("A", 50), T("B", 10)], 50)],
   {"A": "cont", "B": "cont"}))

# 210 药丸
P(("nl4opt-210", ["A", "B"], Sense.minimize, [T("A", 4), T("B", 5)],
   [("sleep", Sense.ge, [T("A", 3), T("B", 6)], 40), ("anti", Sense.ge, [T("A", 5), T("B", 1)], 50)],
   {"A": "cont", "B": "cont"}))

# 214 飞机
P(("nl4opt-214", ["L", "S"], Sense.minimize, [T("L", 1), T("S", 1)],
   [("cars", Sense.ge, [T("L", 30), T("S", 10)], 300), ("L_lt_S", Sense.le, [T("L", 1), T("S", -1)], -1)],
   {"L": "int", "S": "int"}))

# 226 三文鱼
P(("nl4opt-226", ["s", "e"], Sense.minimize, [T("s", 80), T("e", 20)],
   [("cal", Sense.ge, [T("s", 300), T("e", 200)], 2000), ("protein", Sense.ge, [T("s", 15), T("e", 8)], 90), ("e_40", Sense.le, [T("s", -0.4), T("e", 0.6)], 0)],
   {"s": "cont", "e": "cont"}))


def main() -> int:
    # 读取参考答案和 LLM 解
    refs = {}
    for line in DATA.read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            refs[d["id"]] = d["expected"]

    llm_vals = {}
    full = ROOT / "artifacts" / "eval_nl4opt_full.json"
    if full.exists():
        fd = json.loads(full.read_text(encoding="utf-8"))
        for it in fd.get("detail", []):
            llm_vals[it["id"]] = it.get("objective_value")

    print(f"{'id':12s} {'参考答案':>14s} {'LLM解':>14s} {'连续LP':>14s} {'整数MILP':>14s}")
    print("-" * 72)

    for (sid, vars_, sense, obj_terms, cons, vtypes) in PROBLEMS:
        ref = refs.get(sid, {}).get("objective_value")
        llm = llm_vals.get(sid)

        # 连续版
        cont_types = {n: "cont" for n in vars_}
        sol_cont = build(vars_, sense, obj_terms, cons, cont_types)
        # 整数版
        sol_int = build(vars_, sense, obj_terms, cons, vtypes)

        def fmt(v):
            if v is None:
                return "-"
            return f"{v:.4f}"

        print(f"{sid:12s} {fmt(ref):>14s} {fmt(llm):>14s} {fmt(sol_cont.objective_value):>14s} {fmt(sol_int.objective_value):>14s}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
