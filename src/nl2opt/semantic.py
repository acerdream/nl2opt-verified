"""语义覆盖对比：变量名无关、**约束顺序无关**的「漏变量 / 漏约束 / 定义域差异」检测。

核心思想：线性模型的语义由「系数模式」决定，而非变量名或书写顺序。
因此全部判据都用**多重集（multiset）**表示，不受变量重命名与约束顺序影响。

三类判据：
1. 变量「结构角色」指纹：目标系数 + 参与的各约束的 (自身系数, 约束签名[排除自身])
   排序多重集 + 归一化定义域。约束顺序变化不改变该多重集。
2. 约束签名多重集：系数 multiset + 常数 + sense + rhs（变量名无关）。
3. 目标签名：sense + 系数 multiset。

用途：
1. verifier.check_semantic_coverage() 在闭环验证阶段自动检测漏变量/漏约束/定义域差异。
2. scripts/compare_ir.py 命令行离线对拍。
3. scripts/eval_understanding.py 理解正确率评测。
4. scripts/report_orbench_nl.py 自然语言链路结构保真度判读。

历史修正（重要）：
旧实现的变量指纹是「按约束顺序」把系数排成向量，导致**同一模型只要约束顺序不同**
就会被判为「漏变量」（实测 OR-Bench INV-001 因代入消元化简 + 顺序差异，报出 24 组假阳性）。
现改为顺序无关的结构角色多重集。

仍存在的固有局限（非缺陷）：
本模块做的是**结构**比对。若两个模型数学等价但写法不同（如代入消元合并约束、
拆分变量），仍会报告差异——因为结构确实不同。此类情形需结合求解值人工判读。
对称变量（角色完全相同）只能按数量比对，无法定位具体是哪一个。
"""
from __future__ import annotations

from collections import Counter

from .ir import IR

TOL = 1e-6


def _round(x: float) -> float:
    """把浮点系数四舍五入到 6 位，消除浮点误差。"""
    return round(float(x), 6)


def _normalized_domain(ir: IR, var_name: str) -> tuple:
    """归一化变量定义域： (kind, lb, ub)。

    - `binary` 与 `integer + [0,1]` 语义等价，统一归一为 integer/[0,1]，
      避免因建模口径不同产生假差异；
    - `binary` 且未显式给上下界时补全为 [0,1]（PuLP/IR 的语义即如此）。
    """
    var = ir.var_map[var_name]
    kind = var.var_type.value
    lb, ub = var.lower, var.upper
    if kind == "binary":
        kind = "integer"
        lb = 0.0 if lb is None else max(float(lb), 0.0)
        ub = 1.0 if ub is None else min(float(ub), 1.0)
    return (
        kind,
        None if lb is None else _round(lb),
        None if ub is None else _round(ub),
    )


def _constraint_signature(con) -> tuple:
    """约束签名：系数 multiset + 常数 + sense + rhs（变量名无关）。"""
    coeffs = sorted(_round(t.coef) for t in con.lhs.terms)
    # 常数项也算进模式里，避免"漏了常数项"漏检
    coeffs.append(_round(con.lhs.constant))
    return (tuple(coeffs), con.sense.value, _round(con.rhs))


def _constraint_signature_excluding(con, var_name: str) -> tuple:
    """约束签名（排除指定变量自身的系数），用于构造该变量的结构角色。"""
    coeffs = sorted(_round(t.coef) for t in con.lhs.terms if t.var != var_name)
    return (tuple(coeffs), _round(con.lhs.constant), con.sense.value, _round(con.rhs))


def _var_role_fingerprint(ir: IR, var_name: str) -> tuple:
    """变量的「结构角色」指纹： (定义域, 目标系数, 参与约束的角色多重集)。

    - 约束顺序无关：角色用 `sorted(...)` 排序后成元组；
    - 变量名无关：角色里只有系数/常数/sense/rhs，不含任何变量名。
    """
    obj_coef = _round(sum(t.coef for t in ir.objective.expr.terms if t.var == var_name))
    roles = sorted(
        (_round(t.coef), _constraint_signature_excluding(con, var_name))
        for con in ir.constraints
        for t in con.lhs.terms
        if t.var == var_name
    )
    return (_normalized_domain(ir, var_name), obj_coef, tuple(roles))


def _var_fingerprint_multiset(ir: IR) -> Counter:
    """所有变量结构角色指纹的多重集（约束顺序、变量名均不影响）。"""
    return Counter(_var_role_fingerprint(ir, v) for v in ir.all_var_instances)


def _constraint_multiset(ir: IR) -> Counter:
    """所有约束签名的多重集。"""
    return Counter(_constraint_signature(con) for con in ir.constraints)


def _objective_signature(ir: IR) -> tuple:
    """目标函数签名：sense + 系数 multiset。"""
    coeffs = sorted(_round(t.coef) for t in ir.objective.expr.terms)
    return (ir.objective.sense.value, tuple(coeffs))


def _domain_multiset(ir: IR) -> Counter:
    """所有变量归一化定义域的多重集（用于检出整数/连续、上下界差异）。"""
    return Counter(_normalized_domain(ir, v) for v in ir.all_var_instances)


def semantic_diff(gold: IR, llm: IR) -> dict:
    """对比 LLM IR 相对黄金 IR 的语义覆盖（变量名无关、约束顺序无关）。

    返回：
    - missing_variables: 黄金 IR 有、LLM IR 缺的「变量结构角色」（含数量）
    - extra_variables:   LLM IR 多出来的变量结构角色
    - missing_constraints / extra_constraints: 约束签名差异
    - objective_mismatch: 目标函数（sense 或系数多重集）是否不一致
    - domain_mismatch:    变量定义域（类型/上下界）多重集是否不一致
    """
    gold_vars = _var_fingerprint_multiset(gold)
    llm_vars = _var_fingerprint_multiset(llm)

    missing_vars = gold_vars - llm_vars  # Counter 相减，保留正数
    extra_vars = llm_vars - gold_vars

    gold_cons = _constraint_multiset(gold)
    llm_cons = _constraint_multiset(llm)

    missing_cons = gold_cons - llm_cons
    extra_cons = llm_cons - gold_cons

    return {
        "missing_variables": [{"role": _jsonable(k), "count": v} for k, v in missing_vars.items()],
        "extra_variables": [{"role": _jsonable(k), "count": v} for k, v in extra_vars.items()],
        "missing_constraints": [
            {"signature": {"coeffs": list(k[0]), "sense": k[1], "rhs": k[2]}, "count": v}
            for k, v in missing_cons.items()
        ],
        "extra_constraints": [
            {"signature": {"coeffs": list(k[0]), "sense": k[1], "rhs": k[2]}, "count": v}
            for k, v in extra_cons.items()
        ],
        "objective_mismatch": _objective_signature(gold) != _objective_signature(llm),
        "domain_mismatch": _domain_multiset(gold) != _domain_multiset(llm),
        # 目标签名含 tuple，须转 JSON 友好结构，否则 diff 无法随 CheckResult.detail 落盘
        "gold_objective": _jsonable(_objective_signature(gold)),
        "llm_objective": _jsonable(_objective_signature(llm)),
        "gold_domains": [list(d) for d in _domain_multiset(gold)],
        "llm_domains": [list(d) for d in _domain_multiset(llm)],
    }


def _jsonable(value):
    """把嵌套元组结构转成 JSON 友好的列表结构。"""
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    return value


def has_semantic_gap(diff: dict) -> bool:
    """是否存在语义缺口（漏变量/漏约束/目标不一致/定义域不一致）。"""
    return (
        bool(diff["missing_variables"])
        or bool(diff["missing_constraints"])
        or bool(diff["objective_mismatch"])
        or bool(diff.get("domain_mismatch"))
    )


def describe_gap(diff: dict) -> str:
    """把差异渲染成人类可读摘要（供验证器消息使用）。

    不仅给**数量**，还给出**具体对象**（约束签名 / 变量结构角色），便于定位：
    - 变量名无关，故变量只能给「结构角色」（目标系数 + 定义域 + 参与约束数），
      无法给出具体名字（对称变量按数量比对，见模块 docstring）；
    - 约束给「系数多重集 + sense + rhs」的签名（含常数项，与 `_constraint_signature` 一致）。

    每类最多列出 3 个示例，避免消息过长。
    """
    parts: list[str] = []

    if diff["missing_variables"]:
        n = sum(item["count"] for item in diff["missing_variables"])
        descs = []
        for item in diff["missing_variables"][:3]:
            role = item["role"]  # [domain, obj_coef, roles]
            dom, oc, rs = role[0], role[1], role[2]
            kind, lb, ub = dom
            descs.append(f"目标系数={oc} 域={kind}[{lb},{ub}] 参与{len(rs)}条约束")
        suffix = f"，例如 {descs[0]}" if descs else ""
        more = f"（共 {len(diff['missing_variables'])} 种角色）"
        parts.append(f"漏变量 {n} 个{suffix}{more}")

    if diff["missing_constraints"]:
        n = sum(item["count"] for item in diff["missing_constraints"])
        sigs = []
        for item in diff["missing_constraints"][:3]:
            s = item["signature"]
            sigs.append("{" + ", ".join(map(str, s["coeffs"])) + f"}} {s['sense']} {s['rhs']}")
        suffix = f"，例如 {'；'.join(sigs)}" if sigs else ""
        parts.append(f"漏约束 {n} 条{suffix}")

    if diff["objective_mismatch"]:
        go, lo = diff["gold_objective"], diff["llm_objective"]
        parts.append(f"目标函数不一致（参考 {go}，实际 {lo}）")

    if diff.get("domain_mismatch"):
        parts.append("变量定义域不一致（整数/连续或上下界）")

    if diff["extra_variables"]:
        n = sum(item["count"] for item in diff["extra_variables"])
        descs = []
        for item in diff["extra_variables"][:3]:
            role = item["role"]
            dom, oc, rs = role[0], role[1], role[2]
            kind, lb, ub = dom
            descs.append(f"目标系数={oc} 域={kind}[{lb},{ub}] 参与{len(rs)}条约束")
        suffix = f"，例如 {descs[0]}" if descs else ""
        parts.append(f"多出变量 {n} 个{suffix}")

    if diff["extra_constraints"]:
        n = sum(item["count"] for item in diff["extra_constraints"])
        sigs = []
        for item in diff["extra_constraints"][:3]:
            s = item["signature"]
            sigs.append("{" + ", ".join(map(str, s["coeffs"])) + f"}} {s['sense']} {s['rhs']}")
        suffix = f"，例如 {'；'.join(sigs)}" if sigs else ""
        parts.append(f"多出约束 {n} 条{suffix}")

    return "；".join(parts) if parts else "无差异"
