"""代码生成器：IR → 可独立执行的 PuLP 求解源码。

生成的是**真正可执行**的脚本（`generate_code(ir)` → 可被 python 直接运行）：
1. 用于 Docker 沙箱执行（EXECUTE 环节的真实执行路径）；
2. 落盘为 `final_code.py` 供人工审计/复现；
3. 作为 L1 代码层修复的对象。

约定：
- 依赖仅 PuLP（沙箱镜像内预装），不依赖本项目其它模块；
- 结果以 `__NL2OPT_RESULT__<json>` 单行输出，供沙箱解析；
- **变量名与 Python 标识符解耦**：Python 名用安全名 `_v{i}`，IR 名只作为
  `LpVariable` 的显示名参数与结果 JSON 的 key（结果回填用显式「原始名 → 安全表达式」
  映射，不依赖 PuLP 可能改写的 name）。带索引变量族 `y`（index_set=[0,1]）→ `_v0['0']`。

本地（backend=local）仍由 `solver.engine` 直接基于 IR 求解；本模块产出的代码
在 backend=docker 时才是实际执行对象。
"""
from __future__ import annotations

from .ir import IR, LinearExpr, Sense, VarType

RESULT_MARKER = "__NL2OPT_RESULT__"


def generate_code(ir: IR) -> str:
    """生成可独立执行的求解脚本源码。"""
    # 安全标识符：IR 变量名与 Python 名解耦（IR 名可能是 x-y / class / 1x 等非法标识符）
    py_base: dict[str, str] = {}
    for i, var in enumerate(ir.variables):
        py_base[var.name] = _safe_identifier(i)
    name_map = _build_name_map(ir, py_base)
    lines: list[str] = []

    # 0) 头部与导入
    lines.append('"""自动生成：本文件由 nl2opt 依据 IR 生成，可独立执行（依赖 PuLP）。"""')
    lines.append("import json")
    lines.append("")
    lines.append("from pulp import (")
    lines.append("    LpAffineExpression,")
    lines.append("    LpMaximize,")
    lines.append("    LpMinimize,")
    lines.append("    LpProblem,")
    lines.append("    LpStatus,")
    lines.append("    LpVariable,")
    lines.append("    PULP_CBC_CMD,")
    lines.append("    value,")
    lines.append(")")
    lines.append("")

    # 1) 变量定义
    lines.append("# 1) 变量定义")
    for var in ir.variables:
        cat = _cat(var.var_type)
        pyn = py_base[var.name]          # 安全 Python 名
        if var.index_set is None:
            # LpVariable 第一个参数是 PuLP 显示名（保留原始 IR 名），Python 名用安全名
            lines.append(f"{pyn} = LpVariable({_q(var.name)}, {_bounds_args(var)}, cat='{cat}')")
        else:
            lines.append(f"{pyn} = {{}}")
            for inst in var.instances():
                suffix = var.index_suffix(inst)
                lines.append(
                    f"{pyn}[{suffix!r}] = LpVariable({_q(inst)}, {_bounds_args(var)}, cat='{cat}')"
                )
    lines.append("")

    # 2) 问题与目标
    sense = "LpMaximize" if ir.objective.sense == Sense.maximize else "LpMinimize"
    lines.append("# 2) 问题与目标")
    lines.append(f"prob = LpProblem('problem', {sense})")
    obj_expr = _expr_str(ir.objective.expr, name_map)
    if not ir.objective.expr.terms:
        # 目标无变量项：用 LpAffineExpression 承载常数，避免 PuLP 把它当约束
        obj_expr = f"LpAffineExpression(constant={_fmt(ir.objective.expr.constant)})"
    lines.append(f"prob += {obj_expr}, 'objective'")
    lines.append("")

    # 3) 约束
    lines.append("# 3) 约束")
    for con in ir.constraints:
        op = {"<=": "<=", ">=": ">=", "=": "=="}[con.sense.value]
        lines.append(f"prob += {_expr_str(con.lhs, name_map)} {op} {_fmt(con.rhs)}, {_q(con.name)}")
    lines.append("")

    # 4) 求解
    lines.append("# 4) 求解")
    lines.append("prob.solve(PULP_CBC_CMD(msg=False))")
    lines.append("")

    # 5) 结果输出（供沙箱解析）
    # 显式「原始实例名 → 安全表达式」映射，**不依赖 PuLP 的 prob.variables() 与显示名**：
    # PuLP 可能对 name 加后缀去重/改写，`_v.name` 未必等于 IR 实例名；
    # 这里按 name_map 的确定顺序逐个 value(表达式)，key 用原始实例名，映射显式且稳定。
    int_names = sorted(
        inst
        for v in ir.variables
        if v.var_type in (VarType.integer, VarType.binary)
        for inst in v.instances()
    )
    ordered_names = sorted(name_map)            # 原始实例名，确定顺序
    lines.append("# 5) 结果输出（__NL2OPT_RESULT__ 供沙箱解析）")
    lines.append(f"_INT_NAMES = {set(int_names)!r}")
    lines.append(f"_VAR_NAMES = {ordered_names!r}")
    lines.append(f"_VAR_PY = [{', '.join(name_map[n] for n in ordered_names)}]")
    lines.append(
        "_STATUS_MAP = {'Optimal': 'optimal', 'Infeasible': 'infeasible', "
        "'Unbounded': 'unbounded', 'Not Solved': 'error', 'Undefined': 'error'}"
    )
    lines.append("_st = _STATUS_MAP.get(LpStatus[prob.status], 'error')")
    lines.append("# 仅在求得解时回填目标值与变量：不可行/无界时 PuLP 的 varValue 与")
    lines.append("# value(objective) 仍会返回残值（如 0 或松弛解），必须置空以免误报。")
    lines.append("_has_sol = _st in ('optimal', 'feasible')")
    lines.append("_vars = {}")
    lines.append("if _has_sol:")
    lines.append("    for _i, _n in enumerate(_VAR_NAMES):")
    lines.append("        _val = value(_VAR_PY[_i])")
    lines.append("        if _val is not None and _n in _INT_NAMES:")
    lines.append("            _val = float(round(_val))")
    lines.append("        _vars[_n] = None if _val is None else float(_val)")
    lines.append("_obj = value(prob.objective) if _has_sol else None")
    lines.append("print('__NL2OPT_RESULT__' + json.dumps({")
    lines.append("    'status': _st,")
    lines.append("    'objective_value': None if _obj is None else float(_obj),")
    lines.append("    'variables': _vars,")
    lines.append("}))")
    return "\n".join(lines) + "\n"


def _safe_identifier(i: int) -> str:
    """安全 Python 标识符：与 IR 变量名解耦，杜绝非法标识符导致 SyntaxError。"""
    return f"_v{i}"


def _build_name_map(ir: IR, py_base: dict[str, str]) -> dict[str, str]:
    """变量实例名 → Python 表达式（用安全标识符，而非 IR 原始名）。

    带索引变量族一律声明为 dict，键为「索引后缀字符串」，从而统一支持
    整数索引、字符串索引与多维索引：
      index_set=[0,1]            → _v0['0']、_v0['1']（实例名 y_0、y_1）
      index_set=['J1','J2']      → _v0['J1']（实例名 start_J1）
      index_set=[['J1','M1']]    → _v0['J1_M1']（实例名 assign_J1_M1）

    安全标识符由 `py_base`（基名 → `_v{i}`）提供：IR 变量名可能是 `x-y`、`class`、
    `1x` 等非法 Python 标识符，**绝不能直接当 Python 名用**（否则生成的代码
    SyntaxError，落入 L1，而 local 后端按字符串名照常工作 → 后端分叉）。
    """
    m: dict[str, str] = {}
    for var in ir.variables:
        pyn = py_base[var.name]
        if var.index_set is None:
            m[var.name] = pyn
        else:
            for inst in var.instances():
                m[inst] = f"{pyn}[{var.index_suffix(inst)!r}]"
    return m


def _cat(vt: VarType) -> str:
    return {"continuous": "Continuous", "integer": "Integer", "binary": "Binary"}[vt.value]


def _bounds_args(var) -> str:
    """生成 lowBound/upBound 关键字参数。

    与 `solver.engine._split_constraints` 的边界构造**逐字对齐**（双后端一致性）：
    - continuous / integer：`lower=None` → **无下界**（写 `lowBound=None`）；`upper=None` → 无上界；
    - binary：有效域 = `[max(lower, 0), min(upper, 1)]`（与 engine 的 clamp 相同）。

    ⚠️ 历史缺陷（v22 修复）：旧实现在 `lower is None` 且类型**非 continuous** 时
    **强制 `lowBound=0`**，而 engine 用 `-inf`——同一 IR（整数变量、`lower=None`）在两后端
    得到**不同模型**。此前因 `Variable.lower` 默认 0.0、该分支极少触发而未被发现；
    默认值改为 `None` 后它会变成常见情形，故必须对齐。
    """
    import math

    # 有效边界（与 engine 完全一致的口径）
    lb = var.lower if var.lower is not None else -math.inf
    ub = var.upper if var.upper is not None else math.inf
    if var.var_type == VarType.binary:
        lb = max(lb, 0.0)
        ub = min(ub, 1.0)

    parts: list[str] = []
    if var.var_type == VarType.binary:
        # PuLP 的 Binary 类别自带 [0,1]：常见情形（有效域恰为 [0,1]）保持输出 `lowBound=None`
        # 与旧版**逐字一致**（零churn）；仅在 IR 显式收窄时写出边界，与 engine 的 clamp 对齐。
        if lb > 0.0:
            parts.append(f"lowBound={_fmt(lb)}")
        if ub < 1.0:
            parts.append(f"upBound={_fmt(ub)}")
        return ", ".join(parts) if parts else "lowBound=None"

    # continuous / integer：显式写出，None 表示无下界（不依赖 PuLP 默认值，避免隐式非负）
    parts.append("lowBound=None" if lb == -math.inf else f"lowBound={_fmt(lb)}")
    if ub != math.inf:
        parts.append(f"upBound={_fmt(ub)}")
    return ", ".join(parts)


def _expr_str(expr: LinearExpr, name_map: dict[str, str]) -> str:
    """线性表达式 → Python 源码字符串（正确处理正负号）。"""
    pieces: list[str] = []
    for t in expr.terms:
        if t.coef == 0:
            continue
        py = name_map.get(t.var, t.var)
        pieces.append(_term_str(t.coef, py, first=not pieces))
    if expr.constant or not pieces:
        pieces.append(_term_str(expr.constant, "", first=not pieces, is_const=True))
        # constant=0 且无项时，pieces 会得到 "0"
    return " ".join(pieces) if pieces else "0"


def _term_str(coef: float, py: str, first: bool, is_const: bool = False) -> str:
    body = _fmt(abs(coef)) if is_const else (
        py if abs(coef) == 1 else f"{_fmt(abs(coef))}*{py}"
    )
    if first:
        return f"-{body}" if coef < 0 else body
    return f"- {body}" if coef < 0 else f"+ {body}"


def _q(name: str) -> str:
    """把名称安全地放进单引号字符串字面量。"""
    return "'" + str(name).replace("\\", "\\\\").replace("'", "\\'") + "'"


def _fmt(x: float) -> str:
    if x is None:
        return "None"
    if float(x).is_integer():
        return str(int(x))
    return repr(float(x))
