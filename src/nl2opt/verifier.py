"""验证器：对求解结果做可行性、整数性、目标值正确性校验。

- feasibility：把解代回每条约束，逐条核验（含数值容差）。
- objective_consistency：由 IR 重算 Σ coef·x + constant，与报告的 objective_value 比对
  （闭合「执行 → 验证」边界，拒绝伪造 objective / 非有限值）。
- optimality：与参考答案的目标值对比。
- objective_defined：目标非退化（不能无项、不能系数全 0）。
- coefficient_grounding：目标系数必须有**题面依据**（如题面给了逐项权重，就不能全写成 1）。
- semantic_coverage：LLM 产出 IR 相对黄金 IR 的语义覆盖（漏变量/漏约束，变量名无关）。
- 校验失败时输出“结构化差异报告”，供反思环节定位修复。
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from .ir import IR, Sense, VarType
from .solver import SolverResult


@dataclass
class CheckResult:
    """单条校验结论。

    `blocking` 区分校验强度：
    - `blocking=True`（默认）：`ok=False` 会阻断 `all_pass`（硬失败，进入反思/重写）；
    - `blocking=False`：`ok=False` 仅作告警/诊断，**不阻断** `all_pass`
      （用于「弱证据」类检查，如目标系数依据的启发式扫描——证据不足以推翻
      一个本应接受的模型，只记录供人工复核）。

    这样避免「所有检查默认同等强度」：强证据（可解析的结构化权重）硬失败，
    弱证据（关键词邻近）只告警。
    """

    ok: bool
    message: str = ""
    detail: dict = field(default_factory=dict)
    kind: str = "generic"  # feasibility | optimality | status | no_solution | generic
    blocking: bool = True


def check_feasibility(ir: IR, sol: SolverResult, tol: float = 1e-6) -> CheckResult:
    """校验解是否满足全部约束、边界与整数/0-1 要求。

    解的**完整性**先于数值校验：声明变量必须全部出现、不得出现未声明变量、
    每个值必须是有限数（拒绝 NaN/Inf）。用 `x.get(t.var, 0.0)` 会把缺失变量
    静默当 0 处理，从而漏掉「下界应为 1 却缺值」这类错误；NaN 在数值比较里
    通常不触发失败，整数检查 `round(nan)` 还会抛异常——这些都必须在进入
    约束/边界/整数检验之前被明确拦截。
    """
    if not sol.success:
        return CheckResult(False, f"求解未成功，状态={sol.status}，无法做可行性校验", kind="feasibility")

    x = sol.variables
    vmap = ir.var_map
    violations: list[str] = []

    # 1) 解向量完整性：声明变量 ↔ 给出变量的对称性 + 有限值
    required = set(ir.all_var_instances)
    given = set(x)
    missing = required - given
    extra = given - required
    if missing:
        violations.append(f"解缺少已声明变量: {', '.join(sorted(missing))}")
    if extra:
        violations.append(f"解出现未声明变量: {', '.join(sorted(extra))}")
    for name, val in x.items():
        if not math.isfinite(val):
            violations.append(f"变量 {name}: 值非有限（{val!r}）")

    # 2) 约束（仅当解向量完整时才可安全求值）
    if not missing:
        for con in ir.constraints:
            lhs = con.lhs.constant
            for t in con.lhs.terms:
                lhs += t.coef * x[t.var]
            rhs = con.rhs
            if con.sense == Sense.le and lhs > rhs + tol:
                violations.append(f"约束 {con.name}: {lhs:.6f} <= {rhs} 违反")
            elif con.sense == Sense.ge and lhs < rhs - tol:
                violations.append(f"约束 {con.name}: {lhs:.6f} >= {rhs} 违反")
            elif con.sense == Sense.eq and abs(lhs - rhs) > tol:
                violations.append(f"约束 {con.name}: {lhs:.6f} == {rhs} 违反（差 {abs(lhs-rhs):.2e}）")

    # 3) 边界 + 整数性/0-1（对每个**已声明且已给出**的变量；未声明变量已在第 1 步拦截）
    for name in sorted(required & given):
        val = x[name]
        var = vmap[name]
        if var.lower is not None and val < var.lower - tol:
            violations.append(f"变量 {name}: {val} 小于下界 {var.lower}")
        if var.upper is not None and val > var.upper + tol:
            violations.append(f"变量 {name}: {val} 大于上界 {var.upper}")
        if var.var_type == VarType.integer and abs(val - round(val)) > tol:
            violations.append(f"变量 {name}: 应为整数，实际 {val}")
        if var.var_type == VarType.binary and abs(val - 0) > tol and abs(val - 1) > tol:
            violations.append(f"变量 {name}: 应为 0/1，实际 {val}")

    if violations:
        return CheckResult(False, "; ".join(violations), {"violations": violations}, kind="feasibility")
    return CheckResult(True, "可行", kind="feasibility")


def check_objective_consistency(ir: IR, sol: SolverResult, tol: float = 1e-6) -> CheckResult:
    """由 IR 重算目标值，与求解器**报告**的 objective_value 用独立容差比对。

    为什么必须（P0，闭合「执行 → 验证」可信边界）：
    - `check_optimality` 只拿 `sol.objective_value` 与参考答案比，`check_feasibility` 只逐条核
      约束与变量——**谁都不重算目标值**。于是 Docker 后端执行的不可信生成代码（或 L1 修复产物）
      可以输出「可行但非最优的 variables + 一个恰好等于期望的伪造 objective」，仍 all_pass。
    - 本检查从 IR 的 `Σ coef·x + constant` 独立重算，先校验有限性，再与报告值比对，
      任何「报告值与解向量不自洽」的解都会被拦下——不管参考答案是否可信。

    容差 `tol` 默认取可行性容差（1e-6，更严）：两端都源于同一解向量，应近乎相等；
    而 `check_optimality` 对外部参考答案用相对容差（1e-4，更松）。
    """
    if sol.objective_value is None:
        return CheckResult(
            False, "求解结果缺少目标值，无法做一致性校验",
            kind="objective_consistency", blocking=True,
        )
    if not math.isfinite(sol.objective_value):
        return CheckResult(
            False, f"求解报告的目标值非有限（{sol.objective_value!r}）",
            {"reported": sol.objective_value}, kind="objective_consistency", blocking=True,
        )

    expr = ir.objective.expr
    recomputed = float(expr.constant)
    missing: list[str] = []
    nonfinite: list[str] = []
    for t in expr.terms:
        val = sol.variables.get(t.var)
        if val is None:
            missing.append(t.var)
            continue
        if not math.isfinite(val):
            nonfinite.append(t.var)
            continue
        recomputed += float(t.coef) * val

    if missing:
        return CheckResult(
            False, f"无法重算目标值：解向量缺少变量 {', '.join(sorted(missing))}",
            {"missing": missing}, kind="objective_consistency", blocking=True,
        )
    if nonfinite:
        return CheckResult(
            False, f"目标值重算遇非有限变量值: {', '.join(sorted(nonfinite))}",
            {"nonfinite_vars": nonfinite}, kind="objective_consistency", blocking=True,
        )
    if not math.isfinite(recomputed):
        return CheckResult(
            False, f"IR 重算的目标值非有限（{recomputed!r}）",
            {"recomputed": recomputed}, kind="objective_consistency", blocking=True,
        )

    # 相对容差，与 check_optimality 同构：小目标值用绝对容差，大目标值自动放宽
    effective = tol * max(1.0, abs(recomputed))
    if abs(sol.objective_value - recomputed) > effective:
        return CheckResult(
            False,
            f"目标值与 IR 重算不符（报告 {sol.objective_value:.6f}，重算 {recomputed:.6f}）",
            {"reported": sol.objective_value, "recomputed": recomputed,
             "tolerance": effective},
            kind="objective_consistency", blocking=True,
        )
    return CheckResult(
        True, f"目标值与 IR 重算一致（{recomputed:.6f}）",
        {"recomputed": recomputed}, kind="objective_consistency",
    )


def _diagnostic_context(ir: IR, sol: SolverResult, tol: float = 1e-6, max_items: int = 6) -> str:
    """「可行但目标值不符」时的诊断上下文：紧约束 + 卡边界的变量。

    从 IR + 解向量**重算**（不依赖 solver 内部，两后端一致），回答「当前解被什么卡住」：
    - 紧约束（slack ≈ 0）：这条约束在限制最优，可能是 rhs/方向写错的地方；
    - 卡边界的**非 binary** 变量：被下界/上界压住，可能本该更大/更小。
      （binary 卡 0/1 是平凡事实、信息量低，故排除。）

    该文本会经 `describe()` 进入 reflect/rewrite 的 prompt；不含参考答案值，无答案泄露。
    """
    x = sol.variables
    vmap = ir.var_map

    tight: list[str] = []
    for con in ir.constraints:
        lhs = float(con.lhs.constant)
        for t in con.lhs.terms:
            v = x.get(t.var)
            if v is None:
                continue
            lhs += float(t.coef) * v
        if con.sense == Sense.le:
            gap = con.rhs - lhs
        elif con.sense == Sense.ge:
            gap = lhs - con.rhs
        else:  # eq
            gap = abs(lhs - con.rhs)
        if gap <= tol:
            tight.append(con.name)

    bound: list[str] = []
    for name in ir.all_var_instances:
        val = x.get(name)
        if val is None:
            continue
        var = vmap[name]
        if var.var_type == VarType.binary:
            continue  # 0/1 卡边界是平凡事实，不提供诊断信息
        if var.lower is not None and abs(val - var.lower) <= tol:
            bound.append(f"{name}=下界{var.lower}")
        if var.upper is not None and abs(val - var.upper) <= tol:
            bound.append(f"{name}=上界{var.upper}")

    parts: list[str] = []
    if tight:
        shown = ", ".join(tight[:max_items])
        if len(tight) > max_items:
            shown += f"（共{len(tight)}条）"
        parts.append(f"紧约束: {shown}")
    if bound:
        shown = ", ".join(bound[:max_items])
        if len(bound) > max_items:
            shown += f"（共{len(bound)}个）"
        parts.append(f"卡边界: {shown}")
    return "；".join(parts)


def check_optimality(sol: SolverResult, expected_obj: float | None, tol: float = 1e-4,
                     ir: IR | None = None,
                     answer_var: str | None = None) -> CheckResult:
    """对比数值与参考答案。expected_obj 为 None 时跳过数值对比。

    容差为**相对**形式：`tol × max(1, |期望值|)`。
    这样小目标值（如 2.0）用绝对容差，大目标值（如 1e6）自动放宽，
    避免求解器在浮点精度内的正常抖动被误判为「目标值不符」。
    （实测案例 SCH-029：求解 1.9999989999999999 vs 期望 2.0，差 ~1e-6，
      旧实现用可行性容差 1e-6 比较目标值而误判失败。）

    **有限性前置**（P0）：`abs(nan - expected) > tol` 为 False，旧实现会把
    `objective_value=NaN` 误判为「目标值一致」；`expected=NaN/Inf` 同理。生成代码的
    `json.dumps` 默认可写出 NaN（`allow_nan=True`），故这是真实可达路径，必须显式拒绝。

    **答案变量 `answer_var`**（目标规划 / 字典序优化）：这类题的参考答案**不是目标函数值**，
    而是某个偏差变量（例「至少多少不能满足 P3」→ `d3_minus`）。expected 带
    `answer_variable` 时，本函数改为比较**该变量在解中的值**。见 IndustryOR-029。

    **诊断上下文**：失败时若传入 `ir`，detail 会附「紧约束 + 卡边界变量」摘要（不进 message）。
    """
    if expected_obj is None:
        return CheckResult(True, "无参考答案，跳过目标值对比", kind="optimality")
    if not math.isfinite(float(expected_obj)):
        return CheckResult(
            False, f"参考答案目标值非有限（{expected_obj!r}），无法对比",
            {"expected": expected_obj}, kind="optimality",
        )
    if sol.objective_value is None:
        return CheckResult(False, "求解成功但目标值为空", kind="optimality")
    # 定位「被比较的实际值」：默认目标函数值；指定答案变量时改取该变量的值。
    # 目标规划/字典序题的目标函数是加权和（如 `10000*d2_minus + d3_minus`），
    # 其值没有业务含义；真正的答案在偏差变量里（如 `d3_minus`）。
    # 实测 IndustryOR-029：模型解正确（d3_minus=2500=参考），但 objective=10,003,000。
    if answer_var:
        raw = sol.variables.get(answer_var)
        if raw is None:
            return CheckResult(
                False,
                f"答案变量 {answer_var!r} 不在求解结果中：目标规划/字典序题的偏差变量"
                f"须命名为 d{{k}}_minus / d{{k}}_plus 并写进模型",
                {"answer_variable": answer_var,
                 "available_vars": sorted(sol.variables)[:15]},
                kind="optimality",
            )
        actual = float(raw)
    else:
        actual = float(sol.objective_value)
    if not math.isfinite(actual):
        return CheckResult(
            False, f"被比较的数值非有限（{actual!r}）",
            {"actual": actual, "answer_variable": answer_var}, kind="optimality",
        )
    effective = tol * max(1.0, abs(float(expected_obj)))
    if abs(actual - expected_obj) > effective:
        # ⚠️ message 里**不得**出现参考值（expected_obj）。
        # 这条文本会经 `describe()` 进入反思/重写的 prompt——LLM 看得到期望值，
        # 就能"反推出能算出该值的模型"来拟合答案，而不是重读题面修正建模。
        # 这是对**整个重写闭环证据链**的污染，不分 trusted/untrusted
        # （历史案例：converged 判定曾据此被反复拉扯）。
        # 完整数值保留在 detail 字段（供诊断/报告/人工复核），不进入 prompt。
        label = f"答案 `{answer_var}`" if answer_var else "目标值"
        msg = f"{label}与参考不符（求解 {actual:.6f}）"
        detail: dict = {"actual": actual, "expected": expected_obj,
                        "tolerance": effective}
        if answer_var:
            detail["answer_variable"] = answer_var
        # 诊断上下文（紧约束 + 卡边界）放进 **detail**（落盘/人复核可见），
        # **不进 message**——实测 A/B（OR-Bench 8 道「目标值不符」题）显示：
        # 带诊断 1/8 vs 不带 1/8，无提升；且根因多在**目标侧**（系数错），
        # 紧约束是**约束侧**信息，错位 + 过载（16 紧约束/15 卡边界多为「本应如此」）。
        if ir is not None:
            ctx = _diagnostic_context(ir, sol)
            if ctx:
                detail["diagnostic"] = ctx
        return CheckResult(False, msg, detail, kind="optimality")
    if answer_var:
        return CheckResult(True, f"答案 `{answer_var}` 一致", kind="optimality")
    return CheckResult(True, "目标值一致", kind="optimality")


def check_status(expected_status: str | None, sol: SolverResult) -> CheckResult:
    """核对求解状态与参考答案声明的状态（如 infeasible/unbounded）。"""
    if expected_status is None:
        return CheckResult(True, "参考答案未声明状态", kind="status")
    if expected_status != sol.status:
        return CheckResult(
            False,
            f"状态不符：求解 {sol.status}，期望 {expected_status}",
            {"actual": sol.status, "expected": expected_status},
            kind="status",
        )
    return CheckResult(True, f"状态一致（{sol.status}）", kind="status")


def verify(
    ir: IR,
    sol: SolverResult,
    expected: dict,
    feas_tol: float = 1e-6,
    obj_tol: float = 1e-4,
    nl_text: str | None = None,
) -> list[CheckResult]:
    """综合验证：状态 + 可行性 + 目标值（+ 题面依据）。返回所有校验结果列表。

    **可行性容差与目标值容差是两个独立的量**：
    - `feas_tol`：约束/边界/整数性判定（默认 1e-6，越小越严）；
    - `obj_tol`：目标值比较（默认 1e-4，取相对形式，见 `check_optimality`）。

    `nl_text` 非空时额外做「目标系数是否有题面依据」检查（无参考也能做，
    见 `check_coefficient_grounding`）。

    历史缺陷：旧实现只有一个 `tol`，被同时用于两者，导致目标值比较实际用了
    可行性容差 1e-6，把求解器浮点抖动误判为建模错误（实测 OR-Bench SCH-029 误判失败）。
    """
    results: list[CheckResult] = []

    expected_status = expected.get("status")
    expected_obj = expected.get("objective_value")

    # 若参考答案明确声明状态（infeasible/unbounded/no_solution），优先核对状态
    if expected_status in ("infeasible", "unbounded"):
        results.append(check_status(expected_status, sol))
        return results

    if expected_status == "no_solution":
        # “无最优解”：求解返回 infeasible 或 unbounded 都视为符合
        ok = sol.status in ("infeasible", "unbounded")
        results.append(
            CheckResult(
                ok,
                f"无最优解核对：求解状态={sol.status}" + ("" if ok else "，期望无解"),
                {"actual": sol.status, "expected": "no_solution"},
                kind="no_solution",
            )
        )
        return results

    results.append(check_objective_defined(ir, expected))
    if nl_text:
        results.append(check_coefficient_grounding(ir, nl_text, expected))
    results.append(check_feasibility(ir, sol, feas_tol))
    # 目标值一致性（P0）：由 IR 重算，先于参考答案对比——闭合「执行 → 验证」边界，
    # 拒绝「可行解 + 伪造 objective」的自洽性缺口（不依赖参考答案是否可信）。
    results.append(check_objective_consistency(ir, sol, feas_tol))
    results.append(check_optimality(sol, expected_obj, obj_tol, ir,
                                    answer_var=expected.get("answer_variable")))
    return results


# 权重/罚系数语义关键词（中英）。用于判断「题面是否真的给出了权重」。
_WEIGHT_KEYWORD = re.compile(
    r"(?:weight|penalt|priorit|importanc|权重|罚|优先级)", re.IGNORECASE)
_NUM = re.compile(r"\d+(?:\.\d+)?")
# 关键词后多少字符内出现的数字算作「该关键词修饰的数字」
_WEIGHT_WINDOW = 40


def stated_weights(nl_text: str) -> list[float]:
    """提取题面中「带权重语义」的数字，作为候选权重。

    为什么必须带语义邻近：实测 OR-Bench 排班 text_only 题面完全没有权重，
    但权重取值 {1,2,3,4} 恰好都作为时长/释放时间出现过（`dur 3`、`rel 2`）。
    若只做数值匹配，会把它们误当成权重，进而把「模型无法知道权重」
    错判为「模型忽略权重」——这是检测器自身的假阳性。
    """
    out: list[float] = []
    for m in _WEIGHT_KEYWORD.finditer(nl_text):
        seg = nl_text[m.end(): m.end() + _WEIGHT_WINDOW]
        out.extend(float(x) for x in _NUM.findall(seg))
    return out


# 结构化权重提取：权重键（中英）与「对象标签」键。
_WEIGHT_KEYS = ("weight", "priority", "penalty", "权重", "罚", "优先级", "importance")
_LABEL_KEYS = ("id", "name", "job_id", "task_id", "item", "label", "对象", "名称", "编号")


def structured_weight_map(nl_text: str) -> dict[str, float] | None:
    """尝试从题面**可解析的结构化文本**（JSON）提取「对象 → 权重」映射，作为强证据。

    强证据的定义（区别于 `stated_weights` 的关键词邻近扫描）：
    - 能解析出 JSON（对象列表 / 对象字典），且每条记录带**权重键**（weight/priority/…）
      与**对象标签键**（id/name/job_id/…），从而**建立对象 ↔ 权重的对应**；
    - 且得到 ≥2 个不同对象、≥2 个不同权重。

    返回 None 表示无法建立强证据（此时只能退化为弱证据 `stated_weights`）。
    这比关键词邻近扫描可靠得多：关键词扫描无法区分「权重」与「恰好邻近的时长/释放时间」，
    也无法证明对象与权重的对应关系。
    """
    if not nl_text or not nl_text.strip():
        return None

    # 尽力把整段文本（或首个 {…} 片段）解析为 JSON
    import json as _json

    candidates: list[Any] = []
    text = nl_text.strip()
    # 去掉 markdown 围栏
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        candidates.append(_json.loads(text))
    except Exception:
        pass
    # 兜底：取首个 { 到最后一个 } 之间的 JSON
    if not candidates:
        s = text.find("{")
        e = text.rfind("}")
        if s != -1 and e != -1 and e > s:
            try:
                candidates.append(_json.loads(text[s:e + 1]))
            except Exception:
                pass
    if not candidates:
        return None

    def _is_weight_key(k: str) -> bool:
        kl = str(k).lower()
        return any(w == kl for w in _WEIGHT_KEYS)

    pairs: dict[str, float] = {}

    def _collect(obj: Any) -> None:
        """从单个记录对象里取 (标签, 权重)。"""
        if not isinstance(obj, dict):
            return
        weight: float | None = None
        label: str | None = None
        for k, v in obj.items():
            if _is_weight_key(k) and isinstance(v, (int, float)) and not isinstance(v, bool):
                weight = float(v)
            elif str(k).lower() in _LABEL_KEYS and isinstance(v, (str, int)):
                label = str(v)
        if weight is not None and label is not None:
            pairs[label] = weight

    def _walk(node: Any) -> None:
        if isinstance(node, list):
            for item in node:
                _walk(item)
        elif isinstance(node, dict):
            # 顶层 dict 可能是 {对象: 权重}、或 {对象: {weight: N}}、或 {list_key: [...]}
            looks_like_records = False
            for k, v in node.items():
                if isinstance(v, dict) and any(_is_weight_key(kk) for kk in v):
                    looks_like_records = True
                    _collect(v)  # {对象: {weight: N, ...}}
            if looks_like_records:
                return
            # 顶层 dict 本身可能是单条记录（带权重键 + 标签键）
            if any(_is_weight_key(k) for k in node):
                _collect(node)
            # 递归找列表值（如 {"jobs": [...]}）
            for v in node.values():
                _walk(v)

    _walk(candidates[0])

    # 强证据门槛：≥2 个不同对象 + ≥2 个不同权重
    if len(pairs) < 2 or len(set(pairs.values())) < 2:
        return None
    return pairs


def check_coefficient_grounding(ir: IR, nl_text: str, expected: dict) -> CheckResult:
    """校验目标系数**有题面依据**：题面给了逐项权重时，不能把它们全写成同一系数。

    典型错误（OR-Bench 排班题）：题面是 "Minimize weighted tardiness" 并给出各作业权重，
    模型却写成 ΣT_j（系数全 1），等价于丢掉权重信息、把加权目标降级为等权目标。

    **分级判定（P2 修订）**：
    - **强证据**（`structured_weight_map` 能解析出「对象 ↔ 权重」的 JSON 结构，≥2 个
      不同权重）→ 系数全相同则**硬失败**（blocking=True，阻断 all_pass）；
    - **弱证据**（仅 `stated_weights` 关键词邻近扫描，无法建立对象↔权重对应）→
      系数全相同则**仅告警**（blocking=False，不阻断 all_pass，记录供人工复核）。

    为什么弱证据只告警：关键词邻近扫描无法可靠建立「题面对象 → 变量/目标项」的对应，
    题面可能提到多个优先级数字而实际目标应等权——启发式硬失败会阻止本应接受的模型。
    这是检测器自身假阳性的再收敛。

    豁免：不可行/无界/无解题（答案是状态，目标无关）；题面**完全没给**权重时
    全用同一系数是模型无法避免的选择（记入 assumptions，而非判错）。
    """
    if expected.get("status") in ("infeasible", "unbounded", "no_solution"):
        return CheckResult(True, "无解/不可行题，跳过系数依据检查", kind="coefficient_grounding")

    coefs = [abs(float(t.coef)) for t in ir.objective.expr.terms if abs(float(t.coef)) > 1e-12]
    # 单项目标不适用（只有一个系数，无所谓「等权」）
    if len(coefs) < 2:
        return CheckResult(True, "单项目标，不判定系数依据", kind="coefficient_grounding")

    strong = structured_weight_map(nl_text)
    if strong is not None:
        cands = sorted(set(strong.values()))
        evidence = "strong"
        evidence_label = "结构化权重"
    else:
        cands = sorted(set(stated_weights(nl_text)))
        evidence = "weak"
        evidence_label = "关键词邻近"
        if len(cands) < 2:
            return CheckResult(
                True,
                "题面未给出多个权重（或未识别到），不判定系数依据",
                {"stated_weights": cands},
                kind="coefficient_grounding",
            )

    all_equal = len(set(coefs)) == 1
    if not all_equal:
        return CheckResult(
            True,
            f"目标系数有区分（题面权重候选 {len(cands)} 个）",
            {"evidence": evidence, "stated_weights": cands[:12]},
            kind="coefficient_grounding",
        )

    # 系数全相同：按证据强度分级
    if evidence == "strong":
        return CheckResult(
            False,
            f"目标系数未体现题面权重（强证据）：题面结构化给出 {len(cands)} 个不同权重 "
            f"{cands[:6]}，但目标 {len(coefs)} 项系数全为 {coefs[0]}"
            f"（等价于把加权目标降级为等权目标）",
            {"evidence": evidence, "stated_weights": cands[:12],
             "objective_coefs": sorted(set(coefs)), "nonzero_terms": len(coefs),
             "object_weights": dict(list(strong.items())[:12])},
            kind="coefficient_grounding",
            blocking=True,
        )
    # weak → 仅告警，不阻断
    return CheckResult(
        False,
        f"[告警] 目标系数可能未体现题面权重（弱证据）：题面关键词邻近出现 "
        f"{len(cands)} 个不同权重候选 {cands[:6]}，目标 {len(coefs)} 项系数全为 {coefs[0]}。"
        f"因无法建立对象↔权重对应，仅作诊断，不阻断（需人工复核题面是否真为等权）",
        {"evidence": evidence, "stated_weights": cands[:12],
         "objective_coefs": sorted(set(coefs)), "nonzero_terms": len(coefs)},
        kind="coefficient_grounding",
        blocking=False,
    )


def check_semantic_coverage(gold_ir: IR, llm_ir: IR) -> CheckResult:
    """语义覆盖校验：LLM 产出 IR 相对黄金 IR 是否漏变量/漏约束/定义域不符。

    判据变量名无关、约束顺序无关（见 `semantic.py`）。返回 kind="semantic_coverage"。
    """
    from .semantic import describe_gap, has_semantic_gap, semantic_diff

    diff = semantic_diff(gold_ir, llm_ir)
    if not has_semantic_gap(diff):
        return CheckResult(True, "语义覆盖完整", kind="semantic_coverage")

    return CheckResult(
        False,
        "语义覆盖缺失：" + describe_gap(diff),
        {"diff": diff},
        kind="semantic_coverage",
    )


def check_objective_defined(ir: IR, expected: dict) -> CheckResult:
    """校验目标函数**非退化**：不能「无项」或「所有系数为 0」。

    为什么需要这条检查（实测教训）：
    目标值对比无法发现「空目标」——空目标对任何可行解都返回同一个值（通常 0），
    当参考答案恰为 0 时就会被判「正确」。OR-Bench 排班题 SCH-002/SCH-015 即如此：
    LLM 产出的 IR 目标项数为 0（参考模型分别有 4/6 项），却因黄金值为 0 而"通过"。
    这类错解只有**结构判据**能抓到。

    豁免（视为合法）：
    - 不可行/无界/无解题：答案是状态而非目标值，目标可为空；
    - IR 未声明任何变量：没有可优化的对象；
    - 显式声明为可行性问题（`expected["feasibility"] is True`）。
    """
    if expected.get("status") in ("infeasible", "unbounded", "no_solution"):
        return CheckResult(True, "无解/不可行题，跳过目标定义检查", kind="objective_defined")
    if expected.get("feasibility") is True:
        return CheckResult(True, "可行性问题，允许空目标", kind="objective_defined")
    if not ir.variables:
        return CheckResult(True, "IR 未声明变量，跳过目标定义检查", kind="objective_defined")

    if ir.objective_is_degenerate():
        terms = ir.objective.expr.terms
        detail = "目标函数无任何项" if not terms else f"目标 {len(terms)} 项的系数全为 0"
        return CheckResult(
            False,
            f"目标函数退化：{detail}（等价于不优化任何东西，任何可行解都返回同一值）",
            {"term_count": len(terms), "nonzero_terms": ir.objective_term_count(),
             "sense": ir.objective.sense.value},
            kind="objective_defined",
        )
    return CheckResult(
        True,
        f"目标函数有效（{ir.objective_term_count()} 项）",
        kind="objective_defined",
    )


def suspicious_reference(results: list[CheckResult]) -> bool:
    """判断“疑似参考答案标注错误”的情形。

    判定条件：解可行（feasibility 通过），但目标值与参考答案不符（optimality 失败）。
    此情况下求解器已给出可行解，且目标值差异只可能来自：
    (a) 建模错误（但可行说明约束基本对）——不能完全排除；
    (b) 参考答案本身标注噪声（如 NL4OPT 用 GPT-4 转换，本身有错误）。

    ⚠️ 本函数只做**判定**，不决定后续流程。历史教训：调用方曾据此直接 `return`，
    等于在 `trusted_reference=False` 的数据集上（NL4OPT 245/245、IndustryOR 97/97）
    **关闭了整条修复链路**，且无法区分 (a) 与 (b)。
    现由 `orchestrator.run_sample` 标记 `suspicious` 后**仍允许一次重写尝试**，
    并把结果记为 `suspicious_outcome`（converged/unresolved）作为证据，
    但**绝不升级为通过**——保留「不把可疑当通过」的安全性质。
    """
    feas = next((r for r in results if r.kind == "feasibility"), None)
    opt = next((r for r in results if r.kind == "optimality"), None)
    if feas is None or opt is None:
        return False
    return feas.ok and (not opt.ok)


def all_pass(results: list[CheckResult]) -> bool:
    """所有**阻断型**校验项是否全部通过。

    非阻断项（blocking=False，如弱证据告警）即使 ok=False 也不影响判定——
    它们只作为诊断证据记录，不进入「失败 → 反思/重写」的闭环。
    """
    return all(r.ok for r in results if r.blocking)


def describe(results: list[CheckResult]) -> str:
    """把校验结果渲染为人类可读文本。

    PASS / FAIL / WARN 三态：非阻断项失败渲染为 WARN（告警），区别于会阻断闭环的 FAIL。
    """
    lines = []
    for r in results:
        if r.ok:
            mark = "PASS"
        elif r.blocking:
            mark = "FAIL"
        else:
            mark = "WARN"
        lines.append(f"[{mark}] {r.message}")
    return "\n".join(lines)
