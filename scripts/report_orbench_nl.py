"""OR-Bench 自然语言链路（T1 理解）评测报告。

与 structured-solver 路径（schema → IR，100/100）对照，本脚本分析
「prompt_en → LLM 理解 → IR → 求解 → 验证」链路的表现。

关键点：**目标值命中 ≠ 模型保真**。例如排班题黄金值为 0（零拖期）时，
漏掉不重叠约束的弱模型同样能得到 0。因此除了目标值命中，还逐题用
`semantic.semantic_diff` 对比 LLM 产出的 IR 与参考 IR（系数指纹多重集，
变量名无关），给出语义保真度。

用法：
    python -m nl2opt eval --data data/orbench_nl.jsonl --use-llm \
        --no-cache --out artifacts/eval_orbench_nl.json
    python scripts/report_orbench_nl.py [结果路径]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from nl2opt.ir import IR  # noqa: E402
from nl2opt.semantic import _constraint_multiset, _objective_signature, has_semantic_gap, semantic_diff  # noqa: E402
from convert_orbench import BUILDERS  # noqa: E402

PROBLEMS = ROOT / "data" / "or-bench-problems" / "data" / "problems.jsonl"
DEFAULT_RESULT = ROOT / "artifacts" / "eval_orbench_nl_sub30.json"
OUT = ROOT / "artifacts" / "ORBENCH_NL_REPORT.md"
TOL = 1e-4


def _load_problems() -> dict[str, dict]:
    return {r["problem_id"]: r
            for r in (json.loads(l) for l in PROBLEMS.open(encoding="utf-8") if l.strip())}


def _obj_match(res: dict, expected: dict) -> bool:
    if expected.get("status") == "infeasible":
        return res.get("status") == "infeasible"
    gold = expected.get("objective_value")
    val = res.get("objective_value")
    if gold is None or val is None:
        return False
    return abs(val - gold) <= TOL * max(1.0, abs(gold))


def _classify_divergence(ref_coefs, llm_coefs, nl_text: str = "") -> str:
    """细分「目标签名有差异」的类型，把人工判读量压到最小。

    实测（OR-Bench sub12 的 6 道 text_only）表明这些差异**绝大多数是真缺陷**，
    因此不放松判定，而是把原因说清楚：

    - `unit_weights`：题面**明确给出**了多个不同权重，LLM 却全用同一系数
      → **确定缺陷**（如排班题给了权重却写成 ΣT_j，把加权目标降级为等权目标）。
      判定必须结合题面：题面没给权重时全用同一系数是**无法避免**的，
      归入 `under_specified` 而非缺陷（实测 13 道排班题最初被误判为缺陷）；
    - `under_specified`：题面未提供逐项权重，LLM 只能等权处理
      → **非模型缺陷，属题面信息不足**（理想做法是记入 `assumptions`）；
    - `subset`：LLM 系数是参考的**子集**（没有参考中不存在的系数）→ 纯**省略**。
      被省略项若对应变量恒为 0（如必做任务的未指派罚项 `u≡0`）则**无害**；
      否则仍是缺陷。属**低成本核验**（只需查被省略项对应的变量是否恒为 0）；
    - `scaled`：LLM 系数恰为参考的**同一倍数** k≠1
      → 多为单位/口径差异（如按「箱」还是按「件」计），可能等价，但仍需确认；
    - `other`：其余情形（引入了参考中不存在的系数）→ **必须人工判读**。
    """
    rc, lc = Counter(ref_coefs), Counter(llm_coefs)
    if lc and len(lc) == 1 and len(rc) > 1:
        # 系数全相同：是「忽略权重」还是「题面没给权重」取决于题面
        try:
            from nl2opt.verifier import stated_weights
            given = len(set(stated_weights(nl_text))) >= 2
        except Exception:
            given = False
        return "unit_weights" if given else "under_specified"
    if lc and not (lc - rc):
        return "subset"          # 无新增系数，纯省略
    if rc and lc:
        k = None
        for a, b in zip(sorted(rc), sorted(lc)):
            if abs(a) > 1e-12:
                k = b / a
            break
        if k is not None and abs(k - 1.0) > 1e-9:
            scaled = Counter({round(c * k, 6): n for c, n in rc.items()})
            if scaled == lc:
                return "scaled"
    return "other"


def main() -> int:
    ap = argparse.ArgumentParser(description="OR-Bench 自然语言链路报告")
    ap.add_argument("result", nargs="?", default=str(DEFAULT_RESULT), help="评测结果 JSON")
    ap.add_argument("--out", default=str(OUT),
                    help="报告输出路径（默认 artifacts/ORBENCH_NL_REPORT.md；"
                         "多批次分析时用 --out 分别留存，避免互相覆盖）")
    args = ap.parse_args()

    result_path = Path(args.result)
    if not result_path.exists():
        print(f"缺少评测结果: {result_path}", file=sys.stderr)
        return 2

    results = json.loads(result_path.read_text(encoding="utf-8"))
    problems = _load_problems()
    ds = {json.loads(l)["id"]: json.loads(l)
          for l in (ROOT / "data" / "orbench_nl.jsonl").open(encoding="utf-8") if l.strip()}

    cells = defaultdict(lambda: {"n": 0, "built": 0, "obj": 0, "sem": 0, "repaired": 0})
    rounds_hist: Counter = Counter()
    tokens = Counter()
    fail_modes: list[tuple[str, str, str]] = []
    sem_false_pos: list[str] = []   # （保留兼容）
    degenerate_cases: list[tuple[str, bool]] = []  # (题号, 是否巧合命中黄金值)
    divergent_review: list[str] = []  # 目标值命中但目标签名有差异（待人工判读）
    ir_expressiveness: list[str] = []  # 因 IR 表达力（索引变量族）失败
    recs: list[dict] = []             # 逐样本记录（供判别力分层统计）
    divergence_kinds: dict[str, int] = {}          # 目标差异细分计数
    divergence_hits: dict[str, list[str]] = {}     # 各细分下「目标值仍命中」的题号

    _KIND_LABEL = {
        "unit_weights": "`unit_weights`｜忽略权重：**题面已给权重**，LLM 却全用同一系数（**缺陷**）",
        "under_specified": "`under_specified`｜题面未给权重（**非模型缺陷**，属信息不足）",
        "subset": "`subset`｜纯省略：无参考中不存在的系数（被省略项若变量恒为 0 则**无害**）",
        "scaled": "`scaled`｜整体同倍数：可能只是单位/口径差异（仍需确认）",
        "other": "`other`｜系数错配：引入了参考中不存在的系数（**必须人工判读**）",
    }
    _KIND_ORDER = ("unit_weights", "under_specified", "other", "subset", "scaled")
    details: list[str] = []

    for res in results:
        pid = res["id"]
        meta = ds[pid]["extra"]
        fam, variant = meta["family"], meta["prompt_variant"]
        cell = cells[(fam, variant)]
        cell["n"] += 1
        rounds_hist[res.get("rounds", 0)] += 1
        if res.get("rounds", 0) > 1:
            cell["repaired"] += 1
        for key, val in (res.get("token_usage") or {}).items():
            if isinstance(val, (int, float)):
                tokens[key] += val

        # 1) 是否产出有效 IR
        final_ir = res.get("final_ir")
        if not final_ir:
            raw_err = " | ".join(str(e) for e in (res.get("errors") or []))
            errs = raw_err[:160]
            cell["built"] += 0
            fail_modes.append((pid, f"{fam}/{variant}", f"未产出 IR：{errs}"))
            if "index_set" in raw_err:
                # IR 表达力受限：LLM 想用字符串/多维索引变量族，但 IR 只支持一维整数
                ir_expressiveness.append(pid)
            recs.append({"id": pid, "family": fam, "variant": variant, "obj_ok": False,
                         "built": False, "rounds": 0, "degenerate": False})
            details.append(f"| {pid} | {fam} | {variant} | - | ✗未建模 | - | - |")
            continue
        cell["built"] += 1

        # 2) 目标值命中
        obj_ok = _obj_match(res, ds[pid]["expected"])
        if obj_ok:
            cell["obj"] += 1

        # 3) 结构保真度：以「目标签名」为主判据（顺序/变量名/化简形式均无关）
        #    exact  = 目标系数多重集与参考完全一致 → 建模口径一致
        #    subset = LLM 目标项是参考的子集且非空 → 可能是无害省略（如必做任务的罚项恒为 0）
        #    defect = 目标项为空（空洞目标，靠 0 值巧合命中）或引入了参考中不存在的系数
        try:
            llm_ir = IR.model_validate(final_ir)
            ref_ir, _ = BUILDERS[fam](problems[pid]["schema_payload"])
            diff = semantic_diff(ref_ir, llm_ir)  # 仅用于诊断记录

            obj_ref = _objective_signature(ref_ir)
            obj_llm = _objective_signature(llm_ir)
            ref_terms, llm_terms = Counter(obj_ref[1]), Counter(obj_llm[1])

            # 退化目标：无项，或所有系数为 0（等价于「最小化 0」，任何可行解都返回 0）
            # 统一使用 IR.objective_is_degenerate()，与 verifier.check_objective_defined 同源
            degenerate = llm_ir.objective_is_degenerate()

            if obj_ref == obj_llm:
                level = "exact"
            elif degenerate:
                level = "empty"
            else:
                # 细分差异类型，避免把「系数整体错配」与「仅单位不同」混为一谈
                level = _classify_divergence(obj_ref[1], obj_llm[1], (ds.get(pid) or {}).get("nl_text") or "")
                divergence_kinds[level] = divergence_kinds.get(level, 0) + 1
                if obj_ok:
                    divergence_hits.setdefault(level, []).append(pid)

            cons_ref = _constraint_multiset(ref_ir)
            cons_llm = _constraint_multiset(llm_ir)
            ref_total = sum(cons_ref.values()) or 1
            overlap = sum((cons_ref & cons_llm).values()) / ref_total

            if level == "exact":
                cell["sem"] += 1
            if level == "empty":
                # 退化目标无论是否巧合命中黄金值，都是明确缺陷
                degenerate_cases.append((pid, bool(obj_ok)))
            if obj_ok and level == "divergent":
                divergent_review.append(pid)   # 目标命中但结构有差异，需人工判读

            n_ref_t, n_llm_t = sum(ref_terms.values()), sum(llm_terms.values())
            if level == "exact":
                sem_txt = "一致"
            elif level == "empty":
                sem_txt = f"**退化目标**(项数{n_llm_t}/{n_ref_t}，系数全 0 或无项)"
            else:
                sem_txt = f"目标有差异({n_llm_t}/{n_ref_t})、约束重合{overlap:.0%}"
            counts = f"{len(ref_ir.all_var_instances)}/{len(llm_ir.all_var_instances)}"
        except Exception as exc:
            sem_txt = f"对比失败({type(exc).__name__})"
            counts = "-"

        obj_txt = "✓" if obj_ok else f"值{res.get('objective_value')}"
        recs.append({"id": pid, "family": fam, "variant": variant, "obj_ok": obj_ok,
                     "built": True, "rounds": res.get("rounds", 0),
                     "degenerate": level == "empty" if final_ir else False})
        details.append(f"| {pid} | {fam} | {variant} | {res.get('rounds')} | {obj_txt} | {sem_txt} | "
                       f"{counts} | {res.get('status')} |")

    total = len(results)
    built = sum(c["built"] for c in cells.values())
    obj_all = sum(c["obj"] for c in cells.values())
    sem_all = sum(c["sem"] for c in cells.values())
    repaired = sum(c["repaired"] for c in cells.values())

    lines: list[str] = []
    lines.append("# OR-Bench 自然语言链路（T1 理解）评测报告\n")
    lines.append("- 链路：`prompt_en` → LLM 理解 → IR → 求解 → 独立验证")
    lines.append("- 对照：structured-solver 路径（schema → IR）为 **100/100**（见 `ORBENCH_REPORT.md`）")
    lines.append("- 判定：目标值容差 `1e-4 × max(1,|gold|)`；语义保真用系数指纹多重集比对（变量名无关）")
    lines.append(f"- 结果文件：`{result_path.name}`\n")

    lines.append("## 一、总体结果\n")
    lines.append("| 指标 | 值 |")
    lines.append("|---|---|")
    lines.append(f"| 总题数 | {total} |")
    lines.append(f"| 产出有效 IR | **{built}/{total}** ({built/total*100:.1f}%) |")
    lines.append(f"| 目标值命中黄金值（名义） | **{obj_all}/{total}** ({obj_all/total*100:.1f}%) |")
    # 三档口径：把「目标有确定缺陷、却因黄金值巧合而命中」的假象逐层扣除。
    # 确定缺陷 = 目标为空(empty) / 忽略权重(unit_weights) / 系数错配(other)；
    # subset 与 scaled 可能无害（见第六节），故不扣除，仅列为待核验。
    vacuous = sum(1 for _, matched in degenerate_cases if matched)
    # 确定缺陷：目标为空 / 题面已给权重却忽略 / 系数错配
    defect_matched = ({pid for pid, m in degenerate_cases if m}
                      | set(divergence_hits.get("unit_weights", []))
                      | set(divergence_hits.get("other", [])))
    # 题面信息不足：题面未提供逐项权重，模型只能等权处理 —— **不算模型缺陷**
    underspec_matched = set(divergence_hits.get("under_specified", []))
    if defect_matched:
        parts = []
        if vacuous:
            parts.append(f"空洞 {vacuous}")
        n_uw = len(divergence_hits.get("unit_weights", []))
        n_ot = len(divergence_hits.get("other", []))
        if n_uw:
            parts.append(f"忽略权重 {n_uw}")
        if n_ot:
            parts.append(f"系数错配 {n_ot}")
        lines.append(f"| └ 扣除**确定缺陷且巧合命中**（{('、'.join(parts))}） | −{len(defect_matched)} |")
        substantive = obj_all - len(defect_matched)
        lines.append(f"| **实质通过**（扣除确定缺陷后） | **{substantive}/{total}** "
                     f"({substantive/total*100:.1f}%) |")
    else:
        substantive = obj_all
    # 题面信息不足独立于「确定缺陷」显示：即使缺陷为 0，也要如实呈现
    # （题面没给的信息模型不可能知道，既不算通过也不算模型的错）
    if underspec_matched:
        lines.append(f"| └ 其中含**题面信息不足**（题面未给权重，非模型缺陷） | "
                     f"{len(underspec_matched)} |")
        controllable = substantive - len(underspec_matched)
        lines.append(f"| **模型可控通过**（再扣除信息不足） | **{controllable}/{total}** "
                     f"({controllable/total*100:.1f}%) |")
    lines.append(f"| **严格通过**（目标签名与参考完全一致） | **{sem_all}/{total}** "
                 f"({sem_all/total*100:.1f}%) |")
    pending = len(divergence_hits.get("subset", [])) + len(divergence_hits.get("scaled", []))
    if pending:
        lines.append(f"| 待核验（纯省略/同倍数，可能无害） | {pending} |")
    lines.append(f"| 经多轮重写修复 | {repaired}/{total} |")
    lines.append(f"| 平均轮次 | {sum(k*v for k,v in rounds_hist.items())/total:.2f} |")
    if tokens:
        lines.append(f"| Token 合计 | prompt {int(tokens['prompt_tokens'])} + "
                     f"completion {int(tokens['completion_tokens'])} = "
                     f"**{int(tokens['total_tokens'])}** |")
    lines.append("")

    # 黄金值为 0 的题会让「目标值命中」失去区分力（退化模型也返回 0）→ 显著警告
    gold_zero = [r["id"] for r in results
                 if ds.get(r["id"]) and ds[r["id"]]["expected"].get("objective_value") == 0]
    if gold_zero:
        frac = len(gold_zero) / max(1, len(results))
        lines.append(f"### ⚠️ 度量有效性警告\n")
        lines.append(f"本批 {total} 题中 **{len(gold_zero)} 题黄金值为 0**（占 {frac*100:.0f}%）。"
                     "对这类题，「目标值命中」**几乎没有区分力**——任何可行解、甚至目标退化为 0 的"
                     "空模型都会返回 0 而被判「命中」。\n")
        lines.append("因此本批结果必须**以「目标签名一致」为主导指标**，"
                     "「目标值命中」仅作参考（见第五、六节的缺陷与待判读清单）：\n")
        lines.append("`" + "`, `".join(sorted(gold_zero)[:20])
                     + ("` …\n" if len(gold_zero) > 20 else "`\n"))

    # 「判别力分层」：黄金值为 0 的题无法区分「真建模」与「弱模型/空模型」，
    # 因此单独统计真正有判别力的子集（黄金值非 0 或不可行）。
    def _discriminating(pid: str) -> bool:
        exp = (ds.get(pid) or {}).get("expected") or {}
        if exp.get("status") == "infeasible":
            return True
        v = exp.get("objective_value")
        return v is not None and abs(float(v)) > 1e-12

    disc = [r for r in recs if _discriminating(r["id"])]
    nondisc = [r for r in recs if not _discriminating(r["id"])]

    lines.append("## 三、判别力分层（关键）\n")
    lines.append("黄金值为 0 的题**无法区分**「真建模」与「弱模型」——两者都返回 0。"
                 "因此把样本分成两层分别看，才能得出可信结论：\n")
    lines.append("| 分层 | 题数 | 产出 IR | 目标值命中 | 通过率 |")
    lines.append("|---|---|---|---|---|")
    for label, group in (("**可判别**（黄金值≠0 或不可行）", disc), ("不可判别（黄金值=0）", nondisc)):
        if not group:
            continue
        g_ok = sum(1 for r in group if r["obj_ok"])
        g_built = sum(1 for r in group if r["built"])
        lines.append(f"| {label} | {len(group)} | {g_built}/{len(group)} | {g_ok}/{len(group)} | "
                     f"**{g_ok/len(group)*100:.1f}%** |")
    lines.append("")
    if disc:
        bad = [r["id"] for r in disc if not r["obj_ok"]]
        vac_disc = [pid for pid, m in degenerate_cases if m and _discriminating(pid)]
        if bad:
            lines.append(f"可判别层的失败题：`" + "`, `".join(bad) + "`\n")
            lines.append("这类失败**最有诊断价值**：通常是模型**约束不足**（可行域过大 → 目标值偏低），"
                         "或把不可行题误建为可行。\n")
        else:
            lines.append("可判别层全部通过。\n")
        if vac_disc:
            lines.append(f"⚠️ 注意：可判别层有 **{len(vac_disc)} 题目标为空却仍命中**"
                         f"（`" + "`, `".join(vac_disc) + "`），应视为失败而非通过。\n")

    lines.append("## 四、分题族 × prompt 变体\n")
    lines.append("> 变体：`schema_embedded` = 题面内嵌完整 JSON schema（60 题）；"
                 "`text_only` = 仅自然语言 + 紧凑字段行（40 题）\n")
    lines.append("| 题族 | 变体 | 题数 | 产出 IR | 目标值命中 | 结构吻合 | 多轮修复 |")
    lines.append("|---|---|---|---|---|---|---|")
    for fam in ("inventory", "assignment", "scheduling"):
        for variant in ("schema_embedded", "text_only"):
            c = cells.get((fam, variant))
            if not c:
                continue
            lines.append(f"| {fam} | {variant} | {c['n']} | {c['built']}/{c['n']} | "
                         f"{c['obj']}/{c['n']} | {c['sem']}/{c['n']} | {c['repaired']} |")
    lines.append("")

    lines.append("## 五、目标退化缺陷\n")
    if degenerate_cases:
        lines.append("LLM 的目标函数**没有有效变量项**（无项，或所有系数为 0，等价于「最小化 0」）——"
                     "这不是建模，任何可行解都会返回 0：\n")
        lines.append("| 题号 | 是否巧合命中黄金值 |")
        lines.append("|---|---|")
        for pid, matched in sorted(degenerate_cases):
            lines.append(f"| {pid} | {'⚠️ 命中（黄金值为 0，靠巧合通过）' if matched else '未命中'} |")
        lines.append("")
        lines.append("> 排班题黄金值常为 0，**命中**的那类最危险：目标值指标会判它「正确」，"
                     "但实际未建模任何目标。这正是必须配结构判据的原因。\n")
    else:
        lines.append("无。\n")

    if divergent_review or divergence_kinds:
        lines.append("## 六、目标签名有差异（按原因细分）\n")
        lines.append("**不放松判定**：实测这批差异绝大多数是**真缺陷**——差异只说明「模型与参考不等价」，"
                     "不构成「通过」。细分原因是为了压缩人工判读量，而不是给通过找理由。\n")
        lines.append("| 差异类型 | 题数 | 其中「目标值仍命中」（假象） |")
        lines.append("|---|---|---|")
        for kind in _KIND_ORDER:
            n = divergence_kinds.get(kind, 0)
            if not n:
                continue
            hits = divergence_hits.get(kind, [])
            lines.append(f"| {_KIND_LABEL[kind]} | {n} | "
                         f"{('`' + '`, `'.join(hits) + '`') if hits else '—'} |")
        lines.append("")
        if divergence_hits.get("unit_weights"):
            lines.append(f"⚠️ `unit_weights` 是**确定的缺陷**：题面给了权重/差异成本，模型却全用同一系数。"
                         f"受影响的题（`{'`, `'.join(divergence_hits['unit_weights'])}`）"
                         f"在「目标值」口径下会**误判为通过**（黄金值恰为 0）。\n")
        if divergence_hits.get("other"):
            lines.append(f"`other`（引入了参考中不存在的系数）**必须人工核对题面**："
                         f"`{'`, `'.join(divergence_hits['other'][:8])}`。"
                         f"提示：把 LLM 的目标系数与题面数据逐项对照即可，不必重跑。\n")
        if divergence_hits.get("subset"):
            lines.append(f"`subset`（纯省略）可**低成本核验**：只需确认被省略项对应的变量是否恒为 0"
                         f"（如必做任务的未指派罚项 `u≡0`，省略即无害）。涉及："
                         f"`{'`, `'.join(divergence_hits['subset'][:8])}`。\n")
        if divergence_kinds.get("scaled"):
            lines.append("`scaled` 是唯一「可能无害」的类型（整体乘同一倍数，常见于单位/口径差异），"
                         "但也可能是量纲错误，仍需确认。\n")
        lines.append("> 反向局限：目标签名一致**也不充分**——实测有 1 例（v4-pro 的 SCH-002）"
                     "签名一致却求解失败，说明误差在约束侧。任何单一指标都不能证明模型正确。\n")

    if fail_modes:
        lines.append("## 七、失败模式\n")
        lines.append("| 题号 | 题族/变体 | 原因 |")
        lines.append("|---|---|---|")
        for pid, fv, why in fail_modes:
            lines.append(f"| {pid} | {fv} | {why} |")
        lines.append("")

    if ir_expressiveness:
        lines.append("## 八、IR 表达力限制导致的失败（非建模能力问题）\n")
        lines.append("以下题目的理解失败**不是** LLM 建模能力不足，而是本项目 IR 的表达力不足："
                     "LLM 采用了最自然的建模方式——用**字符串/多维索引的变量族**"
                     "（如排班的 `y[job][machine]`、按作业名索引的 `start[J1]`），"
                     "而 IR 的 `index_set` 只支持**一维整数**，因此在 Schema 校验阶段即被拒绝：\n")
        lines.append("`" + "`, `".join(ir_expressiveness) + "`\n")
        lines.append("改进方向（三选一或组合）：")
        lines.append("1. `index_set` 放宽为「任意标量的列表」（允许字符串 ID）；")
        lines.append("2. 支持多维索引（`index_set` 为列表的列表，实例名按 `基名_维度1_维度2` 展开）；")
        lines.append("3. 在理解环节的 System Prompt 中明确要求「不支持字符串索引，"
                     "请把 `y[job][machine]` 展开为多个标量变量或改用整数索引」。\n")

    lines.append("## 九、逐题明细\n")
    lines.append("| 题号 | 题族 | 变体 | 轮次 | 目标值 | 结构 | 变量数 ref/llm | 状态 |")
    lines.append("|---|---|---|---|---|---|---|---|")
    lines.extend(details)
    lines.append("")

    lines.append("## 十、保真度度量说明（重要）\n")
    lines.append("以**目标签名**为主判据（`sense` + 目标系数多重集），它对变量名、约束顺序、"
                 "以及「等价但更精简」的写法都鲁棒：\n")
    lines.append("| 判级 | 含义 |")
    lines.append("|---|---|")
    lines.append("| `exact` | 目标系数多重集与参考完全一致 → 建模口径一致 |")
    lines.append("| `divergent` | 目标项/系数与参考有差异 → **待人工判读**（可能无害省略，也可能有害） |")
    lines.append("| `empty` | 目标项为空（等价于「最小化 0」）→ **明确缺陷**，靠巧合命中黄金值 0 |")
    lines.append("")
    lines.append("> ℹ️ `semantic.py` 已修复**顺序敏感性**（原实现按「约束顺序」取系数，"
                 "同一模型仅约束顺序不同就会误报「漏变量」）：现改用**顺序无关的变量结构角色多重集**"
                 "（目标系数 + 参与约束的「自身系数 × 约束签名[排除自身]」）"
                 "并纳入**归一化定义域**（可检出整数/连续、上下界差异）。"
                 "仍存在的固有局限：本模块做的是**结构**比对，数学等价但写法不同"
                 "（如代入消元合并约束、拆分变量）仍会报告差异——"
                 "INV-001 即属此类（变量 24→18、约束 16→10，目标签名一致），故本报告"
                 "以目标签名为主要判据、结构差异列为待人工判读。\n")

    lines.append("## 十一、结论与解读\n")
    lines.append("1. **难度分层明显**：内嵌 JSON schema 的题（`schema_embedded`）本质是"
                 "「JSON 提取 + 语义映射」，纯文本题（`text_only`）才是真正的自然语言理解。"
                 "两者的差距直接量化了「schema 提示」的价值。")
    lines.append("2. **目标值命中 ≠ 建模正确**：见第三节。"
                 "因此报告同时给出语义保真度（与参考 IR 的系数指纹比对）。")
    lines.append("3. **排班题是最大挑战**：本项目 IR 只支持线性约束，"
                 "排班的不重叠约束必须由模型自行推导 big-M 析取编码，"
                 "这与 structured-solver 路径（程序化生成该编码，100% 正确）形成鲜明对照。")
    lines.append("4. **token 预算是隐性瓶颈**：OR-Bench 的模型规模（多期库存、big-M 排班）"
                 "远超 NL4OPT 的小 LP，`max_tokens` 不足会让 reasoning 吃光输出预算、"
                 "JSON 被截断；本报告已含截断检测（`finish_reason=length`）。")
    lines.append("")

    Path(args.out).write_text("\n".join(lines), encoding="utf-8")
    print(f"报告已写入: {args.out}")
    print(f"产出 IR {built}/{total} | 名义命中 {obj_all}/{total} | 实质通过 "
          f"{obj_all - len(defect_matched)}/{total} | 严格通过 {sem_all}/{total} "
          f"| 确定缺陷命中 {len(defect_matched)} | 待核验 {pending}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
