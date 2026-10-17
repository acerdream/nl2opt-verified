"""NL4OPT 基线对比：历史全量（v1 代码）vs 新批次（分层提示词），**按题号对齐**。

为什么必须按题对齐：跨版本整体均值不可比——v1 之后还改过验证器判据、容差口径、
max_tokens、提示词结构等。按题对齐后能直接看出「哪些题变好/变差」，比均值更有信息量。

**必须声明的混淆**（脚本会写进报告）：
- 历史运行 `reasoning_effort=high`，当前配置 `medium`；
- 历史运行无 `check_objective_defined` / `check_coefficient_grounding`，
  目标值容差也未与可行性容差解耦；
- 历史 `max_tokens=8192`（会把大题的 JSON 截断），当前 32768；
- 提示词结构不同（单体 vs 分层）。
因此差异**不能**单独归因于提示词分层。若要看分层的净影响，需在同一份代码下
再跑 `NL2OPT_ALL_CARDS=1`（≈ 原单体提示词）作对照。

用法：
    python scripts/compare_nl4opt.py \
        --baseline artifacts/eval_nl4opt_full.json \
        --new artifacts/eval_nl4opt_sub50_v5.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# 历史判定 → 统一语义
#   OK         : 可行且目标值一致
#   SUSPICIOUS : 可行但目标值不符（判为参考答案标注噪声）
#   FAIL       : 真失败（如参考答案称无解而求解器找到最优解，或理解/求解失败）
LABEL = {"OK": "通过", "SUSPICIOUS": "疑似参考噪声", "FAIL": "失败"}


def _new_label(res: dict) -> str:
    if res.get("success"):
        return "通过"
    if res.get("suspicious"):
        return "疑似参考噪声"
    return "失败"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="NL4OPT 基线按题对比")
    ap.add_argument("--baseline", default=str(ROOT / "artifacts" / "eval_nl4opt_full.json"))
    ap.add_argument("--new", required=True)
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "NL4OPT_COMPARE.md"))
    args = ap.parse_args(argv)

    base = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
    base_map = {x["id"]: x for x in base.get("detail", [])}
    new = json.loads(Path(args.new).read_text(encoding="utf-8"))
    new_map = {r["id"]: r for r in new if isinstance(r, dict)}

    ids = [i for i in new_map if i in base_map]
    if not ids:
        print("无可对齐的题号", file=sys.stderr)
        return 2
    ids.sort()

    trans: dict[tuple[str, str], list[str]] = {}
    for i in ids:
        a = LABEL.get(base_map[i]["result"], base_map[i]["result"])
        b = _new_label(new_map[i])
        trans.setdefault((a, b), []).append(i)

    cb, cn = {}, {}
    for i in ids:
        cb[LABEL.get(base_map[i]["result"], "?")] = cb.get(LABEL.get(base_map[i]["result"], "?"), 0) + 1
        cn[_new_label(new_map[i])] = cn.get(_new_label(new_map[i]), 0) + 1

    lines = ["# NL4OPT 基线对比（按题号对齐）\n"]
    lines.append(f"- 基线：`{Path(args.baseline).name}`"
                 f"（{base.get('model')} / reasoning_effort={base.get('reasoning_effort')} / "
                 f"全量 {base.get('n')} 题）")
    lines.append(f"- 新批次：`{Path(args.new).name}`（当前代码 + 分层提示词）")
    lines.append(f"- 对齐题数：**{len(ids)}**\n")

    lines.append("## 一、总体\n")
    lines.append("| 判定 | 基线 | 新批次 |")
    lines.append("|---|---|---|")
    for k in ("通过", "疑似参考噪声", "失败"):
        lines.append(f"| {k} | {cb.get(k, 0)} | {cn.get(k, 0)} |")
    p0, p1 = cb.get("通过", 0), cn.get("通过", 0)
    lines.append(f"| **通过率** | **{p0/len(ids)*100:.1f}%** | **{p1/len(ids)*100:.1f}%** |")
    lines.append("")

    lines.append("## 二、判定迁移\n")
    lines.append("| 基线 → 新批次 | 题数 | 题号 |")
    lines.append("|---|---|---|")
    for (a, b), lst in sorted(trans.items(), key=lambda kv: -len(kv[1])):
        tag = "" if a == b else ("  ✅变好" if (a, b) == ("失败", "通过") or
                                 (a, b) == ("疑似参考噪声", "通过") else "  ⚠️变差")
        lines.append(f"| {a} → {b}{tag} | {len(lst)} | `" + "`, `".join(lst[:12])
                     + ("` …" if len(lst) > 12 else "`") + " |")
    lines.append("")

    better = [i for (a, b), lst in trans.items() if a != "通过" and b == "通过" for i in lst]
    worse = [i for (a, b), lst in trans.items() if a == "通过" and b != "通过" for i in lst]
    lines.append("## 三、净变化\n")
    lines.append(f"- 变好（→通过）：**{len(better)}** 题"
                 + (f"：`" + "`, `".join(better) + "`" if better else ""))
    lines.append(f"- 变差（通过→）：**{len(worse)}** 题"
                 + (f"：`" + "`, `".join(worse) + "`" if worse else ""))
    lines.append("")

    # ---- 归因：变好的题里有多少其实来自「参考答案修正」而非本次改动 ----
    ds = {}
    dpath = ROOT / "data" / "nl4opt.jsonl"
    if dpath.exists():
        for line in dpath.open(encoding="utf-8"):
            if line.strip():
                row = json.loads(line)
                ds[row["id"]] = row.get("expected") or {}
    ref_fixed, genuine = [], []
    for i in better:
        b = base_map[i]
        exp = (ds.get(i) or {}).get("objective_value")
        val = new_map[i].get("objective_value")
        # 只有「基线判 SUSPICIOUS（可行但值不符）」+「新值与当前参考值相等」
        # 才归因于参考答案修正。基线若是 FAIL 且 obj=None（求解失败），
        # 那是能力问题而非标注问题，不能算作参考修正。
        if (b.get("result") == "SUSPICIOUS" and exp is not None and val is not None
                and abs(val - exp) <= 1e-6):
            ref_fixed.append(i)
        else:
            genuine.append(i)
    lines.append("## 四、变好题目的归因（重要）\n")
    lines.append(f"- **疑为参考答案修正所致：{len(ref_fixed)} 题**"
                 + (f"（`" + "`, `".join(ref_fixed) + "`）" if ref_fixed else "")
                 + "——这些题的求解值与**当前**参考值精确相等，说明基线判「疑似参考噪声」"
                   "是因为**当时的参考值不同**，而非模型变好了。")
    lines.append(f"- 可能源于本次改动：**{len(genuine)} 题**"
                 + (f"（`" + "`, `".join(genuine) + "`）" if genuine else ""))
    lines.append("")
    lines.append("> 依据：`data/nl4opt.jsonl` 的参考答案曾由 `scripts/fix_nl4opt_full.py` 修正"
                 "（修正「经独立求解器证伪」的标注），而基线运行早于该修正。"
                 "因此**表面通过率的提升不能归因于提示词分层**。\n")

    lines.append("## 五、⚠️ 混淆声明（解读前必读）\n")
    lines.append("两次运行**不止提示词不同**，因此差异不能单独归因于提示词分层：\n")
    lines.append("| 维度 | 基线（v1 代码） | 新批次 | 影响方向 |")
    lines.append("|---|---|---|---|")
    lines.append(f"| 思考强度 | reasoning_effort={base.get('reasoning_effort')} | 当前配置 medium | 未知 |")
    lines.append("| 验证器判据 | 无目标定义/系数依据检查 | 已加入 | 新批次更严（可能压低通过率） |")
    lines.append("| 目标值容差 | 与可行性容差共用 1e-6 | 解耦为相对容差 1e-4 | 新批次更宽容（可能提高通过率） |")
    lines.append("| max_tokens | 8192（大题会被截断） | 32768 | 新批次更有利 |")
    lines.append("| 提示词 | 单体 3075 字符 | 分层（核心+卡片） | 待测 |")
    lines.append("")
    lines.append("> 若净变化接近 0 或为正，可保守地认为「分层未损害 NL4OPT」；"
                 "若明显为负，需用 `NL2OPT_ALL_CARDS=1`（≈ 原单体提示词）在同一份代码下再跑一次，"
                 "以隔离提示词分层的净影响。\n")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"报告已写入: {out}")
    print(f"对齐 {len(ids)} 题｜基线通过 {p0}（{p0/len(ids)*100:.1f}%）→ 新 {p1}（{p1/len(ids)*100:.1f}%）")
    print(f"变好 {len(better)}｜变差 {len(worse)}")
    if better:
        print(f"  变好: {better}")
    if worse:
        print(f"  变差: {worse}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
