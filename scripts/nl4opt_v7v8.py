"""NL4OPT v7 → v8 对比：重点是「疑似噪声」里有多少题其实能修到参考值。

v8 的三项改动都会影响结果，需分开归因：
  ① 疑似短路修复 → 疑似噪声题现在会尝试一次重写（**本脚本关注点**）
  ② 目标常数修复 → 含非零常数的题目标值会变（NL4OPT 245 题中 0 题，预期无影响）
  ③ 新增 fixed_cost/piecewise 卡片 → 触发该两卡的题提示词不同（NL4OPT 命中 2+7 题）
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from nl2opt.pipeline.prompts import select_cards  # noqa: E402

V7 = ROOT / "artifacts" / "eval_nl4opt_full_v7_layered.json"
V8 = ROOT / "artifacts" / "eval_nl4opt_full_v8.json"


def load(p):
    return {r["id"]: r for r in json.loads(Path(p).read_text(encoding="utf-8"))}


def outcome(r):
    """疑似参考噪声的结局：优先显式字段，回退 trajectory（旧批次）。"""
    if r.get("status") == "suspicious_resolved" or r.get("suspicious_resolved"):
        return "converged"
    for s in reversed(r.get("trajectory") or []):
        if isinstance(s, dict) and s.get("suspicious_outcome"):
            return s["suspicious_outcome"]
    return None


def main() -> int:
    if not V8.exists():
        print(f"缺少 {V8.name}（v8 评测尚未完成）", file=sys.stderr)
        return 2
    a, b = load(V7), load(V8)
    ids = sorted(set(a) & set(b))
    ds = {json.loads(l)["id"]: json.loads(l)
          for l in (ROOT / "data" / "nl4opt.jsonl").open(encoding="utf-8") if l.strip()}

    pa = sum(1 for i in ids if a[i]["success"])
    pb = sum(1 for i in ids if b[i]["success"])
    print(f"对齐 {len(ids)} 题")
    print(f"  通过：v7 {pa} → v8 {pb}（{pb-pa:+d}）")

    sa = [i for i in ids if a[i].get("suspicious") and not a[i]["success"]]
    sb = [i for i in ids if b[i].get("suspicious") and not b[i]["success"]]
    print(f"  疑似噪声：v7 {len(sa)} → v8 {len(sb)}")

    conv = [i for i in sb if outcome(b[i]) == "converged"]
    unres = [i for i in sb if outcome(b[i]) == "unresolved"]
    print(f"  └ v8 中【修复后达到参考值】: **{len(conv)}** 题 {conv}")
    print(f"  └ v8 中【仍未达到】      : {len(unres)} 题")
    print()

    only_a = [i for i in ids if a[i]["success"] and not b[i]["success"]]
    only_b = [i for i in ids if b[i]["success"] and not a[i]["success"]]
    print(f"仅 v7 通过 {len(only_a)}: {only_a}")
    print(f"仅 v8 通过 {len(only_b)}: {only_b}")
    print()

    # 归因：这些变化题是否受新卡片影响（②③）
    print("=== 新增卡片的触发情况（③ 的影响面）===")
    fc = [i for i in ids if "fixed_cost" in select_cards(ds[i]["nl_text"])]
    pw = [i for i in ids if "piecewise" in select_cards(ds[i]["nl_text"])]
    print(f"  fixed_cost 命中 {len(fc)} 题: {fc}")
    print(f"  piecewise  命中 {len(pw)} 题: {pw}")
    print(f"  两者并集 {len(set(fc) | set(pw))} 题（仅这些题的两版提示词不同）")
    print()

    # 疑似题的收敛细节
    print("=== v8 中「修复后达到参考值」的题（旧实现会当参考噪声放过）===")
    for i in conv:
        r = b[i]
        exp = (ds[i].get("expected") or {}).get("objective_value")
        old = "（v7 也疑似噪声）" if i in sa else "（v7 非疑似）"
        print(f"  {i}: obj={r['objective_value']} 期望={exp} rounds={r['rounds']} {old}")
    print()

    # 仍疑似噪声的题：参考是否真不可达
    print("=== v8 中「仍未达到」的题（疑似噪声解释站得住）===")
    for i in unres[:25]:
        r = b[i]
        exp = (ds[i].get("expected") or {}).get("objective_value")
        val = r.get("objective_value")
        gap = ""
        if exp not in (None, 0) and val is not None:
            gap = f"｜相对差 {abs(val-exp)/max(1e-9,abs(exp))*100:.2f}%"
        print(f"  {i}: obj={val} 期望={exp} rounds={r['rounds']}{gap}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
