"""对比 test 40 题在「无 companion」与「有 companion」下的通过率。

- 无 companion：从 artifacts/eval_orbench_nl_v20.json 取 test split 的 40 题
- 有 companion：artifacts/eval_orbench_test40_companion.json（新跑）

用法：python scripts/_compare_companion.py [新结果路径]
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "or-bench-problems" / "data" / "problems.jsonl"
raw = {r["problem_id"]: r for r in
       (json.loads(l) for l in RAW.open(encoding="utf-8") if l.strip())}
SPLIT = {pid: r["split"] for pid, r in raw.items()}
# §2.6 认定的「给定输入下不可解」清单（散文缺必需数值）
UNANSWERABLE = ({f"INV-{i:03d}" for i in range(21, 32)}
                | {f"ASN-{i:03d}" for i in range(21, 32)}
                | {"SCH-029"})


def load(p):
    return {r["id"]: r for r in json.loads((ROOT / p).read_text(encoding="utf-8"))}


def summarize(recs, label, only_test=False):
    items = [(pid, r) for pid, r in recs.items()
             if (not only_test) or SPLIT.get(pid) == "test"]
    ok = [pid for pid, r in items if r.get("success")]
    print(f"\n=== {label} ===")
    print(f"  总计 {len(ok)}/{len(items)} = {len(ok)/len(items)*100:.1f}%")
    by_fam: dict[str, list[int]] = {}
    for pid, r in items:
        fam = pid.split("-")[0]
        a = by_fam.setdefault(fam, [0, 0])
        a[0] += 1
        a[1] += bool(r.get("success"))
    for fam, (n, k) in sorted(by_fam.items()):
        print(f"    {fam}: {k}/{n} = {k/n*100:.0f}%")
    # 可答题口径
    ans = [(pid, r) for pid, r in items if pid not in UNANSWERABLE]
    ans_ok = sum(1 for _, r in ans if r.get("success"))
    print(f"  剔除 §2.6 认定的 {len(UNANSWERABLE & {p for p,_ in items})} 道不可解题后："
          f"{ans_ok}/{len(ans)} = {ans_ok/len(ans)*100:.1f}%")
    return {pid: r.get("success") for pid, r in items}


def main():
    new_path = sys.argv[1] if len(sys.argv) > 1 else "artifacts/eval_orbench_test40_companion.json"
    old = load("artifacts/eval_orbench_nl_v20.json")
    new = load(new_path)
    a = summarize(old, "test 40 题：无 companion（v20 实测）", only_test=True)
    b = summarize(new, "test 40 题：有 companion（新跑）", only_test=True)

    print("\n=== 逐题变化（test 40）===")
    fixed = broke = 0
    for pid in sorted(set(a) | set(b)):
        x, y = a.get(pid), b.get(pid)
        if x is None or y is None:
            continue
        if not x and y:
            fixed += 1
            print(f"  ↑ {pid:10s} 失败 → 通过   {'(不可解)' if pid in UNANSWERABLE else ''}")
        elif x and not y:
            broke += 1
            print(f"  ↓ {pid:10s} 通过 → 失败")
    print(f"\n  修复 {fixed} 题，回退 {broke} 题，净 {fixed-broke:+d}")


if __name__ == "__main__":
    main()
