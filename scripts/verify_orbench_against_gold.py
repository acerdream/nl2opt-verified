"""核实 test 40 题 100% 通过不是判定放宽：逐题比对「求解值 vs 官方 gold_objective/status」。

判定依据：data/or-bench-problems 原始记录的 gold_objective（官方 OR-Tools CP-SAT 求得）。
若求解值与 gold 一致、且 infeasible 题被识别为 infeasible，则通过是真实的。
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
raw = {r["problem_id"]: r for r in
       (json.loads(l) for l in (ROOT / "data" / "or-bench-problems" / "data" / "problems.jsonl")
        .open(encoding="utf-8") if l.strip())}
res = {r["id"]: r for r in
       json.loads((ROOT / "artifacts" / "eval_orbench_test40_companion.json").read_text(encoding="utf-8"))}

print(f"{'id':10s} {'gold_obj':>9s} {'gold_status':>11s} | {'求解obj':>9s} {'求解status':>11s} {'判定':>5s} {'匹配':>5s}")
print("-" * 74)
mismatch = 0
for pid in sorted(res):
    g, r = raw[pid], res[pid]
    go, gs = g.get("gold_objective"), (g.get("gold_status") or "")
    ro, rs = r.get("objective_value"), (r.get("status") or "")
    if gs == "infeasible":
        match = (rs == "infeasible")
    else:
        match = ro is not None and go is not None and abs(float(ro) - float(go)) < 1e-6
    mismatch += (not match)
    print(f"{pid:10s} {str(go):>9s} {gs:>11s} | {str(ro):>9s} {rs:>11s} "
          f"{str(r.get('success')):>5s} {'✓' if match else '✗':>5s}")
print(f"\n与官方 gold 不一致的题: {mismatch}/{len(res)}")

# 8 道不可行题
inf = [pid for pid, g in raw.items() if g.get("gold_status") == "infeasible"]
inf_ok = sum(1 for pid in inf if res.get(pid, {}).get("status") == "infeasible")
print(f"不可行题识别: {inf_ok}/{len(inf)}")
