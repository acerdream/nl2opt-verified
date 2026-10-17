"""离线评估 v9 判定变更的影响：把历史结果按新语义重新裁决。

不需要重跑 API——结果文件里的 trajectory 已记录 `suspicious_outcome`，
据此按 v9 语义重新裁决即可。

**注意**：结果文件里的 `success` 字段是**当时**（v8 及更早）的判定，不代表新语义。
本脚本用 `verdict_v9()` **重新裁决**，而不是读取旧字段——否则会把旧判定当新判定，
得出「无变化」的错误结论（这是本脚本第一版犯过的错）。
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FILES = [
    ("NL4OPT 245", "eval_nl4opt_full_v8.json"),
    ("IndustryOR 10", "eval_industryor10_v8.json"),
    ("NL4OPT 50(v5)", "eval_nl4opt_sub50_v5.json"),
]


def outcome(r):
    if r.get("status") == "suspicious_resolved" or r.get("suspicious_resolved"):
        return "converged"
    for s in reversed(r.get("trajectory") or []):
        if isinstance(s, dict) and s.get("suspicious_outcome"):
            return s["suspicious_outcome"]
    return None


def verdict_v9(r) -> bool:
    """按 v9 语义重新裁决：converged 计通过，unresolved 不计。"""
    if r.get("success"):
        return True
    return bool(r.get("suspicious")) and outcome(r) == "converged"


def main() -> int:
    print(f"{'批次':18s} {'题数':>5s} {'旧判定(v8)':>11s} {'新判定(v9)':>11s} {'变化':>6s}  说明")
    print("-" * 96)
    for name, f in FILES:
        p = ROOT / "artifacts" / f
        if not p.exists():
            print(f"  {name}: 缺文件 {f}")
            continue
        rows = [r for r in json.loads(p.read_text(encoding="utf-8")) if isinstance(r, dict)]
        n = len(rows)
        old = sum(1 for r in rows if r["success"])
        conv = [r for r in rows if r.get("suspicious") and outcome(r) == "converged"]
        new = sum(1 for r in rows if verdict_v9(r))
        unres = [r for r in rows if r.get("suspicious") and outcome(r) != "converged"]
        first = sum(1 for r in rows if r["success"] and (r.get("rounds") or 0) == 1)
        print(f"{name:18s} {n:>5d} {old:>5d}/{n:<5d} {new:>5d}/{n:<5d} {new-old:>+6d}  "
              f"converged {len(conv)}｜unresolved {len(unres)}｜首轮通过 {first}")
        if unres:
            print(f"{'':18s} {'':>5s} unresolved 题号: {[r['id'] for r in unres]}")

    print()
    print("=== 安全不变量：按 v9 语义，unresolved 必须仍不计通过 ===")
    bad_total = 0
    for name, f in FILES:
        p = ROOT / "artifacts" / f
        if not p.exists():
            continue
        rows = [r for r in json.loads(p.read_text(encoding="utf-8")) if isinstance(r, dict)]
        bad = [r["id"] for r in rows
               if r.get("suspicious") and outcome(r) != "converged" and verdict_v9(r)]
        bad_total += len(bad)
        print(f"  {name}: unresolved 中被 v9 判通过的题 {len(bad)} {bad}")
    print(f"  合计 {bad_total}（应为 0）")
    return 0 if bad_total == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
