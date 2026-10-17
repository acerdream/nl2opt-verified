"""离线重验：用当前验证器逻辑重跑历史评测结果，量化判定差异。

轨迹里保存了每轮的 `ir`、`status`、`objective_value`、`variables`，
因此**无需重新调用 LLM**即可重放验证，用于：
1. 验证器逻辑/容差修复后，评估对历史结果的影响（如目标值容差解耦）；
2. 复核某批评测是否存在误判。

用法：
    python scripts/reverify.py artifacts/eval_orbench_nl_sched.json
    python scripts/reverify.py artifacts/eval_orbench_nl_sched.json --last-round-only
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import nl2opt.verifier as verifier  # noqa: E402
from nl2opt.config import load_config  # noqa: E402
from nl2opt.ir import IR  # noqa: E402
from nl2opt.solver import SolverResult  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="离线重验历史评测结果（零 API 调用）")
    ap.add_argument("results", help="评测结果 JSON")
    ap.add_argument("--data", default=str(ROOT / "data" / "orbench_nl.jsonl"), help="数据集（取黄金值）")
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "REVERIFY_REPORT.md"))
    args = ap.parse_args(argv)

    cfg = load_config()
    feas_tol = float(cfg.pipeline.get("feasibility_tolerance", 1e-6))
    obj_tol = float(cfg.pipeline.get("objective_tolerance", 1e-4))

    gold: dict[str, dict] = {}
    texts: dict[str, str] = {}
    dpath = Path(args.data)
    if dpath.exists():
        for line in dpath.open(encoding="utf-8"):
            if line.strip():
                row = json.loads(line)
                gold[row["id"]] = row.get("expected") or {}
                texts[row["id"]] = row.get("nl_text") or ""

    results = json.loads(Path(args.results).read_text(encoding="utf-8"))
    rounds_total = 0
    changed: list[tuple[str, bool, bool, str]] = []
    n = 0
    for r in results:
        if not isinstance(r, dict):
            continue
        steps = [s for s in (r.get("trajectory") or []) if "round" in s and s.get("ir")]
        if not steps:
            continue
        n += 1
        rounds_total += len(steps)
        last = steps[-1]
        ir = IR.model_validate(last["ir"])
        sol = SolverResult(
            status=last.get("status") or "unknown",
            objective_value=last.get("objective_value"),
            variables=last.get("variables") or {},
        )
        expected = gold.get(r["id"]) or (r.get("expected") or {})
        checks = verifier.verify(ir, sol, expected, feas_tol, obj_tol, texts.get(r["id"]))
        new_ok = verifier.all_pass(checks)
        old_ok = bool(r.get("success"))
        if new_ok != old_ok:
            failed = [c.message for c in checks if not c.ok]
            changed.append((r["id"], old_ok, new_ok, "；".join(failed) or "—"))

    lines = ["# 离线重验报告\n"]
    lines.append(f"- 数据来源：`{Path(args.results).name}`｜可重放样本：{n}｜可重放轮次：{rounds_total}")
    lines.append(f"- 重验口径：`feasibility_tolerance={feas_tol}`、`objective_tolerance={obj_tol}`"
                 f"（目标值取相对容差 `tol × max(1,|期望|)`）")
    lines.append(f"- 重验方式：取每样本**最后一轮**的 `ir`/`status`/`variables` 重放验证，**零 API 调用**\n")
    lines.append(f"| 指标 | 值 |")
    lines.append("|---|---|")
    lines.append(f"| 原判定通过 | {sum(1 for r in results if isinstance(r, dict) and r.get('success'))} |")
    lines.append(f"| 重验后判定差异 | {len(changed)} |")
    lines.append("")
    if changed:
        lines.append("## 判定变化\n")
        lines.append("| 题号 | 原判定 | 重验判定 | 当前未通过的检查 |")
        lines.append("|---|---|---|---|")
        for pid, old, new, why in changed:
            lines.append(f"| {pid} | {'通过' if old else '失败'} | {'通过' if new else '失败'} | {why} |")
        lines.append("")
        lines.append("> 说明：原判定来自当时的运行结果；重验差异通常源于**验证器逻辑/容差修复**。"
                     "若不希望历史结果被误判，应据重验结论修正，而不是重跑（重跑还会引入 LLM 采样波动）。\n")
    else:
        lines.append("判定无变化——当前验证器与原判定一致。\n")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"报告已写入: {out}")
    print(f"可重放样本 {n} | 判定差异 {len(changed)}")
    for pid, old, new, why in changed:
        print(f"  {pid}: {'通过' if old else '失败'} → {'通过' if new else '失败'}  ({why})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
