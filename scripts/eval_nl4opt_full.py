"""全量评测：对 data/nl4opt.jsonl 的全部 245 题（含 no_solution）跑完整闭环。

输出：
- 每题结果（OK / SUSPICIOUS / FAIL / NO_SOLUTION 判定）
- 汇总统计
- 结果落盘 artifacts/eval_nl4opt_full.json

用法：python scripts/eval_nl4opt_full.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nl2opt.config import load_config  # noqa: E402
from nl2opt.dataset import load_jsonl  # noqa: E402
from nl2opt.llm import make_provider  # noqa: E402
from nl2opt.pipeline import run_dataset  # noqa: E402

DEFAULT_DATA = Path(__file__).resolve().parents[1] / "data" / "nl4opt.jsonl"


def main(argv: list[str] | None = None) -> int:
    cfg = load_config()
    llm = make_provider(cfg)
    ds = load_jsonl(DEFAULT_DATA)

    total = len(ds)
    print(f"模型: {llm.model} | 思考强度: {llm.reasoning_effort}")
    print(f"全量评测: {total} 题")
    print(f"开始 @ {time.strftime('%H:%M:%S')}\n")

    t0 = time.time()
    results = run_dataset(ds, llm=llm, cfg=cfg)
    elapsed = time.time() - t0

    ok = [r for r in results if r.success]
    suspicious = [r for r in results if r.suspicious]
    hard_fail = [r for r in results if not r.success and not r.suspicious]
    repaired = [r for r in ok if r.rounds > 1]

    print("\n" + "=" * 70)
    print(f"{'id':14s} {'结果':10s} {'轮次':4s} {'目标值':>14s}")
    print("-" * 70)
    for r in results:
        if r.success:
            tag = "OK"
        elif r.suspicious:
            tag = "SUSPICIOUS"
        else:
            tag = "FAIL"
        obj = f"{r.objective_value:.4f}" if r.objective_value is not None else "-"
        print(f"{r.sample_id:14s} {tag:10s} {r.rounds:<4d} {obj:>14s}")
    print("=" * 70)

    print(f"\n总题数: {total}")
    print(f"通过: {len(ok)} ({len(ok)/total*100:.1f}%)")
    print(f"  首轮通过: {len(ok) - len(repaired)}")
    print(f"  经重写修复: {len(repaired)}")
    print(f"疑似标注错误(suspicious): {len(suspicious)}")
    print(f"硬失败: {len(hard_fail)}")
    print(f"平均轮次: {sum(r.rounds for r in results)/total:.2f}")
    print(f"耗时: {elapsed:.0f}s ({elapsed/total:.1f}s/题)")

    if suspicious:
        print("\n--- 疑似标注错误（待核验）---")
        for r in suspicious:
            print(f"  {r.sample_id}: {r.errors}")

    if hard_fail:
        print("\n--- 硬失败（需检查）---")
        for r in hard_fail:
            print(f"  {r.sample_id} (轮次{r.rounds}): {r.errors}")

    # 落盘
    out = Path(__file__).resolve().parents[1] / "artifacts" / "eval_nl4opt_full.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "model": llm.model,
                "reasoning_effort": llm.reasoning_effort,
                "n": total,
                "passed": len(ok),
                "first_round_passed": len(ok) - len(repaired),
                "repaired": len(repaired),
                "suspicious": [r.sample_id for r in suspicious],
                "hard_fail": [r.sample_id for r in hard_fail],
                "avg_rounds": sum(r.rounds for r in results) / total,
                "elapsed_seconds": elapsed,
                "detail": [
                    {
                        "id": r.sample_id,
                        "result": "OK" if r.success else ("SUSPICIOUS" if r.suspicious else "FAIL"),
                        "rounds": r.rounds,
                        "objective_value": r.objective_value,
                        "errors": r.errors,
                    }
                    for r in results
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n结果已写入: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
