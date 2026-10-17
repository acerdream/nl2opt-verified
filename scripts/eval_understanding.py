"""理解正确率评测：LLM 输出 IR 与黄金 IR 的语义一致性。

对内置样例（有黄金 IR），用 LLM 理解（--force-llm-understand），
对比 LLM IR 与黄金 IR 的语义覆盖（变量名无关的系数指纹，见 semantic.py）。

指标：
- 语义完全一致率：LLM IR 与黄金 IR 零语义缺口（无漏变量/漏约束/目标不一致）
- 各缺口类型分布

用法：python scripts/eval_understanding.py [N] [seed]
需要：DEEPSEEK_API_KEY。
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nl2opt.config import load_config  # noqa: E402
from nl2opt.dataset import load_builtin  # noqa: E402
from nl2opt.gold_ir import get_gold_ir  # noqa: E402
from nl2opt.llm import make_provider  # noqa: E402
from nl2opt.pipeline import run_sample  # noqa: E402
from nl2opt.semantic import semantic_diff  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    n = int(args[0]) if len(args) > 0 else 20
    seed = int(args[1]) if len(args) > 1 else 42

    cfg = load_config()
    llm = make_provider(cfg)
    ds = load_builtin()

    # 只抽有黄金 IR 且有数值目标值的题
    pool = [s for s in ds if get_gold_ir(s.id) is not None and "objective_value" in s.expected]
    rng = random.Random(seed)
    subset = rng.sample(pool, min(n, len(pool)))

    print(f"模型: {llm.model} | 理解正确率评测: {len(subset)} 题 (seed={seed})")
    print(f"开始 @ {time.strftime('%H:%M:%S')}\n")

    t0 = time.time()
    results = run_dataset(subset, llm=llm, cfg=cfg, force_llm_understand=True)
    elapsed = time.time() - t0

    exact = 0  # 语义完全一致
    gap_missing_var = 0
    gap_missing_con = 0
    gap_obj = 0
    gap_domain = 0
    hard_fail = 0  # 理解失败/求解失败

    print(f"{'id':20s} {'语义':8s} {'缺口'}")
    print("-" * 60)
    for r, s in zip(results, subset):
        gold = get_gold_ir(s.id)
        if r.final_ir is None:
            hard_fail += 1
            print(f"{s.id:20s} FAIL     理解失败")
            continue
        try:
            from nl2opt.ir import IR
            from nl2opt.semantic import describe_gap, has_semantic_gap
            llm_ir = IR.model_validate(r.final_ir)
            diff = semantic_diff(gold, llm_ir)
            if not has_semantic_gap(diff):
                exact += 1
                print(f"{s.id:20s} OK       -")
            else:
                parts = []
                if diff["missing_variables"]:
                    gap_missing_var += 1
                    parts.append(f"漏变量{sum(i['count'] for i in diff['missing_variables'])}")
                if diff["missing_constraints"]:
                    gap_missing_con += 1
                    parts.append(f"漏约束{sum(i['count'] for i in diff['missing_constraints'])}")
                if diff["objective_mismatch"]:
                    gap_obj += 1
                    parts.append("目标不一致")
                if diff.get("domain_mismatch"):
                    gap_domain += 1
                    parts.append("定义域不一致")
                print(f"{s.id:20s} GAP      {'+'.join(parts)}  ({describe_gap(diff)})")
        except Exception as exc:
            hard_fail += 1
            print(f"{s.id:20s} FAIL     {type(exc).__name__}")

    total = len(subset)
    print("=" * 60)
    print(f"\n总题数: {total}")
    print(f"语义完全一致: {exact} ({exact/total*100:.1f}%)")
    print(f"  漏变量: {gap_missing_var}")
    print(f"  漏约束: {gap_missing_con}")
    print(f"  目标不一致: {gap_obj}")
    print(f"  定义域不一致: {gap_domain}")
    print(f"理解/求解失败: {hard_fail}")
    print(f"耗时: {elapsed:.0f}s ({elapsed/total:.1f}s/题)")

    # 落盘
    out = Path(__file__).resolve().parents[1] / "artifacts" / f"eval_understanding_n{total}_seed{seed}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({
            "model": llm.model,
            "n": total,
            "seed": seed,
            "exact": exact,
            "gap_missing_variable": gap_missing_var,
            "gap_missing_constraint": gap_missing_con,
            "gap_objective": gap_obj,
            "gap_domain": gap_domain,
            "hard_fail": hard_fail,
            "elapsed_seconds": elapsed,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n结果已写入: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
