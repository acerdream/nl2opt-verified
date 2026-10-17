"""快速演示：跑通确定性链路（无需 LLM / 网络）。

用法：python scripts/run_demo.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# 允许从源码直接运行（无需安装包）
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nl2opt.config import load_config  # noqa: E402
from nl2opt.dataset import load_builtin  # noqa: E402
from nl2opt.pipeline import run_dataset  # noqa: E402


def main() -> int:
    cfg = load_config()
    ds = load_builtin()
    results = run_dataset(ds, llm=None, cfg=cfg)

    print(f"{'id':18s} {'成功':6s} {'轮次':4s} {'目标值':>10s}  期望")
    for r, s in zip(results, ds.samples):
        exp = s.expected.get("objective_value")
        exp_s = f"{exp:.4f}" if isinstance(exp, (int, float)) else str(exp)
        obj = f"{r.objective_value:.4f}" if r.objective_value is not None else "-"
        print(f"{r.sample_id:18s} {str(r.success):6s} {r.rounds:<4d} {obj:>10s}  {exp_s}")

    n = len(results)
    ok = sum(1 for r in results if r.success)
    print(f"\n通过 {ok}/{n}")

    # 打印一例完整轨迹
    first = results[0]
    print(f"\n--- 样例 {first.sample_id} 的生成代码（节选）---")
    print(first.generated_code[:800])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
