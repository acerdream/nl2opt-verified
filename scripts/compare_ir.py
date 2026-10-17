"""语义覆盖对比 CLI：检测 LLM 产出 IR 相对黄金 IR 的「漏变量 / 漏约束」。

变量名无关（系数指纹对比），LLM 把 x1 写成 item_1 不会被误判。

用法：
    python scripts/compare_ir.py <gold_ir_json> <llm_ir_json>
    python scripts/compare_ir.py --sample knapsack-01 <gold_ir_json> <llm_ir_json>

核心逻辑在 nl2opt/semantic.py，已接入 verifier.check_semantic_coverage()。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nl2opt.ir import IR  # noqa: E402
from nl2opt.semantic import semantic_diff  # noqa: E402


def _load_ir(path: str) -> IR:
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    # 兼容：直接 IR，或包裹在 result 里的 final_ir
    if "final_ir" in data and isinstance(data["final_ir"], dict):
        data = data["final_ir"]
    return IR.model_validate(data)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="对比 LLM IR 与黄金 IR 的语义覆盖")
    p.add_argument("gold_json", nargs="?", help="黄金 IR JSON 文件路径")
    p.add_argument("llm_json", nargs="?", help="LLM IR JSON 文件路径")
    p.add_argument("--sample", help="（可选）内置黄金 IR 的样例 id，替代 gold_json")
    args = p.parse_args(argv)

    if args.sample:
        from nl2opt.gold_ir import get_gold_ir
        gold = get_gold_ir(args.sample)
        if gold is None:
            print(f"无该样例的黄金 IR: {args.sample}", file=sys.stderr)
            return 1
        if not args.llm_json:
            print("需要提供 llm_json 路径", file=sys.stderr)
            return 2
        llm = _load_ir(args.llm_json)
    else:
        if not args.gold_json or not args.llm_json:
            p.print_help()
            return 2
        gold = _load_ir(args.gold_json)
        llm = _load_ir(args.llm_json)

    result = semantic_diff(gold, llm)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
