"""转换 CardinalOperations/NL4OPT 原始数据 → 本项目 JSONL。

输入：NL4OPT/NL4OPT_with_optimal_solution.json（JSONL，每行 {en_question, en_answer}）
输出：data/nl4opt.jsonl（本项目的标准样例格式）

字段映射：
- id            → nl4opt-<序号 3 位补零>
- nl_text       → en_question
- problem_type  → auto（NL4OPT 未标注变量离散/连续，由 LLM 理解时判定）
- expected.objective_value → float(en_answer)，当 answer 为数值时
- expected.status → "no_solution"，当 answer 为 "No Best Solution" 或 -99999 时
  （原数据集用这两类占位符表示“无最优解”，不区分不可行/无界）

注意：NL4OPT 只提供最优目标值，无黄金 IR、无变量解，
因此该数据集走「LLM 理解链路」（配 DEEPSEEK_API_KEY 后评测），
确定性黄金 IR 链路对其不适用。

用法：python scripts/convert_nl4opt.py [输入JSON] [输出JSONL]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

DEFAULT_INPUT = Path(__file__).resolve().parents[1] / "NL4OPT" / "NL4OPT_with_optimal_solution.json"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "data" / "nl4opt.jsonl"

# “无最优解”占位符：字符串形式与数值形式
NO_SOLUTION_STRINGS = {"no best solution", "no solution", "none", "n/a", "nan", "inf", "-inf"}
NO_SOLUTION_NUMERIC = -99999.0


def _is_no_solution(answer_raw) -> bool:
    """判断 answer 是否为“无最优解”占位符。"""
    if isinstance(answer_raw, str):
        s = answer_raw.strip().lower()
        if s in NO_SOLUTION_STRINGS:
            return True
        try:
            return float(s) == NO_SOLUTION_NUMERIC
        except ValueError:
            return False
    if isinstance(answer_raw, (int, float)):
        return float(answer_raw) == NO_SOLUTION_NUMERIC
    return False


def convert(input_path: Path, output_path: Path) -> dict:
    """转换并返回统计信息。"""
    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")

    rows: list[dict] = []
    bad_lines: list[int] = []
    non_numeric: list[int] = []
    no_solution: list[int] = []

    with input_path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                bad_lines.append(lineno)
                continue

            question = obj.get("en_question", "").strip()
            answer_raw = obj.get("en_answer")
            if not question or answer_raw is None:
                bad_lines.append(lineno)
                continue

            expected: dict = {}
            if _is_no_solution(answer_raw):
                expected["status"] = "no_solution"
                no_solution.append(lineno)
            else:
                try:
                    expected["objective_value"] = float(answer_raw)
                except (TypeError, ValueError):
                    non_numeric.append(lineno)
                    continue

            rows.append(
                {
                    "id": f"nl4opt-{len(rows):03d}",
                    "nl_text": question,
                    "problem_type": "auto",
                    "expected": expected,
                    "trusted_reference": False,  # NL4OPT 参考答案由 GPT-4 转换求解，可能有标注噪声
                }
            )

    # 写输出
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    return {
        "input": str(input_path),
        "output": str(output_path),
        "total": len(rows),
        "bad_lines": bad_lines,
        "non_numeric": non_numeric,
        "no_solution": no_solution,
    }


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    input_path = Path(args[0]) if len(args) > 0 else DEFAULT_INPUT
    output_path = Path(args[1]) if len(args) > 1 else DEFAULT_OUTPUT

    stats = convert(input_path, output_path)

    print(f"输入: {stats['input']}")
    print(f"输出: {stats['output']}")
    print(f"成功转换: {stats['total']} 条")
    print(f"  其中『无最优解』(no_solution): {len(stats['no_solution'])} 条")
    if stats["bad_lines"]:
        print(f"跳过坏行: {stats['bad_lines']}")
    if stats["non_numeric"]:
        print(f"非数值目标值行: {stats['non_numeric']}")

    # 目标值分布概况（仅含数值答案）
    vals = []
    with output_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            e = json.loads(line)["expected"]
            if "objective_value" in e:
                vals.append(e["objective_value"])
    if vals:
        print(f"数值目标值: {len(vals)} 条 | 范围: min={min(vals):.6g}, max={max(vals):.6g}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
