"""核验辅助：读取全量评测结果，把每个 SUSPICIOUS / FAIL 题导出成核验清单。

对每个待核验题输出：
- id、参考答案（expected）、LLM 求解值
- 完整题目原文
供人工独立建模核验（区分"参考答案错" vs "LLM 建模错"）。

用法：python scripts/review_items.py [评测结果JSON]
默认读 artifacts/eval_nl4opt_full.json，输出 artifacts/review_items.txt
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT = ROOT / "artifacts" / "eval_nl4opt_full.json"
DATA = ROOT / "data" / "nl4opt.jsonl"


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    result_path = Path(args[0]) if args else DEFAULT_RESULT
    if not result_path.exists():
        print(f"评测结果不存在: {result_path}（评测可能尚未完成）", file=sys.stderr)
        return 1

    result = json.loads(result_path.read_text(encoding="utf-8"))

    # 建 id -> 题目 映射
    qmap = {}
    for line in DATA.read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            qmap[d["id"]] = d

    suspicious_ids = set(result.get("suspicious", []))
    hard_fail_ids = set(result.get("hard_fail", []))

    out_lines: list[str] = []
    out_lines.append("# 待核验题清单")
    out_lines.append(f"# 评测模型: {result.get('model')} | reasoning_effort: {result.get('reasoning_effort')}")
    out_lines.append(f"# suspicious: {len(suspicious_ids)} 题 | hard_fail: {len(hard_fail_ids)} 题")
    out_lines.append("")

    for item in result.get("detail", []):
        sid = item["id"]
        if sid not in suspicious_ids and sid not in hard_fail_ids:
            continue
        tag = "SUSPICIOUS" if sid in suspicious_ids else "FAIL"
        q = qmap.get(sid, {})
        out_lines.append("=" * 80)
        out_lines.append(f"[{tag}] {sid}")
        out_lines.append(f"参考答案: {json.dumps(q.get('expected', {}), ensure_ascii=False)}")
        out_lines.append(f"LLM 求解: {item.get('objective_value')} | 轮次: {item.get('rounds')}")
        out_lines.append(f"题目: {q.get('nl_text', '')}")
        out_lines.append("")

    out = ROOT / "artifacts" / "review_items.txt"
    out.write_text("\n".join(out_lines), encoding="utf-8")
    print(f"已导出 {len(suspicious_ids) + len(hard_fail_ids)} 道待核验题 -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
