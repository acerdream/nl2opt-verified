"""修正 data/nl4opt.jsonl 里经独立求解器证伪的参考答案（全量评测版）。

判定依据：scripts/verify_answers.py 的独立建模求解结果。

两类修正：
1. 数值答案错误（LLM 对、参考答案错）→ 改为正确最优值。
2. no_solution 误标（实际有可行解）→ 改为正确目标值。

共 27 处，两个来源：
- 全量核验 19 处（12 数值错 + 7 no_solution 误标）；
- 早期抽样 8 处（原 `fix_nl4opt_answers.py`，已并入本脚本，避免重新转换后丢失）。

幂等：重复运行不重复修改。

用法：python scripts/fix_nl4opt_full.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DST = ROOT / "data" / "nl4opt.jsonl"

# sid -> 正确目标值（来自 verify_answers.py 的独立求解，取连续LP解，整数题取MILP解）
FIXES = {
    # ---- 全量核验：数值答案错误（12 处）----
    "nl4opt-021": 1072.0,
    "nl4opt-022": 100.34482758620689,
    "nl4opt-051": 291.6666666666667,
    "nl4opt-068": 36.666666666666664,
    "nl4opt-073": 2333.3333333333335,
    "nl4opt-085": 37083.333333333336,
    "nl4opt-105": 342857.14285714284,
    "nl4opt-131": 2.4,
    "nl4opt-167": 22.75,
    "nl4opt-174": 5625.0,
    "nl4opt-206": 166.66666666666669,
    "nl4opt-202": 66500.0,
    # ---- 全量核验：no_solution 误标（实际有解，7 处）----
    "nl4opt-028": 3000.0,
    "nl4opt-032": 233.33333333333334,
    "nl4opt-036": 1215.0,
    "nl4opt-071": 76.0,
    "nl4opt-197": 3220.0,
    "nl4opt-210": 47.77777777777778,
    "nl4opt-214": 16.0,
    # ---- 早期抽样核验（8 处，原 fix_nl4opt_answers.py 并入）----
    # 判定口径同 verify_answers.py 的独立建模求解
    "nl4opt-015": 12600.0,  # 已同步进原始文件，此处为幂等兜底
    "nl4opt-178": 17.0,
    "nl4opt-057": 970.0,
    "nl4opt-201": 670.0,
    "nl4opt-232": 1760.0,
    "nl4opt-031": 375.0,
    "nl4opt-079": 21.333333333333336,
    "nl4opt-181": 366.66666666666663,
}


def fix() -> dict:
    rows = [json.loads(l) for l in DST.read_text(encoding="utf-8").splitlines() if l.strip()]
    changed: dict[str, tuple] = {}
    for r in rows:
        sid = r.get("id")
        if sid in FIXES:
            newval = FIXES[sid]
            exp = r["expected"]
            old = exp.get("objective_value")
            old_status = exp.get("status")
            # no_solution 误标 → 删除 status，设置 objective_value
            if old_status == "no_solution":
                del exp["status"]
                exp["objective_value"] = newval
                changed[sid] = (f"no_solution", f"{newval}")
            elif old is not None and abs(old - newval) > 1e-6:
                exp["objective_value"] = newval
                changed[sid] = (old, newval)
    DST.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    return changed


def main() -> int:
    changed = fix()
    print(f"修正 data/nl4opt.jsonl 的参考答案：")
    for sid, (old, new) in changed.items():
        print(f"  {sid}: {old} -> {new}")
    if not changed:
        print("  （无需修改，已是最新）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
