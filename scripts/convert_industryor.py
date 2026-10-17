"""转换 CardinalOperations/IndustryOR → 本项目 JSONL（剔除非线性题）。

输入：
- data/IndustryOR.json               原始数据（100 题，list）
- data/industryor_classification.json 线性性分类结果（由 classify_industryor.py 产出）

输出：data/industryor.jsonl

字段映射：
- id            → industryor-<序号 3 位补零>
- nl_text       → en_question
- problem_type  → auto（原始数据未标注 LP/IP/MIP，由 LLM 理解时判定）
- expected.objective_value → float(en_answer)
- trusted_reference → False（工业题参考答案未独立核验，容忍机制生效）

**参考答案人工核定**（原始值保留在 `expected.reference_original`，可追溯）：
- `corrected`：原 en_answer 与题面/最优性不符，改用按题面推导的值；
- `ambiguous`：参考值来源不明（多种自然口径都不符），标记但不改值、不计入模型能力。

剔除非线性：默认使用分类结果里的 `nonlinear` 清单；
未判定的题（unknown）默认**剔除**（保守，避免混入非线性题），可用 --keep-unknown 保留。

用法：
    python scripts/convert_industryor.py                 # 全部线性题
    python scripts/convert_industryor.py --limit 10      # 只取前 10 道线性题
    python scripts/convert_industryor.py --out data/industryor10.jsonl --limit 10
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "IndustryOR.json"
DEFAULT_CLS = ROOT / "data" / "industryor_classification.json"
DEFAULT_OUTPUT = ROOT / "data" / "industryor.jsonl"

# 参考答案人工核定（逐题人工核对后登记；原始值会保留在 reference_original）
REFERENCE_FIXES: dict[str, dict] = {
    "005": {
        "status": "corrected",
        "value": 1329.0 / 70.0,
        "note": (
            "原答案 18.6943 **不是最优**：它对应 okra=498.86g + carrot=101.14g，花费 $14.184，"
            "**预算还剩 $0.816 未花**——这笔钱换成 okra 可继续提高纤维。最优解为 "
            "okra=557.142857g + carrot=42.857143g，正好花满 $15，纤维 = 1329/70 ≈ 18.9857142857。"
            "（题面：4 种蔬菜的纤维/价格、总量 600g、预算 $15、选 1 种蛋白质、至少 2 种蔬菜；"
            "肉类纤维为 0，故最优只买 okra + carrot 两种。）"
        ),
    },
    "040": {
        "status": "ambiguous",
        "note": (
            "参考值来源不明，判为「参考口径有误 / 题面 GDP 口径歧义」，不计入模型能力："
            "含投入产出平衡（题面明写各产品生产消耗，属严格建模）得 36,288,567；"
            "去掉平衡后 y 自由则**无界**、加 y≤x 得 83,000,000（均远离 43,090,833）；"
            "4 种 GDP 口径 × 3 种人年→人月换算共 12 组合也都复现不了参考值。"
        ),
    },
    "002": {
        "status": "ambiguous",
        "note": (
            "**目标口径歧义**，题面未消除时序读法：「每架训练机每年训 5 名飞行员」+「训练持续 "
            "2 年」可读作 (a) 每架每年 5 人、2 年 10 人（a1 全训 100 + a2 训 1 年 75 = 175）；"
            "(b) 每架总共 5 人（25×5 = 125，即参考值）。参考 125 只在最宽松读法 (b) 下自洽；"
            "我们的模型按「能配到作战机的飞行员数」得 22。差异源于目标定义，非建模错误。"
        ),
    },
    "011": {
        "status": "ambiguous",
        "note": (
            "**参考值可疑**：反解 3867 对应 x2=3.375，但该点不落在可行域任何顶点上"
            "（制造 120/120 满、装配 53.625/80、检验 38.25/40 均不满），而 LP 最优必然在顶点。"
            "且 p3（闲置按 12:8:10 加权最小化）的最优点与利润最优点**重合**（均为 x1=6, x2=3.667），"
            "故正确实现 p3 不改变最优解（仍 4078.33）；「利润−加权闲置」得 3883.67（距参考 0.43%），"
            "但那是**重复扣减**闲置（按产能计 2480 已含闲置成本）。判为参考口径不明。"
        ),
    },
    "065": {
        "status": "inconsistent",
        "note": (
            "**题面自相矛盾**：同时要求「A、B 均满负荷」与「x≥80（微波炉）、y≥50（热水器）」时"
            "模型 **infeasible**——B 满负荷 x+3y=150 且 y≥50 推出 x≤0，与 x≥80 冲突；"
            "A 满负荷 2x+y=250 在 B 满负荷下最多只能到约 183.33h。"
            "参考 30500 精确对应「A 满负荷 + B 不设约束」的解 x=100, y=50，"
            "但该解需 B 用 100+3×50=250h，**超出题面的 150h**。需人工澄清题面"
            "（如 B 的可用工时应为 250h）。"
        ),
    },
}


def convert(
    input_path: Path,
    cls_path: Path,
    output_path: Path,
    limit: int | None = None,
    keep_unknown: bool = False,
) -> dict:
    rows = json.loads(input_path.read_text(encoding="utf-8"))

    nonlinear: set[int] = set()
    unknown: set[int] = set()
    if cls_path.exists():
        cls = json.loads(cls_path.read_text(encoding="utf-8"))
        nonlinear = {int(i) for i in cls.get("nonlinear", [])}
        unknown = {int(i) for i in cls.get("unknown", [])}
    else:
        print(f"[警告] 未找到分类结果 {cls_path.name}，将保留全部题目（可能含非线性）")

    dropped_nonlinear: list[int] = []
    dropped_unknown: list[int] = []
    dropped_bad: list[int] = []
    out: list[dict] = []

    for r in rows:
        rid = int(r["id"])
        if rid in nonlinear:
            dropped_nonlinear.append(rid)
            continue
        if rid in unknown and not keep_unknown:
            dropped_unknown.append(rid)
            continue
        q = (r.get("en_question") or "").strip()
        if not q:
            dropped_bad.append(rid)
            continue
        try:
            obj = float(r["en_answer"])
        except (TypeError, ValueError):
            dropped_bad.append(rid)
            continue
        expected_obj: dict = {"objective_value": obj}
        # 少数题的参考答案**不是目标函数值**，而是模型里某个变量的值（目标规划/字典序）。
        # 029：题面问「至少多少不能满足 P3」→ 答案是 P3 目标的缺口 d3_minus（=2500）。
        # 若按目标函数值比较，是 10000*d2_minus + p3_unmet = 10,003,000，与参考不可比。
        _av = {"029": "d3_minus"}.get(f"{rid:03d}")
        if _av:
            expected_obj["answer_variable"] = _av
        # 人工核定（原始值保留在 reference_original，可追溯）：
        #   corrected —— 原 en_answer 与题面/最优性不符，改用按题面推导的值；
        #   ambiguous —— 参考值来源不明（多种自然口径都不符），标记但不改值、不计入能力。
        _fix = REFERENCE_FIXES.get(f"{rid:03d}")
        if _fix:
            expected_obj["reference_original"] = obj
            expected_obj["reference_status"] = _fix["status"]
            expected_obj["reference_note"] = _fix["note"]
            if _fix["status"] == "corrected":
                expected_obj["objective_value"] = _fix["value"]
        out.append(
            {
                "id": f"industryor-{rid:03d}",
                "nl_text": q,
                "problem_type": "auto",
                "expected": expected_obj,
                "trusted_reference": False,
            }
        )

    out.sort(key=lambda x: x["id"])
    if limit:
        out = out[:limit]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for r in out:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    return {
        "total_input": len(rows),
        "kept": len(out),
        "dropped_nonlinear": sorted(dropped_nonlinear),
        "dropped_unknown": sorted(dropped_unknown),
        "dropped_bad": sorted(dropped_bad),
        "output": str(output_path),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="IndustryOR → 本项目 JSONL（剔除非线性）")
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--cls", type=Path, default=DEFAULT_CLS, help="分类结果 JSON")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--limit", type=int, default=None, help="只输出前 N 道")
    ap.add_argument("--keep-unknown", action="store_true", help="保留未判定的题（默认剔除）")
    args = ap.parse_args(argv)

    stats = convert(args.input, args.cls, args.out, args.limit, args.keep_unknown)

    print(f"输入题目: {stats['total_input']}")
    print(f"剔除非线性: {len(stats['dropped_nonlinear'])} 题 -> {stats['dropped_nonlinear']}")
    print(f"剔除未判定: {len(stats['dropped_unknown'])} 题 -> {stats['dropped_unknown']}")
    if stats["dropped_bad"]:
        print(f"剔除异常: {len(stats['dropped_bad'])} 题 -> {stats['dropped_bad']}")
    print(f"输出: {stats['kept']} 题 -> {Path(stats['output']).relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
