"""用 LLM 判定 IndustryOR 每道题能否用「线性模型」表达。

背景：IndustryOR 的 100 道题无类型标注，且题面里的
"square meters / cubic meters / cm^3" 只是**单位**，不代表非线性。
纯关键词法会严重误杀，因此用 LLM 逐题判定。

判定口径：
- linear   ：目标与约束均为决策变量的线性函数（含固定成本/big-M/指标变量等 MILP 技巧）
- nonlinear：需变量乘积、变量相除、幂次/开方、对数/指数、三角、变量作下标等

输出：data/industryor_classification.json
    {"linear": [id...], "nonlinear": [id...], "detail": {id: {"label":..., "reason":...}}}

用法：python scripts/classify_industryor.py [--batch 8]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nl2opt.config import load_config  # noqa: E402
from nl2opt.llm import LLMMessage, make_provider, extract_json  # noqa: E402

SRC = ROOT / "data" / "IndustryOR.json"
OUT = ROOT / "data" / "industryor_classification.json"

SYSTEM = """你是运筹学专家。用户会给出若干道优化问题的题面，请逐题判断它能否用**线性模型**精确表达。

判定口径：
- linear：目标函数与所有约束都是决策变量的**线性**函数。
  以下仍算 linear：固定成本/启动成本（配 0-1 变量 + big-M）、指标变量、
  分段线性、绝对值（可线性化）、多期/多品类求和、给定常数表与单位换算。
- nonlinear：需要决策变量的**乘积**、**相除**、**幂次/开方**、**对数/指数**、
  三角函数、以变量作下标，或目标/约束中出现变量间的高阶关系。

关键提醒（避免误判）：
- "square meters"、"cubic meters"、"cm^3"、"m²" 是**计量单位**，不代表非线性。
- "面积/体积"作为**给定参数**（如仓库面积上限）不影响线性性；
  但若要求"最大面积/体积"且长宽高是决策变量，则为 nonlinear。
- 可靠性/概率的**乘积**、方差、风险、期望效用等通常为 nonlinear。
- 仅凭题面无法确定时，倾向判定 nonlinear 并说明理由。

只输出 JSON，格式：{"题号": {"label": "linear"|"nonlinear", "reason": "不超过15字的理由"}}
不要输出其他文字。"""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="IndustryOR 线性性分类")
    ap.add_argument("--batch", type=int, default=8, help="每批题数（默认 8）")
    args = ap.parse_args(argv)

    cfg = load_config()
    llm = make_provider(cfg)
    rows = json.loads(SRC.read_text(encoding="utf-8"))

    detail: dict[str, dict] = {}
    batches = [rows[i : i + args.batch] for i in range(0, len(rows), args.batch)]
    print(f"模型: {llm.model} | 共 {len(rows)} 题，分 {len(batches)} 批（每批 {args.batch}）\n")

    for bi, batch in enumerate(batches, 1):
        parts = []
        for r in batch:
            parts.append(f"### 题号 {r['id']}\n{r['en_question']}")
        user = "请判定下列各题：\n\n" + "\n\n".join(parts)
        messages = [
            LLMMessage(role="system", content=SYSTEM),
            LLMMessage(role="user", content=user),
        ]
        try:
            resp = llm.complete(messages)
            data = extract_json(resp.text)
        except Exception as exc:
            print(f"  批 {bi} 失败: {type(exc).__name__}: {exc}")
            continue
        got = 0
        for r in batch:
            key = str(r["id"])
            item = data.get(key)
            if isinstance(item, dict):
                detail[key] = item
                got += 1
            elif isinstance(item, str):  # 容忍 "linear" 直接作答
                detail[key] = {"label": item, "reason": ""}
                got += 1
        print(f"  批 {bi}/{len(batches)}: 解析 {got}/{len(batch)} 题")

    linear = sorted(int(k) for k, v in detail.items() if str(v.get("label", "")).lower().startswith("linear"))
    nonlinear = sorted(int(k) for k, v in detail.items() if str(v.get("label", "")).lower().startswith("nonlinear"))
    unknown = sorted(r["id"] for r in rows if str(r["id"]) not in detail)

    OUT.write_text(
        json.dumps(
            {"linear": linear, "nonlinear": nonlinear, "unknown": unknown, "detail": detail},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nlinear   : {len(linear)}")
    print(f"nonlinear: {len(nonlinear)} -> {nonlinear}")
    print(f"未判定   : {len(unknown)} -> {unknown}")
    print(f"\n结果已写入: {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
