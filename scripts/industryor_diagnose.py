"""IndustryOR 10 题冒烟诊断：链路是否跑通、失败模式落在哪、缺哪张卡片。

用法：python scripts/industryor_diagnose.py [结果文件]
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from nl2opt.ir import IR  # noqa: E402
from nl2opt.pipeline.prompts import select_cards  # noqa: E402


def _outcome(r: dict) -> str | None:
    """读取疑似参考噪声的结局。

    优先用结果里的显式字段；旧批次（该字段引入前）回退到 trajectory 里的
    `suspicious_outcome` —— 否则会把「修复后达到参考值」误报成「仍未达到」。
    """
    if r.get("status") == "suspicious_resolved" or r.get("suspicious_resolved"):
        return "converged"
    for s in reversed(r.get("trajectory") or []):
        if isinstance(s, dict) and s.get("suspicious_outcome"):
            return s["suspicious_outcome"]
    return None


def classify_failure(r: dict) -> str:
    """按可操作的缺口给失败分类。"""
    errs = " | ".join(str(e) for e in (r.get("errors") or []))
    # 疑似参考噪声的两种结局：都不是通过，但信息价值不同，必须分开显示
    if r.get("suspicious"):
        if _outcome(r) == "converged":
            return "疑似参考噪声·**修复后达到参考值**（参考可达，多半是建模错）"
        return "疑似参考噪声·修复后仍未达到（疑似噪声才站得住）"
    if not r.get("final_ir"):
        if "截断" in errs or "max_tokens" in errs:
            return "输出截断（预算）"
        if "理解失败" in errs:
            return "理解失败（LLM 未产出可用 IR）"
        return "未产出 IR（其它）"
    if "目标函数退化" in errs:
        return "目标退化（建模缺陷）"
    if "目标值不符" in errs:
        return "目标值不符（建模错误）"
    if "不可行" in errs and "期望" in errs:
        return "可行性与参考不符"
    if "超时" in errs or "timeout" in errs.lower():
        return "求解超时"
    return "其它"


def main(argv: list[str]) -> int:
    p = Path(argv[1]) if len(argv) > 1 else ROOT / "artifacts" / "eval_industryor10_v7_layered.json"
    if not p.exists():
        print(f"缺少结果文件：{p}", file=sys.stderr)
        return 2
    rows = json.loads(p.read_text(encoding="utf-8"))
    ds = {json.loads(l)["id"]: json.loads(l)
          for l in (ROOT / "data" / "industryor10.jsonl").open(encoding="utf-8") if l.strip()}

    ok = [r for r in rows if r.get("success")]
    resolved = [r for r in rows if r.get("suspicious") and _outcome(r) == "converged"]
    unresolved = [r for r in rows if r.get("suspicious") and _outcome(r) != "converged"]
    print(f"结果文件：{p.name}｜{len(rows)} 题｜通过 {len(ok)}/{len(rows)}"
          f"（{len(ok)/max(1,len(rows))*100:.0f}%）")
    if resolved or unresolved:
        print(f"  疑似参考噪声：**修复后达到参考值 {len(resolved)} 题**"
              f"（参考可达，多半是建模错）｜仍未达到 {len(unresolved)} 题")
    print(f"模型 {rows[0].get('model')}｜协议 v{rows[0].get('protocol_version')}"
          f"｜prompt_mode {rows[0].get('prompt_mode')}")
    print()

    print(f"{'题号':16s} {'结果':6s} {'轮次':>4s} {'求解值':>16s} {'期望':>16s} {'卡片':22s} 失败模式")
    print("-" * 116)
    modes = Counter()
    for r in sorted(rows, key=lambda x: x["id"]):
        mode = "" if r.get("success") else classify_failure(r)
        if mode:
            modes[mode] += 1
        cards = []
        for s in (r.get("trajectory") or []):
            m = s.get("understand_meta") if isinstance(s, dict) else None
            if m and m.get("cards") is not None:
                cards = list(m["cards"])
                break
        exp = (ds.get(r["id"], {}).get("expected") or {}).get("objective_value")
        print(f"{r['id']:16s} {'通过' if r.get('success') else '失败':6s} {r.get('rounds',0):>4d} "
              f"{str(r.get('objective_value'))[:16]:>16s} {str(exp)[:16]:>16s} "
              f"{','.join(cards) or '（无）':22s} {mode}")

    print()
    print("=== 失败模式汇总 ===")
    for k, v in modes.most_common():
        print(f"  {k}: {v} 题")
    if not modes:
        print("  （无失败）")

    print()
    print("=== 卡片加载情况 ===")
    cc = Counter()
    for r in rows:
        for s in (r.get("trajectory") or []):
            m = s.get("understand_meta") if isinstance(s, dict) else None
            if m and m.get("cards") is not None:
                cc[tuple(m["cards"])] += 1
                break
    for combo, n in cc.most_common():
        print(f"  {list(combo) or '（无卡片）'}: {n} 题")

    print()
    print("=== 失败题详情（供判断缺口）===")
    for r in sorted(rows, key=lambda x: x["id"]):
        if r.get("success"):
            continue
        print(f"\n【{r['id']}】{classify_failure(r)}")
        txt = " ".join(ds.get(r["id"], {}).get("nl_text", "").split())
        print(f"  题面: {txt[:200]}…")
        has_tex = bool(re.search(r"\$[^$]{1,60}\$|\\begin\{tabular\}", txt))
        print(f"  含 LaTeX: {'是' if has_tex else '否'}")
        for e in (r.get("errors") or []):
            print(f"  错误: {str(e)[:200]}")
        if r.get("final_ir"):
            try:
                ir = IR.model_validate(r["final_ir"])
                print(f"  IR: 变量 {len(ir.variables)}｜约束 {len(ir.constraints)}"
                      f"｜目标项 {len(ir.objective.expr.terms)}"
                      f"｜assumptions {len(ir.assumptions)}")
                if ir.assumptions:
                    print(f"    假设: {ir.assumptions[:3]}")
            except Exception as exc:
                print(f"  IR 解析失败: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
