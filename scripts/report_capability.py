"""汇总当前建模能力：跨数据集通过率 + 题族拆分 + 失败模式分类 + 轮次分布。"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ART = Path(__file__).resolve().parents[1] / "artifacts"


def load(name):
    p = ART / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def terms_of(r):
    ir = r.get("final_ir") or {}
    return (ir.get("objective") or {}).get("expr", {}).get("terms", [])


def classify(r):
    """把一次失败归类到可复现的失败模式。"""
    if r.get("success"):
        return "通过"
    err = " ".join(r.get("errors") or [])
    if "目标函数退化" in err or "目标函数无任何项" in err:
        return "空目标(退化)"
    if "状态不符" in err:
        return "状态判断错"
    if "infeasible" in err or "不可行" in err:
        return "建出不可行模型"
    if "目标值与参考不符" in err:
        return "可行但目标值不符"
    if "截断" in err:
        return "输出截断"
    if "重复" in err or r.get("status") == "repeated_error":
        return "反复同错终止"
    if r.get("status") == "max_rounds_exceeded":
        return "轮次耗尽"
    if "SyntaxError" in err or "NameError" in err or "TypeError" in err:
        return "代码层错误"
    return "其它"


def summarize(recs, label, note=""):
    n = len(recs)
    ok = [r for r in recs if r.get("success")]
    first = [r for r in ok if r.get("rounds", 1) == 1]
    repaired = [r for r in ok if r.get("rounds", 1) > 1]
    print(f"\n=== {label}（{n} 题）{note} ===")
    print(f"  通过 {len(ok)}/{n} = {len(ok)/n*100:.1f}%"
          f"   （首轮 {len(first)}，经重写修复 {len(repaired)}）")
    print(f"  平均轮次 {sum(r.get('rounds',0) for r in recs)/n:.2f}")
    fails = [r for r in recs if not r.get("success")]
    if fails:
        cnt = Counter(classify(r) for r in fails)
        print(f"  失败模式（{len(fails)} 题）:")
        for k, v in cnt.most_common():
            print(f"    {k:16s} {v}")
    return ok, fails


def by_family(recs, label):
    print(f"\n--- {label} 按题族 ---")
    for fam, name in (("INV", "多期库存"), ("ASN", "指派/匹配"), ("SCH", "排班/调度")):
        sub = [r for r in recs if r["id"].startswith(fam)]
        if not sub:
            continue
        ok = sum(1 for r in sub if r.get("success"))
        fails = [r for r in sub if not r.get("success")]
        modes = Counter(classify(r) for r in fails)
        detail = " ".join(f"{k}×{v}" for k, v in modes.most_common())
        print(f"  {fam} {name}: {ok}/{len(sub)} = {ok/len(sub)*100:.0f}%   {detail}")


def main():
    print("#" * 78)
    print("# 建模能力汇总（基于现有 artifact；注意各批次协议版本不同）")
    print("#" * 78)

    nl = load("eval_nl4opt_full_v20.json")
    if nl:
        summarize(nl, "NL4OPT 245", "（v20；参考不可信 trusted=False）")

    ob = load("eval_orbench_nl_v20.json")
    if ob:
        # ⚠️ 必须按 split 看：development 题面内嵌 schema、test 题面为散文且
        # 配套 prompt_json 未被喂入 → 直接汇总 100 题会把「输入缺失」误算成模型能力。
        # 详见 METHODOLOGY.md §2.6 / §2.7。
        summarize(ob, "OR-Bench NL 100（**勿直接汇总，看下方 split 拆分**）", "（v20）")
        _by_split(ob)

    comp = load("eval_orbench_test40_companion.json")
    if comp:
        summarize(comp, "OR-Bench test 40（**补入 JSON companion**）",
                  "（题面完整 → 这是可公平计分的口径）")

    fix = load("eval_empty_obj_fix_v21.json")
    if fix:
        summarize(fix, "空目标定向重跑 22 题", "（v21 修复后，无 companion）")

    sch6 = load("eval_orbench_nl_sched_v6_layered.json")
    if sch6:
        summarize(sch6, "OR-Bench SCH 33", "（v6 历史基线，同题集）")

    # 正确的能力口径：按 split 合并（development 96.7% + test 补 companion 100%）
    if ob and comp:
        dev = [r for r in ob if _split_of(r["id"]) == "development"]
        d_ok = sum(1 for r in dev if r.get("success"))
        c_ok = sum(1 for r in comp if r.get("success"))
        t_old = [r for r in ob if _split_of(r["id"]) == "test"]
        t_old_ok = sum(1 for r in t_old if r.get("success"))
        print(f"\n=== OR-Bench 真实能力口径（题面均完整）===")
        print(f"  development（内嵌 schema）: {d_ok}/{len(dev)}")
        print(f"  test（+ JSON companion）  : {c_ok}/{len(comp)}")
        print(f"  合计: **{d_ok + c_ok}/{len(dev) + len(comp)}**")
        print(f"  （对照：test 无 companion 时仅 {t_old_ok}/{len(t_old)} "
              f"—— 低分源于输入缺失，非能力）")

    # 轮次分布：一次性建模能力
    for recs, label in ((nl, "NL4OPT 245"), (ob, "OR-Bench 100"), (comp, "OR-Bench test40(+companion)")):
        if not recs:
            continue
        dist = Counter(r.get("rounds") for r in recs)
        print(f"\n  {label} 轮次分布: {dict(sorted(dist.items()))}")


def _split_of(pid: str) -> str:
    """按 id 判定官方 split。

    官方划分：development 为各族编号 001–020（共 60 题，题面内嵌 SCHEMA_BEGIN）；
    test 为编号 021 起（共 40 题，散文模板，配套 prompt_json 未喂入）。
    """
    return "development" if int(pid.split("-")[1]) <= 20 else "test"


def _by_split(recs):
    """按官方 split 拆分（development 题面内嵌 schema / test 为散文模板）。"""
    groups: dict[str, list[int]] = {}
    for r in recs:
        a = groups.setdefault(_split_of(r["id"]), [0, 0])
        a[0] += 1
        a[1] += bool(r.get("success"))
    print("\n  --- 按 split（关键！）---")
    for k, (n, ok) in sorted(groups.items()):
        note = "（题面内嵌 schema）" if k == "development" else "（散文，配套 prompt_json 未喂入）"
        print(f"    {k:12s}: {ok}/{n} = {ok/n*100:.1f}%  {note}")


if __name__ == "__main__":
    main()
