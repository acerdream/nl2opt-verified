"""NL4OPT 同代码 A/B 归因：差异题是否恰是「卡片选择不同」的题？

两臂唯一变量是卡片策略：
- 分层臂：只加载命中的卡片
- 全卡片臂：总是加载 7 张

因此可归因的差异应集中在**两臂卡片集合不同**的题上；若差异出现在卡片集合相同的题上，
那只能是采样噪声（同一提示词、temperature=0，但推理路径仍可能不同）。

用法：python scripts/nl4opt_ab_analyze.py <layered.json> <allcards.json>
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from nl2opt.pipeline.prompts import select_cards  # noqa: E402

ALL = ["ratio", "multiple", "weighted", "blending", "scheduling", "inventory", "assignment"]


def load(p: str) -> dict:
    return {r["id"]: r for r in json.loads(Path(p).read_text(encoding="utf-8"))}


def load_expected() -> dict:
    """结果文件不含 expected，从数据集补齐（供人工核对求解值）。"""
    out: dict = {}
    for name in ("nl4opt.jsonl", "orbench_nl.jsonl"):
        f = ROOT / "data" / name
        if not f.exists():
            continue
        for line in f.open(encoding="utf-8"):
            if line.strip():
                row = json.loads(line)
                out[row["id"]] = row.get("expected")
    return out


EXPECTED = load_expected()


def cards_of(r: dict) -> list[str] | None:
    for s in (r.get("trajectory") or []):
        m = s.get("understand_meta") if isinstance(s, dict) else None
        if m and m.get("cards") is not None:
            return list(m["cards"])
    return None


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print(__doc__)
        return 2
    A, B = load(argv[1]), load(argv[2])
    ids = sorted(set(A) & set(B))
    print(f"对齐 {len(ids)} 题\n")

    onlyA = [i for i in ids if A[i]["success"] and not B[i]["success"]]
    onlyB = [i for i in ids if B[i]["success"] and not A[i]["success"]]
    agree = len(ids) - len(onlyA) - len(onlyB)
    print(f"分层通过 {sum(1 for i in ids if A[i]['success'])}/{len(ids)}｜"
          f"全卡片通过 {sum(1 for i in ids if B[i]['success'])}/{len(ids)}｜判定一致 {agree}")
    print(f"仅分层通过 {len(onlyA)}｜仅全卡片通过 {len(onlyB)}\n")

    # 卡片集合是否真的不同
    diff_cards, same_cards = [], []
    for i in ids:
        ca, cb = cards_of(A[i]), cards_of(B[i])
        if ca is None or cb is None:
            continue
        (diff_cards if set(ca) != set(cb) else same_cards).append(i)
    print(f"两臂卡片集合**不同**的题：{len(diff_cards)}｜**相同**的题：{len(same_cards)}")
    if diff_cards:
        n = Counter(len(cards_of(A[i]) or []) for i in diff_cards)
        print(f"  分层臂在这些题上加载卡片数的分布：{dict(sorted(n.items()))}")
    print()

    for label, group in (("仅分层通过", onlyA), ("仅全卡片通过", onlyB)):
        if not group:
            continue
        print(f"【{label}】{len(group)} 题")
        for i in group:
            ca, cb = cards_of(A[i]) or [], cards_of(B[i]) or []
            kind = "卡片集合不同" if set(ca) != set(cb) else "**卡片集合相同（应为噪声）**"
            exp = (A[i].get("expected") or EXPECTED.get(i) or {}).get("objective_value")
            print(f"  {i}: 分层 obj={A[i]['objective_value']} rounds={A[i]['rounds']} "
                  f"| 全卡片 obj={B[i]['objective_value']} rounds={B[i]['rounds']} "
                  f"| 期望={exp} | {kind}")
            print(f"      分层卡片={ca}")
            print(f"      全卡片卡片={len(cb)} 张")
        print()

    # 只在卡片集合不同的题上统计差异（可归因部分）
    if diff_cards:
        a_ok = sum(1 for i in diff_cards if A[i]["success"])
        b_ok = sum(1 for i in diff_cards if B[i]["success"])
        print(f"仅在「卡片集合不同」的 {len(diff_cards)} 题上：分层 {a_ok} vs 全卡片 {b_ok}")
    if same_cards:
        a_ok = sum(1 for i in same_cards if A[i]["success"])
        b_ok = sum(1 for i in same_cards if B[i]["success"])
        print(f"仅在「卡片集合相同」的 {len(same_cards)} 题上：分层 {a_ok} vs 全卡片 {b_ok}"
              f"（差异只能归为噪声）")

    # 成本与缓存
    sys.path.insert(0, str(ROOT / "scripts"))
    import report_cost as rc  # noqa: E402
    from nl2opt.config import load_config  # noqa: E402
    cfg = load_config()
    _, blk = rc._resolve_pricing(cfg, "deepseek-flash")
    p = (rc._price(blk, "idle", "input_cache_hit", None),
         rc._price(blk, "idle", "input_cache_miss", None),
         rc._price(blk, "idle", "output", None))
    print()
    for tag, M in (("分层", A), ("全卡片", B)):
        hit = miss = out = calls = 0
        for r in M.values():
            h, m, o = rc._tokens(r.get("token_usage") or {})
            hit += h; miss += m; out += o
            calls += int((r.get("token_usage") or {}).get("calls", 0) or 0)
        print(f"  {tag:8s} 调用 {calls:4d}｜缓存命中率 {hit/max(1,hit+miss)*100:5.1f}%｜"
              f"输出 {out:>8,d}｜成本 {rc._cost(hit, miss, out, *p):.4f} 元")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
