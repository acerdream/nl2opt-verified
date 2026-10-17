"""OR-Bench 数据完整性审计：题面形态、split 划分、companion 附加情况。

存在意义（实测驱动）：曾因 `convert_orbench.py` 只取 `prompt_en`、丢弃 `prompt_json`，
使 test split 40 题的题面残缺（INV 甚至引用了不存在的"JSON companion"），
而评测把由此产生的失败**误归因为模型建模能力弱**。本脚本用于随时复查数据侧是否完整。

审计项：
 ① split × 题面变体（schema_embedded / text_only）分布
 ② text_only 题：自然语言题面是否给出建模必需参数（成本/权重/需求）
 ③ 题面是否**显式引用** JSON companion
 ④ 当前 data/orbench_nl.jsonl 的 companion 附加情况（逐题核对附件=原始 prompt_json）

用法：
    python scripts/audit_orbench_data.py
    python scripts/audit_orbench_data.py --dataset data/orbench_nl_nocompanion.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "or-bench-problems" / "data" / "problems.jsonl"
MARK = "\n\n--- JSON companion ---\n"

# 各族「建模必需」的目标侧参数（散文若不给，模型无法建出正确目标）
KEY_KEYS = {
    "inventory": ["unit_cost", "holding_cost", "shortage_cost", "pack_size"],
    "assignment": ["cost", "penalty", "capacity"],
    "scheduling": ["weight"],
}
COMPANION_PAT = re.compile(r"json companion|companion", re.IGNORECASE)


def load_raw() -> dict[str, dict]:
    if not RAW.exists():
        print(f"找不到原始数据: {RAW}", file=sys.stderr)
        print("请先下载 or-bench-problems（见 convert_orbench.py 的提示）", file=sys.stderr)
        raise SystemExit(2)
    return {r["problem_id"]: r for r in
            (json.loads(l) for l in RAW.open(encoding="utf-8") if l.strip())}


def _variant(en: str) -> str:
    return "schema_embedded" if "SCHEMA_BEGIN" in en else "text_only"


def _key_values(payload, keys) -> list:
    out: list = []

    def walk(node):
        if isinstance(node, dict):
            for k, v in node.items():
                if k in keys and isinstance(v, (int, float)) and not isinstance(v, bool):
                    out.append(v)
                walk(v)
        elif isinstance(node, list):
            for x in node:
                walk(x)

    walk(payload)
    return out


def audit_raw(raw: dict[str, dict]) -> None:
    print("=" * 66)
    print("① split × 题面变体")
    print("=" * 66)
    c = Counter((r["split"], _variant(r.get("prompt_en") or "")) for r in raw.values())
    for k, v in sorted(c.items()):
        print(f"  {k[0]:12s} {k[1]:16s}: {v} 题")

    print()
    print("=" * 66)
    print("② text_only 题：自然语言题面是否给出建模必需参数")
    print("=" * 66)
    incomplete: list[str] = []
    for pid, r in sorted(raw.items()):
        en = r.get("prompt_en") or ""
        if _variant(en) != "text_only":
            continue
        fam = r["family"]
        if fam == "scheduling":
            # 语义邻近判据：'weighted tardiness' 只是形容词，需 weight 后跟数字才算给出
            given = bool(re.findall(r"weight\w*\s*[:=]?\s*\d+", en, re.IGNORECASE))
            missing = not given
            detail = f"weight给定={given}"
        else:
            vals = _key_values(r.get("schema_payload") or {}, KEY_KEYS[fam])
            en_nums = set(re.findall(r"\d+(?:\.\d+)?", en))
            miss = [v for v in vals if str(v) not in en_nums and str(int(v)) not in en_nums]
            missing = bool(miss)
            detail = f"缺 {len(miss)}/{len(vals)}"
        if missing:
            incomplete.append(pid)
        print(f"  {pid:10s} {fam:11s} {detail:16s} "
              f"{'← 题面不完整' if missing else ''}")
    print(f"\n  text_only 共 {sum(1 for r in raw.values() if _variant(r.get('prompt_en') or '') == 'text_only')} 题，"
          f"题面不完整 {len(incomplete)} 题")

    print()
    print("=" * 66)
    print("③ 题面是否显式引用 JSON companion")
    print("=" * 66)
    ref = [pid for pid, r in raw.items() if COMPANION_PAT.search(r.get("prompt_en") or "")]
    by_fam = Counter(pid.split("-")[0] for pid in ref)
    print(f"  显式引用 companion: {len(ref)} 题  {dict(by_fam)}")
    print("  注意：未显式引用 ≠ 不需要附件——同一 split 的设计一致，")
    print("        只是 ASN/SCH 模板省略了这句话（其 cost/penalty/weight 同样只在 prompt_json 里）。")


def audit_dataset(path: Path, raw: dict[str, dict]) -> None:
    print()
    print("=" * 66)
    print(f"④ 生成数据集审计：{path.name}")
    print("=" * 66)
    if not path.exists():
        print(f"  文件不存在: {path}")
        return
    rows = {r["id"]: r for r in
            (json.loads(l) for l in path.open(encoding="utf-8") if l.strip())}
    att = [pid for pid, r in rows.items() if MARK in r["nl_text"]]
    splits = Counter(rows[pid]["extra"].get("split") for pid in att)
    print(f"  题数 {len(rows)}；附加 companion {len(att)} 题  {dict(splits)}")

    bad = []
    for pid, r in rows.items():
        has = MARK in r["nl_text"]
        raw_en = raw[pid].get("prompt_en") or pid
        raw_pj = (raw[pid].get("prompt_json") or "").strip()
        if has:
            head, comp = r["nl_text"].split(MARK, 1)
            if head != raw_en:
                bad.append(f"{pid}: 题面被改动")
            if comp.strip() != raw_pj:
                bad.append(f"{pid}: 附件 ≠ 原始 prompt_json")
    print(f"  逐题核对（题面未改 + 附件=原始 prompt_json）: "
          f"{'全部通过 ✓' if not bad else f'{len(bad)} 处问题 ✗'}")
    for b in bad[:6]:
        print("   ", b)
    for pid, r in rows.items():
        v = r["extra"].get("prompt_variant")
        attached = r["extra"].get("companion_attached")
        if v == "text_only" and not attached:
            print(f"  ⚠ {pid} 是 text_only 但未附加 companion")
        if v == "schema_embedded" and attached:
            print(f"  ⚠ {pid} 已内嵌 schema 却又附加了 companion（重复）")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="OR-Bench 数据完整性审计")
    ap.add_argument("--dataset", default=str(ROOT / "data" / "orbench_nl.jsonl"))
    args = ap.parse_args(argv)
    raw = load_raw()
    audit_raw(raw)
    audit_dataset(Path(args.dataset), raw)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
