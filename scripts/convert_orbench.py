"""OR-Bench Problems → 本项目 JSONL 转换（schema_payload → IR）。

数据来源：HuggingFace `AriaAICompany/or-bench-problems`（CC-BY-4.0，100 题）
  data/problems.jsonl   每题含 schema_payload（结构化 schema）+ prompt_en（自然语言题面）
                        + gold_objective / gold_status / gold_solution（OR-Tools CP-SAT 求得）

三题族的建模语义严格对齐官方 CP-SAT 参考实现（`demo/src/optimizer.py`）：

1. inventory（34 题）—— 整数规划，完全线性
   · packs[s][t] ≥ 0 整数；qty = pack_size * packs
   · inv[t] + sold[t] == prev_inv + qty；sold[t] + short[t] == demand[t]
   · 每期 Σ_s inv[s][t] ≤ warehouse_capacity；Σ unit_cost*qty ≤ budget
   · cover_required → short == 0
   · min Σ (unit_cost*qty + holding_cost*inv + shortage_cost*short)

2. assignment（33 题）—— 0-1 规划，完全线性
   · x[task][agent] ∈ {0,1}，仅对「资格 + 有成本」的组合建变量
     资格 = 技能交集非空 或 任务无技能要求
   · Σ_agent x + u[task] == 1；required → u[task] == 0
   · Σ_task size*x ≤ agent.capacity
   · min Σ cost*x + Σ penalty*u

3. scheduling（33 题）—— 需 big-M 析取编码（CP-SAT 用 NoOverlap 区间）
   · start[j] ∈ [release, H]，end[j] = start[j] + duration
   · tardy[j] ≥ end[j] - due；hard_due → end[j] ≤ due
   · 每作业恰选一台可用机器；同机器两两作业用 y 二进制 + big-M 保证不重叠
   · 先序 end[before] ≤ start[after]
   · min Σ weight * tardy（与 gold_objective 口径一致：官方报告的就是加权拖期，
     其求解器内部另有 makespan 作为 tie-break）

用法：
    python scripts/convert_orbench.py                        # 全量 100 题
    python scripts/convert_orbench.py --split development    # 仅开发集 60 题
    python scripts/convert_orbench.py --family inventory     # 单题族
    python scripts/convert_orbench.py --limit 5 --verbose    # 小样本 + 打印 IR 摘要
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nl2opt.ir import (  # noqa: E402
    Constraint,
    IR,
    LinearExpr,
    Objective,
    Sense,
    Term,
    VarType,
    Variable,
)

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "or-bench-problems" / "data" / "problems.jsonl"
DST = ROOT / "data" / "orbench.jsonl"


# --------------------------------------------------------------------------- #
# 建模工具
# --------------------------------------------------------------------------- #

def _expr(terms: list[tuple[str, float]] | None = None, constant: float = 0.0) -> LinearExpr:
    """(var, coef) 列表 → LinearExpr。"""
    return LinearExpr(
        terms=[Term(var=v, coef=float(c)) for v, c in (terms or [])],
        constant=float(constant),
    )


def _con(name: str, sense: Sense, terms, rhs: float, constant: float = 0.0) -> Constraint:
    return Constraint(name=name, sense=sense, lhs=_expr(terms, constant), rhs=float(rhs))


def _ivar(name: str, lower: float, upper: float) -> Variable:
    return Variable(name=name, var_type=VarType.integer, lower=float(lower), upper=float(upper))


def _bvar(name: str) -> Variable:
    return Variable(name=name, var_type=VarType.binary)


# --------------------------------------------------------------------------- #
# family: inventory
# --------------------------------------------------------------------------- #

def build_inventory(payload: dict[str, Any]) -> tuple[IR, dict[str, Any]]:
    periods = int(payload["periods"])
    budget = float(payload["budget"])
    cap = float(payload["warehouse_capacity"])
    cover_required = bool(payload.get("cover_required", False))
    skus = payload["skus"]

    max_demand = max((max(s["demand"]) for s in skus), default=1)
    max_pack = max((int(s["pack_size"]) for s in skus), default=1)
    max_order = max(max_demand * periods + max_pack * 4, 1)

    variables: list[Variable] = []
    constraints: list[Constraint] = []
    obj: list[tuple[str, float]] = []

    for s in skus:
        sid = s["sku_id"]
        pack = int(s["pack_size"])
        uc = float(s["unit_cost"])
        hc = float(s["holding_cost"])
        sc = float(s["shortage_cost"])
        init = float(s["initial_stock"])
        for t in range(1, periods + 1):
            demand = float(s["demand"][t - 1])
            packs = f"packs_{sid}_{t}"
            inv = f"inv_{sid}_{t}"
            sold = f"sold_{sid}_{t}"
            short = f"short_{sid}_{t}"

            variables.append(_ivar(packs, 0, max_order // pack + 1))
            variables.append(_ivar(inv, 0, cap))
            variables.append(_ivar(sold, 0, demand))
            variables.append(_ivar(short, 0, demand))

            # inv[t] + sold[t] == prev + pack*packs
            # t=1：prev = initial_stock（常数，移到 rhs）；t>1：prev = inv[s][t-1]（变量项）
            if t == 1:
                constraints.append(_con(
                    f"balance_{sid}_{t}", Sense.eq,
                    [(inv, 1), (sold, 1), (packs, -pack)], init,
                ))
            else:
                constraints.append(_con(
                    f"balance_{sid}_{t}", Sense.eq,
                    [(inv, 1), (sold, 1), (packs, -pack), (f"inv_{sid}_{t - 1}", -1)], 0.0,
                ))
            # sold + short == demand
            constraints.append(_con(
                f"demand_{sid}_{t}", Sense.eq,
                [(sold, 1), (short, 1)], demand,
            ))
            if cover_required:
                constraints.append(_con(f"cover_{sid}_{t}", Sense.eq, [(short, 1)], 0.0))

            obj.append((packs, uc * pack))
            obj.append((inv, hc))
            obj.append((short, sc))

    # 每期总库存 ≤ 仓库容量
    for t in range(1, periods + 1):
        constraints.append(_con(
            f"warehouse_{t}", Sense.le,
            [(f"inv_{s['sku_id']}_{t}", 1) for s in skus], cap,
        ))

    # 采购总成本 ≤ 预算
    constraints.append(_con(
        "budget", Sense.le,
        [(f"packs_{s['sku_id']}_{t}", float(s["unit_cost"]) * int(s["pack_size"]))
         for s in skus for t in range(1, periods + 1)],
        budget,
    ))

    ir = IR(
        variables=variables,
        objective=Objective(sense=Sense.minimize, expr=_expr(obj)),
        constraints=constraints,
        description=f"OR-Bench inventory: {len(skus)} SKU × {periods} 期库存计划",
    )
    meta = {"n_vars": len(variables), "n_cons": len(constraints)}
    return ir, meta


# --------------------------------------------------------------------------- #
# family: assignment
# --------------------------------------------------------------------------- #

def build_assignment(payload: dict[str, Any]) -> tuple[IR, dict[str, Any]]:
    tasks = payload["tasks"]
    agents = payload["agents"]
    costs = {(c["task_id"], c["agent_id"]): float(c["cost"]) for c in payload["costs"]}

    variables: list[Variable] = []
    constraints: list[Constraint] = []
    obj: list[tuple[str, float]] = []

    def eligible(task: dict, agent: dict) -> bool:
        return bool(set(task["skills"]) & set(agent["skills"])) or not task["skills"]

    x_pairs: list[tuple[str, str, str]] = []  # (var, task_id, agent_id)
    for task in tasks:
        tid = task["task_id"]
        assigned_terms: list[tuple[str, float]] = []
        for agent in agents:
            aid = agent["agent_id"]
            if eligible(task, agent) and (tid, aid) in costs:
                var = f"x_{tid}_{aid}"
                variables.append(_bvar(var))
                x_pairs.append((var, tid, aid))
                assigned_terms.append((var, 1.0))
                obj.append((var, costs[(tid, aid)]))

        u = f"u_{tid}"
        variables.append(_bvar(u))
        # Σ x + u == 1
        constraints.append(_con(
            f"assign_{tid}", Sense.eq, assigned_terms + [(u, 1.0)], 1.0,
        ))
        if bool(task.get("required", True)):
            constraints.append(_con(f"required_{tid}", Sense.eq, [(u, 1.0)], 0.0))
        obj.append((u, float(task["penalty"])))

    # 每个代理的容量约束（按任务 size 计）
    size_of = {t["task_id"]: float(t["size"]) for t in tasks}
    for agent in agents:
        aid = agent["agent_id"]
        load = [(var, size_of[tid]) for (var, tid, a) in x_pairs if a == aid]
        if load:
            constraints.append(_con(
                f"capacity_{aid}", Sense.le, load, float(agent["capacity"]),
            ))

    ir = IR(
        variables=variables,
        objective=Objective(sense=Sense.minimize, expr=_expr(obj)),
        constraints=constraints,
        description=f"OR-Bench assignment: {len(tasks)} 任务 × {len(agents)} 代理",
    )
    meta = {"n_vars": len(variables), "n_cons": len(constraints)}
    return ir, meta


# --------------------------------------------------------------------------- #
# family: scheduling（big-M 析取编码）
# --------------------------------------------------------------------------- #

def build_scheduling(payload: dict[str, Any]) -> tuple[IR, dict[str, Any]]:
    machines = list(payload["machines"])
    jobs = payload["jobs"]
    precedences = payload.get("precedences", []) or []

    total_dur = sum(int(j["duration"]) for j in jobs)
    max_release = max((int(j["release"]) for j in jobs), default=0)
    # 与官方参考一致的地平线口径
    horizon = max(int(payload["horizon"]), total_dur + max_release + 2)
    max_dur = max((int(j["duration"]) for j in jobs), default=1)
    big_m = float(horizon + max_dur + 1)

    variables: list[Variable] = []
    constraints: list[Constraint] = []
    obj: list[tuple[str, float]] = []

    job_by_id = {j["job_id"]: j for j in jobs}
    for j in jobs:
        jid = j["job_id"]
        rel = int(j["release"])
        dur = int(j["duration"])
        variables.append(_ivar(f"start_{jid}", rel, horizon))
        variables.append(_ivar(f"end_{jid}", rel + dur, horizon + dur))
        variables.append(_ivar(f"tardy_{jid}", 0, horizon + dur))
        # end == start + duration
        constraints.append(_con(
            f"dur_{jid}", Sense.eq,
            [(f"end_{jid}", 1), (f"start_{jid}", -1)], dur,
        ))
        # tardy >= end - due
        constraints.append(_con(
            f"tardy_def_{jid}", Sense.ge,
            [(f"tardy_{jid}", 1), (f"end_{jid}", -1)], -float(j["due"]),
        ))
        if bool(j.get("hard_due", False)):
            constraints.append(_con(f"hard_due_{jid}", Sense.le, [(f"end_{jid}", 1)], float(j["due"])))
        obj.append((f"tardy_{jid}", float(j["weight"])))

    # 机器分配：恰选一台可用机器
    eligible_map: dict[str, list[str]] = {}
    for j in jobs:
        jid = j["job_id"]
        elig = [m for m in machines if m in set(j.get("machine_eligible") or [])]
        if not elig:
            elig = list(machines)
        eligible_map[jid] = elig
        for m in elig:
            variables.append(_bvar(f"a_{jid}_{m}"))
        constraints.append(_con(
            f"pick_{jid}", Sense.eq, [(f"a_{jid}_{m}", 1.0) for m in elig], 1.0,
        ))

    # 同机器两两不重叠（big-M 析取）
    for m in machines:
        on_m = [j["job_id"] for j in jobs if m in eligible_map[j["job_id"]]]
        for a in range(len(on_m)):
            for b in range(a + 1, len(on_m)):
                i, k = on_m[a], on_m[b]
                y = f"y_{i}_{k}_{m}"
                variables.append(_bvar(y))
                ai, ak = f"a_{i}_{m}", f"a_{k}_{m}"
                # end_i - start_k + M*y + M*ai + M*ak <= 3M      （y=1 时 i 在 k 前）
                constraints.append(_con(
                    f"noverlap1_{i}_{k}_{m}", Sense.le,
                    [(f"end_{i}", 1), (f"start_{k}", -1), (y, big_m), (ai, big_m), (ak, big_m)],
                    3.0 * big_m,
                ))
                # end_k - start_i - M*y + M*ai + M*ak <= 2M       （y=0 时 k 在 i 前）
                constraints.append(_con(
                    f"noverlap2_{i}_{k}_{m}", Sense.le,
                    [(f"end_{k}", 1), (f"start_{i}", -1), (y, -big_m), (ai, big_m), (ak, big_m)],
                    2.0 * big_m,
                ))

    # 先序约束
    for edge in precedences:
        before, after = edge.get("before"), edge.get("after")
        if before in job_by_id and after in job_by_id:
            constraints.append(_con(
                f"prec_{before}_{after}", Sense.le,
                [(f"start_{before}", 1), (f"start_{after}", -1)],
                -float(job_by_id[before]["duration"]),
            ))

    ir = IR(
        variables=variables,
        objective=Objective(sense=Sense.minimize, expr=_expr(obj)),
        constraints=constraints,
        description=f"OR-Bench scheduling: {len(jobs)} 作业 × {len(machines)} 机器（big-M 不重叠）",
    )
    meta = {"n_vars": len(variables), "n_cons": len(constraints), "horizon": horizon, "big_m": big_m}
    return ir, meta


BUILDERS = {
    "inventory": build_inventory,
    "assignment": build_assignment,
    "scheduling": build_scheduling,
}


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #

def convert(rows: list[dict], verbose: bool = False, strip_ir: bool = False,
            attach_companion: bool = True) -> list[dict]:
    """schema_payload → 项目 JSONL。

    strip_ir=True 时不写入 extra.ir，使链路走「自然语言 → LLM 理解 → IR」，
    用于评测 T1（理解）环节；否则用内嵌 IR 走 T2–T5（建模→生成→执行→验证）。

    attach_companion=True 时，对**题面未内嵌 schema** 的题（官方 test split 的 prose
    模板）把 `prompt_json` 作为「JSON companion」**独立段落**附在题面后。

    为什么必须附（历史缺陷，实测驱动）：
      官方 test split 的设计是「散文题面 + JSON 附件」——
        · INV 题面明写 "Costs and opening stock are in the JSON companion."；
        · ASN/SCH 模板省略了这句，但其 cost / penalty / weight 同样**只存在于**
          prompt_json 中（散文里没有任何数值）。
      此前只取 `prompt_en`、丢弃 `prompt_json`，导致 test 40 题题面残缺
      （INV 甚至变成「指向不存在的附件」）。实测后果：
        development（题面内嵌 schema）58/60 = 96.7%
        test        （附件被丢弃）     11/40 = 27.5%
      故 test 的低分源于**输入缺失**，不能归因于模型建模能力。

    注意：development 题面已内嵌完整 SCHEMA_BEGIN 块，故**不重复附加**（nl_text 不变，
    保持与历史评测可比）。附件是**题目参数表**（成本/权重/需求），不是答案：
    黄金 IR 与目标值都不会因此泄露。
    """
    out: list[dict] = []
    for row in rows:
        family = row["family"]
        ir, meta = BUILDERS[family](row["schema_payload"])
        gold_status = row.get("gold_status")
        gold_obj = row.get("gold_objective")
        expected: dict[str, Any] = {"objective_value": gold_obj}
        if gold_status == "infeasible":
            expected = {"objective_value": None, "status": "infeasible"}

        prompt_en = row.get("prompt_en") or row["problem_id"]
        variant = "schema_embedded" if "SCHEMA_BEGIN" in prompt_en else "text_only"
        companion = (row.get("prompt_json") or "").strip()
        # 题面是否**自己**声明了附件（仅 INV 模板有；ASN/SCH 靠同一 split 的设计推断）
        referenced = "json companion" in prompt_en.lower()
        nl_text = prompt_en
        attached = False
        if attach_companion and variant == "text_only" and companion:
            nl_text = f"{prompt_en}\n\n--- JSON companion ---\n{companion}"
            attached = True

        extra: dict[str, Any] = {
            "family": family,
            "split": row.get("split"),
            "difficulty": row.get("difficulty"),
            "gold_status": gold_status,
            "template_id": row.get("template_id"),
            "prompt_variant": variant,
            "companion_attached": attached,
            "companion_referenced_in_prompt": referenced,
            "source": "AriaAICompany/or-bench-problems",
            "build_meta": meta,
        }
        if not strip_ir:
            # 参考 IR 仍保留（便于复盘对比），但链路会用 extra.ir 直接建模
            extra["ir"] = ir.model_dump()
        else:
            # 仅留参考 IR 的规模信息，避免污染理解链路
            extra["reference_ir_size"] = {"vars": meta["n_vars"], "cons": meta["n_cons"]}
        out.append({
            "id": row["problem_id"],
            "problem_type": "milp",
            "nl_text": nl_text,
            "expected": expected,
            "extra": extra,
            # 黄金值由 OR-Tools CP-SAT 在官方 schema 上求得，可信
            "trusted_reference": True,
        })
        if verbose:
            print(f"  {row['problem_id']:9s} {family:11s} vars={meta['n_vars']:4d} "
                  f"cons={meta['n_cons']:4d} gold={gold_obj} ({gold_status}) "
                  f"companion={'+' if attached else '-'}")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="OR-Bench Problems → 项目 JSONL")
    ap.add_argument("--src", default=str(SRC), help="problems.jsonl 路径")
    ap.add_argument("--out", default=str(DST), help="输出 JSONL 路径")
    ap.add_argument("--split", choices=["development", "test", "all"], default="all")
    ap.add_argument("--family", choices=["inventory", "assignment", "scheduling", "all"], default="all")
    ap.add_argument("--limit", type=int, default=0, help="仅取前 N 题（0=全部）")
    ap.add_argument("--strip-ir", action="store_true",
                    help="不写入内嵌 IR，使链路走「自然语言 → LLM 理解 → IR」（T1 评测）")
    ap.add_argument("--no-companion", action="store_true",
                    help="不附加 JSON companion（生成历史对照数据用；默认附加）")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    src = Path(args.src)
    if not src.exists():
        print(f"找不到数据文件: {src}", file=sys.stderr)
        print("请先下载：git clone --depth 1 https://hf-mirror.com/datasets/"
              "AriaAICompany/or-bench-problems data/or-bench-problems", file=sys.stderr)
        return 2

    rows = [json.loads(line) for line in src.open(encoding="utf-8") if line.strip()]
    if args.split != "all":
        rows = [r for r in rows if r.get("split") == args.split]
    if args.family != "all":
        rows = [r for r in rows if r.get("family") == args.family]
    if args.limit:
        rows = rows[: args.limit]

    print(f"转换 {len(rows)} 题 …")
    out = convert(rows, verbose=args.verbose, strip_ir=args.strip_ir,
                  attach_companion=not args.no_companion)

    dst = Path(args.out)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8") as fh:
        for item in out:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")

    by_family: dict[str, int] = {}
    infeasible = 0
    for item in out:
        by_family[item["extra"]["family"]] = by_family.get(item["extra"]["family"], 0) + 1
        if item["expected"].get("status") == "infeasible":
            infeasible += 1
    print(f"已写入: {dst}")
    print(f"题族分布: {by_family} | 不可行题: {infeasible}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
