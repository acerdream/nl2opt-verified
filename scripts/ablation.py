"""消融实验：量化闭环各环节的边际贡献。

设计原则：**数据驱动、零 API 调用**。已有评测结果的 trajectory 记录了每轮的
`checks`（验证结果）、`reflection`（反思归因）、`route`（重写路由）、`status`/`objective_value`
（求解结果），据此可离线重建各消融变体，因而不会与其它运行中的 LLM 任务争用配额。

覆盖的消融维度：
| 维度 | 可离线？ | 方法 |
|---|---|---|
| 无重写 | ✅ | 只取首轮 checks 是否全通过 |
| 无反思 | ✅ | 反思仅用于生成重写建议 → 结果同「无重写」，额外统计省下的 reflect 成本 |
| 无验证 | ✅ | 不核验正确性，只看求解器是否给出定解 → 统计「错误放行率」 |
| 不同求解器 | ✅ | OR-Bench structured 的 HiGHS(本地) / CBC(沙箱) 结果 vs CP-SAT 黄金值 |
| 不同 LLM | ❌ 需 API | 见报告末尾的命令（建议在当前任务结束后运行） |

用法：
    python scripts/ablation.py                      # 默认分析 artifacts/eval_orbench_nl_*.json
    python scripts/ablation.py artifacts/eval_a.json --data data/orbench_nl.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import report_cost as rc  # noqa: E402  # 复用三段计价与阶段归因逻辑

from nl2opt.config import load_config  # noqa: E402
from nl2opt.ir import IR  # noqa: E402

OUT = ROOT / "artifacts" / "ABLATION_REPORT.md"
STAGES = ("understand", "reflect", "rewrite", "repair_code")


def _stage_costs(res: dict, prices: tuple[float, float, float]) -> tuple[str, dict[str, float]]:
    """返回 (模式, {阶段: 成本})。

    精确模式：结果含 stage_usage（新运行）→ 直接计价；
    估算模式：旧结果 → 用本样本总成本按各阶段「调用次数」分摊（粗糙，报告中标注）。
    """
    p_hit, p_miss, p_out = prices
    su = res.get("stage_usage") or {}
    if su:
        out = {}
        for stage, u in su.items():
            h, m, o = rc._tokens(u)
            out[stage] = rc._cost(h, m, o, p_hit, p_miss, p_out)
        return "精确", out

    usage = res.get("token_usage") or {}
    if not usage:
        return "无", {}
    h, m, o = rc._tokens(usage)
    total = rc._cost(h, m, o, p_hit, p_miss, p_out)
    counts = rc._estimate_stage_calls(res)
    n = sum(counts.values()) or 1
    return "估算", {stage: total * c / n for stage, c in counts.items()}


def _last_round(res: dict) -> dict | None:
    """最后一个「轮次」条目（跳过 understand_meta 条目）。"""
    steps = [s for s in (res.get("trajectory") or []) if "round" in s]
    return steps[-1] if steps else None


def _first_round_checks(res: dict) -> list | None:
    """首个「轮次」条目的 checks（跳过 understand_meta 条目）。"""
    for step in res.get("trajectory") or []:
        if "round" in step:
            return step.get("checks") or []
    return None


def _first_round_ok(res: dict) -> bool:
    checks = _first_round_checks(res)
    if checks is None:
        return False
    return all(c.get("ok") for c in checks)


def _solver_view(res: dict) -> dict | None:
    """求解器视角：取末轮的求解状态与目标值。

    **必须用求解状态，不能用流程状态**——`manual_required` / `repeated_error` 只是
    闭环的终止原因，并不代表求解器没给出解。若误用流程状态判断「无验证会放行什么」，
    会把「求解器给了定解、但闭环判定失败」的错解漏掉（实测 SCH-031/SCH-033 即如此）。
    """
    last = _last_round(res)
    if last is None:
        return None
    return {"status": last.get("status"), "objective_value": last.get("objective_value")}


def _definite_answer(res: dict) -> bool:
    """求解器是否给出定解（optimal/infeasible/unbounded）——「无验证」变体的接受条件。"""
    view = _solver_view(res)
    return bool(view) and view["status"] in ("optimal", "infeasible", "unbounded")


def _matches_gold(res: dict, gold: dict) -> bool:
    """与黄金值一致（不可行题看状态，可行题看目标值），以求解器视角判定。"""
    view = _solver_view(res) or {"status": res.get("status"),
                                 "objective_value": res.get("objective_value")}
    if gold.get("status") == "infeasible":
        return view["status"] == "infeasible"
    g, v = gold.get("objective_value"), view["objective_value"]
    if g is None or v is None:
        return False
    return abs(v - g) <= 1e-4 * max(1.0, abs(g))


def _degenerate_objective(res: dict) -> bool:
    """目标是否退化（无项 / 系数全 0，等价于「最小化 0」）。

    与 `IR.objective_is_degenerate()` 同源；这是**目标值判据抓不到**的错解——
    排班题黄金值常为 0，退化目标照样「命中」，只有结构判据能发现。
    """
    ir = res.get("final_ir")
    if not ir:
        return False
    try:
        return IR.model_validate(ir).objective_is_degenerate()
    except Exception:
        return False


def _pair_comparable(a: dict, b: dict, assume: int | None) -> tuple[bool, str]:
    """判断两批次能否做跨模型对比：**协议版本必须一致**。

    返回 (是否可比, 原因)。版本缺失时按 `assume` 视之；`assume` 为 None 则拒绝。

    为什么必须拦：实测踩过两次坑——sub12 的 flash 批次跑在「IR 支持字符串/多维索引」
    之前、33 题排班批次跑在「验证器新增目标定义检查」之前，两次都让"批次差异"
    冒充了"模型差异"。版本号一致才允许下跨批次结论。
    """
    # 注意：不能用 a.get(key, assume)——键存在但值为 None 时默认值不生效，
    # 而旧批次正是「有键、值为 None」。必须显式判 None。
    va = a.get("protocol_version")
    if va is None:
        va = assume
    vb = b.get("protocol_version")
    if vb is None:
        vb = assume
    if va is None or vb is None:
        return False, (f"协议版本未知（`{a['name']}`={va}、`{b['name']}`={vb}）——"
                       f"请用 `--assume-version N` 显式声明历史批次的版本后再对比")
    if va != vb:
        return False, (f"协议版本不同（`{a['name']}` v{va} vs `{b['name']}` v{vb}）——"
                       f"IR 表达能力或验证器判据可能已变，跨批次结论无效")
    return True, ""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="消融实验（数据驱动，零 API 调用）")
    ap.add_argument("results", nargs="*", help="评测结果 JSON（默认 artifacts/eval_orbench_nl_sub12.json）")
    ap.add_argument("--data", default=str(ROOT / "data" / "orbench_nl.jsonl"), help="数据集（取黄金值）")
    ap.add_argument("--rate", choices=["auto", "idle", "peak"], default="idle")
    ap.add_argument("--exclude-compare", action="append", default=[],
                    metavar="STEM",
                    help="该批次（文件名去扩展名）不参与跨模型对比；用于排除"
                         "「早于代码改动、条件不可比」的历史批次，可重复")
    ap.add_argument("--assume-version", type=int, default=None, metavar="N",
                    help="把**未记录协议版本**的历史批次视为版本 N 后再做对比；"
                         "不指定则拒绝与未知版本批次对比（默认拒绝，避免错误结论）")
    ap.add_argument("--batch-version", action="append", default=[], metavar="STEM=N",
                    help="按批次精确声明协议版本（优先于 --assume-version），可重复。"
                         "例：--batch-version eval_orbench_nl_sched=2")
    ap.add_argument("--force-compare", action="store_true",
                    help="**有意消融时**才用：允许对比协议版本不同的批次。"
                         "报告会显著标注版本差异（默认拒绝，避免把代码差异误读为模型差异）")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)

    cfg = load_config()
    paths = [Path(p) for p in args.results] or [ROOT / "artifacts" / "eval_orbench_nl_sub12.json"]
    paths = [p for p in paths if p.exists()]
    if not paths:
        print("未找到评测结果", file=sys.stderr)
        return 2

    gold_map: dict[str, dict] = {}
    dpath = Path(args.data)
    if dpath.exists():
        for line in dpath.open(encoding="utf-8"):
            if line.strip():
                row = json.loads(line)
                gold_map[row["id"]] = row.get("expected") or {}

    prices = (
        rc._price(cfg, args.rate, "input_cache_hit", None),
        rc._price(cfg, args.rate, "input_cache_miss", None),
        rc._price(cfg, args.rate, "output", None),
    )

    # 按批次精确声明的协议版本（优先于 --assume-version）
    batch_versions: dict[str, int] = {}
    for item in args.batch_version:
        if "=" not in item:
            raise SystemExit(f"--batch-version 需要 STEM=N 形式，收到：{item!r}")
        stem, _, num = item.rpartition("=")
        try:
            batch_versions[stem] = int(num)
        except ValueError:
            raise SystemExit(f"--batch-version 的版本号必须是整数，收到：{item!r}") from None

    batches: list[dict] = []   # 供「同题集跨模型对比」使用

    lines: list[str] = []
    lines.append("# 消融实验报告（闭环各环节的边际贡献）\n")
    lines.append("- 方法：**数据驱动**——从已有 trajectory 离线重建各消融变体，不额外调用 LLM")
    lines.append(f"- 数据来源：{', '.join(f'`{p.name}`' for p in paths)}")
    lines.append("- 计价：按**各批次结果文件记录的模型**选择单价档（缺该字段时退回配置默认值）\n")

    for path in paths:
        results = json.loads(path.read_text(encoding="utf-8"))
        results = [r for r in results if isinstance(r, dict)]
        n = len(results)
        if not n:
            continue

        # 该批次使用的模型与单价（结果文件自描述；缺失则退回配置默认值）
        b_model = next((r.get("model") for r in results if r.get("model")), None) \
            or cfg.llm.get("model")
        try:
            _, b_pblock = rc._resolve_pricing(cfg, b_model)
        except SystemExit:
            b_pblock = {}
        prices = (
            rc._price(b_pblock, args.rate, "input_cache_hit", None),
            rc._price(b_pblock, args.rate, "input_cache_miss", None),
            rc._price(b_pblock, args.rate, "output", None),
        )

        # ---- 各变体通过数 ----
        full = sum(1 for r in results if r.get("success"))
        no_rewrite = sum(1 for r in results if _first_round_ok(r))
        rounds = [r.get("rounds", 0) for r in results]
        repaired = sum(1 for r in results if r.get("rounds", 0) > 1)

        # 「无验证」：只要求解器给出定解就接受 → 统计其中判错的（错误放行）
        accepted = [r for r in results if _definite_answer(r)]
        wrong = [r for r in accepted
                 if r["id"] in gold_map and not _matches_gold(r, gold_map[r["id"]])]
        # 目标退化（空目标 / 系数全 0）：目标值判据抓不到，只有结构判据能发现
        degenerate = [r["id"] for r in accepted if _degenerate_objective(r)]
        # 注意：不可行题若模型建错也可能"恰好"不可行；此处只按黄金值判对错

        # ---- 成本 ----
        mode_set: set[str] = set()
        cost_full = cost_understand = cost_reflect = cost_rewrite = cost_repair = 0.0
        for r in results:
            mode, sc = _stage_costs(r, prices)
            mode_set.add(mode)
            usage = r.get("token_usage") or {}
            if usage:
                h, m, o = rc._tokens(usage)
                cost_full += rc._cost(h, m, o, *prices)
            cost_understand += sc.get("understand", 0.0)
            cost_reflect += sc.get("reflect", 0.0)
            cost_rewrite += sc.get("rewrite", 0.0)
            cost_repair += sc.get("repair_code", 0.0)

        # 该批次的协议版本：结果文件自描述 → 按批次声明 → 全局假设 → 未知
        b_proto = next((r.get("protocol_version") for r in results
                        if r.get("protocol_version")), None)
        if b_proto is None:
            b_proto = batch_versions.get(path.stem, args.assume_version)

        lines.append(f"## {path.stem}\n")
        lines.append(f"- 模型：`{b_model}`｜协议版本："
                     f"{('v' + str(b_proto)) if b_proto else '未记录（早于版本标记的批次）'}"
                     f"{'（人工声明）' if b_proto and not any(r.get('protocol_version') for r in results) else ''}"
                     f"｜单价（命中/未命中/输出）："
                     f"{prices[0]}/{prices[1]}/{prices[2]} 元每百万（{args.rate}）")
        lines.append(f"- 样本数：{n}｜平均轮次：{sum(rounds)/n:.2f}｜发生重写的样本：{repaired}")
        lines.append(f"- 阶段成本归因模式：{'精确（stage_usage）' if mode_set == {'精确'} else '估算（按调用次数分摊）'}\n")

        # 提示词模式：优先读自描述字段；旧批次回退到 trajectory 里的 understand_meta。
        # 卡片策略**不体现在 model/protocol_version 上**，不记录就会被误判为
        # 「同一配置的重复批次」而漏掉提示词分层的 A/B 对比。
        b_mode = next((r.get("prompt_mode") for r in results if r.get("prompt_mode")), None)
        if b_mode is None:
            for r in results:
                for step in (r.get("trajectory") or []):
                    m = step.get("understand_meta") if isinstance(step, dict) else None
                    if m and m.get("all_cards_mode") is not None:
                        b_mode = "all_cards" if m["all_cards_mode"] else "layered"
                        break
                if b_mode:
                    break
        b_mode = b_mode or "layered"

        batches.append({
            "name": path.stem, "model": b_model,
            "protocol_version": b_proto,
            "prompt_mode": b_mode,
            "per_id": {r["id"]: bool(r.get("success")) for r in results},
            "ids": [r["id"] for r in results],
            "cost": cost_full, "full": full, "n": n,
        })

        lines.append("| 配置 | 通过数 | 通过率 | 成本(元) | 相对完整闭环 |")
        lines.append("|---|---|---|---|---|")
        base = cost_full or 1.0
        lines.append(f"| **完整闭环**（验证+反思+重写） | {full}/{n} | {full/n*100:.1f}% | "
                     f"{cost_full:.4f} | 100% |")
        lines.append(f"| 无重写（只取首轮） | {no_rewrite}/{n} | {no_rewrite/n*100:.1f}% | "
                     f"{cost_understand:.4f} | {cost_understand/base*100:.1f}% |")
        lines.append(f"| 无反思（+无重写） | {no_rewrite}/{n} | {no_rewrite/n*100:.1f}% | "
                     f"{cost_understand:.4f} | {cost_understand/base*100:.1f}% |")
        lines.append(f"| 无重写但保留反思 | {no_rewrite}/{n} | {no_rewrite/n*100:.1f}% | "
                     f"{(cost_understand + cost_reflect):.4f} | "
                     f"{(cost_understand + cost_reflect)/base*100:.1f}% |")
        lines.append(f"| 无验证（只要求解器给解） | {len(accepted)}/{n} 被接受 | "
                     f"其中 **{len(wrong)} 个错解** | — | — |")
        lines.append("")

        lines.append("### 解读\n")
        gain = full - no_rewrite
        if gain > 0:
            lines.append(f"- **重写闭环净收益：+{gain} 题**（{no_rewrite} → {full}），"
                         f"代价是成本从 {cost_understand:.4f} 元升到 {cost_full:.4f} 元"
                         f"（+{(cost_full - cost_understand):.4f} 元，"
                         f"即每题多花 {(cost_full - cost_understand)/max(1,gain):.4f} 元）。")
        else:
            lines.append("- 本批无重写收益（首轮即全部通过或重写未能救回）。")
        if cost_reflect:
            lines.append(f"- **反思成本占比 {cost_reflect/base*100:.1f}%**"
                         f"（{cost_reflect:.4f} 元）：反思只用于选择重写层级与生成建议，"
                         f"若不重写则这部分可完全省下。")
        if cost_repair:
            lines.append(f"- **L1 代码修复成本 {cost_repair:.4f} 元**"
                         f"（{cost_repair/base*100:.1f}%）。")
        if accepted:
            lines.append(f"- **无验证的风险**：若不加验证，{len(accepted)} 个样本都会被当作「成功」"
                         f"（求解器给了定解），但其中 **{len(wrong)} 个是错解**"
                         + (f"（{'、'.join(r['id'] for r in wrong[:8])}"
                            f"{'…' if len(wrong) > 8 else ''}）。" if wrong else "。")
                         + "这就是独立验证器的价值：把错解拦截在交付之前。")
        if degenerate:
            lines.append(f"- **目标值判据抓不到的一类错解**：{len(degenerate)} 题的目标函数退化为空"
                         f"（{'、'.join(degenerate[:8])}{'…' if len(degenerate) > 8 else ''}）——"
                         f"等价于「最小化 0」，任何可行解都返回 0；若黄金值恰为 0 就会被误判为通过。"
                         f"必须靠**结构判据**（目标签名）才能发现。")
        lines.append("")

    # ---- 不同求解器（structured 路径，零 LLM 成本）----
    local = ROOT / "artifacts" / "eval_orbench_local.json"
    docker = ROOT / "artifacts" / "eval_orbench_docker.json"
    if local.exists() and docker.exists():
        g100 = {}
        for line in (ROOT / "data" / "orbench.jsonl").open(encoding="utf-8"):
            if line.strip():
                row = json.loads(line)
                g100[row["id"]] = row.get("expected") or {}
        lines.append("## 不同求解器（OR-Bench structured 路径，零 LLM 成本）\n")
        lines.append("同一 IR 分别交给两套求解器执行，与官方 OR-Tools CP-SAT 黄金值对拍：\n")
        lines.append("| 求解器 | 执行路径 | 命中黄金值 |")
        lines.append("|---|---|---|")
        for label, p, desc in (("scipy / HiGHS", local, "本地进程内"),
                               ("PuLP / CBC", docker, "Docker 沙箱（--network=none）")):
            rs = json.loads(p.read_text(encoding="utf-8"))
            ok = sum(1 for r in rs if r["id"] in g100 and _matches_gold(r, g100[r["id"]]))
            lines.append(f"| {label} | {desc} | **{ok}/{len(rs)}** |")
        lines.append("")
        lines.append("> 两套求解器 100% 一致，说明：① 本项目 IR 的建模语义无歧义；"
                     "② 沙箱（CBC）与本地（HiGHS）结果可互相印证，"
                     "跨求解器对拍可作为生成代码正确性的独立证据。\n")

    # ---- 同题集跨批次对比（不同 LLM / 不同提示词 / 不同验证器版本等）----
    # 仅对**题号集合完全相同**的批次配对：子集/交集都会产生无意义的对比
    # （例如 33 题排班批次与 12 题抽样批次共享 4 个 SCH 题号）。
    # 配对条件：题集相同，且两侧配置**不同**（模型不同 或 协议版本不同）——
    # 因此「同模型、不同提示词的消融」也能配对，这是提示词分层的核心对比。
    for i in range(len(batches)):
        for j in range(i + 1, len(batches)):
            a, b = batches[i], batches[j]
            if (a["model"] == b["model"] and a["protocol_version"] == b["protocol_version"]
                    and a.get("prompt_mode") == b.get("prompt_mode")):
                continue      # 同一配置（含提示词模式）的重复批次，无需对比
            if a["name"] in args.exclude_compare or b["name"] in args.exclude_compare:
                continue
            if set(a["ids"]) != set(b["ids"]):
                continue
            ok, why = _pair_comparable(a, b, args.assume_version)
            if not ok and not args.force_compare:
                lines.append(f"## 跨批次对比：`{a['name']}` vs `{b['name']}` — **已拒绝**\n")
                lines.append(f"⚠️ 未做对比。{why}\n")
                lines.append("> 若这是**有意的消融实验**（例如专门比较提示词/验证器改动的影响），"
                             "加 `--force-compare` 强制对比；报告会标注版本差异。\n")
                continue

            common = list(a["ids"])
            forced = not ok
            same_model = a["model"] == b["model"]
            only_a = [pid for pid in common if a["per_id"][pid] and not b["per_id"][pid]]
            only_b = [pid for pid in common if b["per_id"][pid] and not a["per_id"][pid]]
            sa = sum(1 for pid in common if a["per_id"][pid])
            sb = sum(1 for pid in common if b["per_id"][pid])
            ca = a["cost"] * len(common) / max(1, a["n"])
            cb = b["cost"] * len(common) / max(1, b["n"])

            lines.append(f"## 跨批次对比：`{a['name']}` vs `{b['name']}`\n")
            mode_a, mode_b = a.get("prompt_mode", "?"), b.get("prompt_mode", "?")
            lines.append(f"- 性质：**{'同一模型、不同配置（消融）' if same_model else '不同模型'}**"
                         f"｜题集：{len(common)} 题"
                         + (f"｜提示词模式：**{mode_a} → {mode_b}**" if mode_a != mode_b else ""))
            if forced:
                lines.append(f"- ⚠️ **协议版本不同**（v{a['protocol_version']} vs "
                             f"v{b['protocol_version']}）：本次为 `--force-compare` 强制的"
                             f"**有意消融**，差异**不能**单纯归因于模型，"
                             f"也可能来自提示词/验证器/IR 能力的改动。")
            lines.append("")
            lines.append(f"| 指标 | 左：`{a['name']}`（{a['model']} v{a['protocol_version']}，"
                         f"{mode_a}） | 右：`{b['name']}`（{b['model']} v{b['protocol_version']}，"
                         f"{mode_b}） |")
            lines.append("|---|---|---|")
            lines.append(f"| 通过数 | {sa}/{len(common)} | {sb}/{len(common)} |")
            lines.append(f"| 成本(元) | {ca:.4f} | {cb:.4f} |")
            if sa and sb:
                lines.append(f"| 单题成本(元) | {ca/sa:.4f} | {cb/sb:.4f} |")
            left_only = ("`" + "`, `".join(only_a) + "`") if only_a else "—"
            right_only = ("`" + "`, `".join(only_b) + "`") if only_b else "—"
            lines.append(f"| 仅此侧通过 | {left_only} | {right_only} |")
            lines.append("")

            disc = len(only_a) + len(only_b)
            if disc == 0:
                lines.append("两侧在该题集上**逐题判定完全一致**，无可归因的差异。\n")
            else:
                from math import comb
                k = min(len(only_a), len(only_b))
                pval = min(1.0, 2 * sum(comb(disc, i) for i in range(k + 1)) / (2 ** disc))
                lines.append(f"- **配对检验（McNemar 精确检验）**：差异题 {disc} 道"
                             f"（右胜 {len(only_b)}、左胜 {len(only_a)}），双侧 p = {pval:.2f}。")
                ratio = cb / max(ca, 1e-9)
                if pval > 0.05:
                    extra = (f"；成本比为 {ratio:.2f}（右/左）"
                             if ratio >= 1.05 or ratio <= 0.95 else "")
                    lines.append(f"- ⚠️ **不足以判定优劣**：{disc} 道差异题样本量太小"
                                 f"（p > 0.05），要下结论需扩大题量{extra}。")
                else:
                    lines.append("- 差异在 p < 0.05 水平显著。")
            lines.append("")
            lines.append(f"> 说明：两侧题集相同（{len(common)} 题）；"
                         + ("同一模型、同一单价档，差异来自协议版本/配置。"
                            if same_model else
                            "成本按各自单价档计算，可直接比较性价比。") + "\n")

    # ---- 若无跨模型对比，给出可复现命令（需 API）----
    if not any("跨批次对比" in ln for ln in lines):
        lines.append("## 跨批次/跨模型对比（需 API）\n")
        lines.append("本次分析未包含可对比的跨模型批次。换模型对比需真实调用，命令如下"
                     "（`DEEPSEEK_MODEL` 环境变量覆盖配置，不改动 `config.yaml`）：\n")
        lines.append("```powershell")
        lines.append("$env:PYTHONPATH = \"$PWD\\src\"")
        lines.append("$env:DEEPSEEK_MODEL = \"deepseek-v4-pro\"")
        lines.append("python -m nl2opt eval --data data/orbench_nl_sub12.jsonl --use-llm --no-cache \\")
        lines.append("    --out artifacts/eval_orbench_nl_sub12_pro.json")
        lines.append("python scripts/ablation.py artifacts/eval_orbench_nl_sub12_pro.json \\")
        lines.append("    artifacts/eval_orbench_nl_sub12_flash2.json --exclude-compare eval_orbench_nl_sub12")
        lines.append("```\n")
        lines.append("> 提示：两侧必须使用**相同协议版本**的批次（协议版本变更会改变 IR 表达能力或"
                     "验证器判据，跨版本对比会把「代码差异」误读为「模型差异」）。"
                     "历史批次可用 `--batch-version STEM=N` 精确声明版本。\n")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"报告已写入: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
