"""零依赖回归入口：最小 smoke + 关键历史回归 + pytest 聚合派发。

定位（P2 测试结构分层后）：
- **smoke**：确定性链路的「能跑通」级检查（黄金 IR 求解 / 特殊状态 / e2e / 重写闭环）。
- **关键历史回归**：与具体 P0/P1/P2 缺陷或协议约定绑定的**工具/脚本级**断言
  （config / pricing / 协议版本 / 状态映射 / 差异分类 / 提示词分层 / 语义不变 / 沙箱信封 /
  消融口径 / 分阶段记账）。
- **pytest 派发**：核心单元/集成断言已迁移到 `tests/`（test_ir_validation /
  test_solver_backend_parity / test_verifier_solution_integrity / test_llm_response_handling /
  test_orchestrator_routing / test_cache_context）。若环境中安装了 pytest，则本入口
  一并运行 `pytest tests/` 并聚合结果；否则打印清晰提示（`pip install -e .[dev]`）。

这样：无 pytest 环境仍能跑完 smoke + 关键回归；有 pytest 环境则得到完整分层覆盖。
"""
from __future__ import annotations

import json
import os
import re
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from nl2opt.gold_ir import get_gold_ir  # noqa: E402
from nl2opt.solver import solve  # noqa: E402
from nl2opt import verifier  # noqa: E402
from nl2opt.config import load_config  # noqa: E402
from nl2opt.dataset import load_builtin  # noqa: E402
from nl2opt.pipeline import run_dataset, run_sample  # noqa: E402


def approx(a, b, tol=1e-4):
    return abs(a - b) <= tol


def _run_pytest() -> tuple[int, int]:
    """尝试运行 tests/ 下的 pytest 套件，返回 (passed, total) 或 (0, -1) 表示不可用。"""
    try:
        import pytest  # noqa: F401
    except ImportError:
        return 0, -1
    import subprocess

    root = Path(__file__).resolve().parents[1]
    # 不用 -q：需要 "N passed" 汇总行；--no-header 去掉 platform 头部
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests", "--no-header", "-p", "no:cacheprovider"],
        cwd=str(root),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    tail = (proc.stdout or "") + (proc.stderr or "")
    m = re.search(r"(\d+)\s+passed", tail)
    passed = int(m.group(1)) if m else 0
    m2 = re.search(r"(\d+)\s+failed", tail)
    failed = int(m2.group(1)) if m2 else 0
    m3 = re.search(r"(\d+)\s+error", tail)
    errors = int(m3.group(1)) if m3 else 0
    total = passed + failed + errors
    if proc.returncode != 0 or failed or errors:
        print("\n[Pytest 套件输出]")
        print(tail[-4000:])
    return passed, total


def main() -> int:
    failures: list[str] = []
    total = 0

    def check(name, fn):
        nonlocal total
        total += 1
        try:
            fn()
            print(f"[PASS] {name}")
        except Exception as exc:
            failures.append(name)
            print(f"[FAIL] {name}: {exc}")
            traceback.print_exc()

    # ===================== Smoke：确定性链路能跑通 =====================

    # 1) 黄金 IR 求解值
    for sid, exp in [
        ("lp-01", 158.0), ("knapsack-01", 220.0), ("knapsack-02", 90.0),
        ("transport-01", 420.0), ("assignment-01", 5.0), ("integer-01", 40.0),
        ("diet-01", 15.6),
    ]:
        def _t(sid=sid, exp=exp):
            ir = get_gold_ir(sid)
            sol = solve(ir)
            assert sol.success, sol.message
            assert approx(sol.objective_value, exp), (sid, sol.objective_value, exp)
        check(f"gold_ir[{sid}] -> {exp}", _t)

    # 2) 特殊状态
    def _infeasible():
        assert solve(get_gold_ir("infeasible-01")).status == "infeasible"
    check("infeasible-01 status", _infeasible)

    def _unbounded():
        assert solve(get_gold_ir("unbounded-01")).status == "unbounded"
    check("unbounded-01 status", _unbounded)

    # 3) 端到端内置样例全部通过（无 LLM，确定性链路）
    def _e2e():
        cfg = load_config()
        ds = load_builtin()
        results = run_dataset(ds, llm=None, cfg=cfg)
        bad = [(r.sample_id, r.errors) for r in results if not r.success]
        assert not bad, bad
    check("e2e builtin all pass", _e2e)

    # 4) 重写闭环 smoke：首轮错 → 反思 → 重写修复
    def _rewrite_loop():
        from nl2opt.dataset import Sample
        from nl2opt.llm.fake import ScriptedProvider

        correct = get_gold_ir("knapsack-01").model_dump()
        wrong = json.loads(json.dumps(correct))
        for t in wrong["objective"]["expr"]["terms"]:
            if t["var"] == "x2":
                t["coef"] = 1000.0
        llm = ScriptedProvider([
            json.dumps(wrong, ensure_ascii=False),
            "建模错误：系数错误",
            json.dumps(correct, ensure_ascii=False),
        ])
        result = run_sample(Sample(id="test-rewrite", nl_text="测试重写闭环",
                                   problem_type="milp", expected={"objective_value": 220.0}),
                            llm=llm, cfg=load_config())
        assert result.success is True, result.errors
        assert result.rounds == 2
        assert result.objective_value == 220.0
    check("rewrite loop repairs IR (smoke)", _rewrite_loop)

    # ===================== 关键历史回归（工具/脚本/协议级） =====================

    # 5) 配置包装器：`in` / 遍历（否则会 KeyError: 0 崩溃）
    def _config_contains():
        cfg = load_config()
        models = cfg.llm.pricing.models
        assert "deepseek-flash" in models, "Config.__contains__ 未生效"
        assert "no-such-model" not in models
        assert set(iter(models)) >= {"deepseek-flash", "deepseek-v4-pro"}, list(models)
        assert "models" in cfg.llm.pricing and "default_model" in cfg.llm.pricing
        assert len(models) >= 2
    check("config supports 'in' & iteration", _config_contains)

    # 6) 成本报表：按模型选价（未登记模型须显式报错）
    def _pricing_by_model():
        import report_cost as rc
        cfg = load_config()
        name, blk = rc._resolve_pricing(cfg, "deepseek-v4-pro")
        assert name == "deepseek-v4-pro"
        assert rc._price(blk, "idle", "output", None) == 13.5
        _, blk2 = rc._resolve_pricing(cfg, "deepseek-flash")
        assert rc._price(blk2, "idle", "output", None) == 4.0
        assert rc._price(blk2, "peak", "output", None) == 8.0
        try:
            rc._resolve_pricing(cfg, "gpt-9")
        except SystemExit as exc:
            assert "未在" in str(exc)
        else:
            raise AssertionError("未登记模型未报错")
        assert rc._price(blk, "idle", "output", 99.0) == 99.0
    check("cost report picks price by model", _pricing_by_model)

    # 7) 协议版本校验：跨批次对比须版本一致，未知版本默认拒绝
    def _protocol_version():
        import ablation
        from nl2opt import PROTOCOL_VERSION
        assert isinstance(PROTOCOL_VERSION, int) and PROTOCOL_VERSION >= 1
        a = {"name": "a", "protocol_version": 3}
        b = {"name": "b", "protocol_version": 3}
        legacy = {"name": "legacy", "protocol_version": None}
        older = {"name": "older", "protocol_version": 2}
        assert ablation._pair_comparable(a, b, None)[0] is True
        ok, why = ablation._pair_comparable(a, older, None)
        assert ok is False and "v3" in why and "v2" in why, why
        ok, why = ablation._pair_comparable(a, legacy, None)
        assert ok is False and "未知" in why, why
        assert ablation._pair_comparable(a, legacy, 3)[0] is True
        assert ablation._pair_comparable(legacy, legacy, 3)[0] is True
        assert ablation._pair_comparable(legacy, legacy, None)[0] is False
        assert ablation._pair_comparable(a, legacy, 2)[0] is False
    check("protocol version gates cross-batch compare", _protocol_version)

    # 8) _map_status：status=1 一律 timeout（不变量：只设 time_limit）
    def _map_status_mapping():
        import inspect
        from nl2opt.solver import engine
        from nl2opt.solver.engine import _map_status

        class R:
            def __init__(self, st, msg=""):
                self.status = st
                self.message = msg

        assert _map_status(R(0)) == "optimal"
        for msg in ("", "time limit reached", "iteration limit", "whatever"):
            assert _map_status(R(1, msg)) == "timeout", msg
        assert _map_status(R(2)) == "infeasible"
        assert _map_status(R(3)) == "unbounded"
        assert _map_status(R(99)) == "error"
        assert _map_status(R(None)) == "error"
        src = inspect.getsource(engine.solve)
        assert "time_limit" in src
        for forbidden in ("node_limit", "iteration_limit", "simplex_iteration_limit"):
            assert forbidden not in src, f"新增 {forbidden} 后须重新区分 status=1"
    check("_map_status mapping + timeout invariant", _map_status_mapping)

    # 9) 目标差异细分：忽略权重 / 纯省略 / 同倍数 / 系数错配
    def _divergence_classify():
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "report_orbench_nl", Path(__file__).resolve().parents[1] / "scripts" / "report_orbench_nl.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        c = mod._classify_divergence
        with_w = '{"jobs": [{"weight": 4}, {"weight": 3}, {"weight": 1}]}'
        assert c([1, 2, 2, 3, 3, 3, 4, 4], [1] * 8, with_w) == "unit_weights"
        assert c([1, 2, 4, 4], [1, 1, 1, 1], with_w) == "unit_weights"
        assert c([1, 2, 4, 4], [1, 1, 1, 1], "no weights stated here") == "under_specified"
        assert c([1, 2, 4, 4], [1, 1, 1, 1]) == "under_specified"
        assert c([4, 5, 7, 11, 12, 12, 22, 38, 40], [4, 5, 7, 11, 12, 12, 38]) == "subset"
        assert c([1, 2, 3], [2, 4, 6]) == "scaled"
        assert c([3, 3, 8, 8], [5, 5, 6, 6]) == "other"
        assert c([2, 2, 2], [3, 3, 3], with_w) != "unit_weights"
    check("objective divergence classification", _divergence_classify)

    # 10) 提示词分层：卡片按题面触发，核心不含数据集专属词汇
    def _prompt_layering():
        from nl2opt.pipeline.prompts import CARDS, CORE_PROMPT, build_system_prompt, select_cards
        assert select_cards("at most 40% of shifts can be vans") == ["ratio"]
        assert select_cards("there must be at least 3 times as many X as Y") == ["multiple"]
        assert select_cards("Schedule jobs on machines M1 without overlap") == ["scheduling"]
        assert select_cards("plan 3 weeks of inventory for 2 SKUs") == ["inventory"]
        assert select_cards("Assign each task to at most one eligible technician") == ["assignment"]
        assert select_cards("Assign tasks to machines on a schedule") == ["scheduling", "assignment"]
        for phrase, want in [
            ("Invest at least three times as much as in healthcare", "multiple"),
            ("there must be three times more escalators than elevators", "multiple"),
            ("at most half as much chlorine as water softener", "multiple"),
            ("Limit the number of low-powered models to only 30%", "ratio"),
            ("a minimum of 20% should occur on Pi TV", "ratio"),
            ("the first mix contains 20% cat paw snacks", "ratio"),
        ]:
            got = select_cards(phrase)
            assert want in got, f"{phrase!r} 应触发 {want}，实得 {got}"
        assert "ratio" in select_cards("investments earn 3% and healthcare earn 1%")
        assert "收益率" in CARDS["ratio"]["body"]
        assert "fixed_cost" in CARDS and "piecewise" in CARDS
        # 核心不含数据集专属词（防过拟合：核心是跨族通用，专属词须进卡片）
        for token in ("pill", "dose", "van", "trip", "bowl", "meal", "serving", "diet"):
            assert token not in CORE_PROMPT, f"数据集专属词 {token} 泄漏进核心提示词"
        _ = build_system_prompt("test")
    check("prompt layering (core + cards)", _prompt_layering)

    # 11) 消融口径：prompt_mode 记录用于 A/B 配对
    def _prompt_mode_recorded():
        import importlib
        from nl2opt import cli
        from nl2opt.pipeline import PipelineResult

        def mk(meta):
            r = PipelineResult(sample_id="x")
            r.trajectory = [{"understand_meta": meta}]
            return r

        assert cli._prompt_mode(mk({"all_cards_mode": True})) == "all_cards"
        assert cli._prompt_mode(mk({"all_cards_mode": False})) == "layered"
        assert cli._prompt_mode(mk({})) == "layered"
        assert cli._prompt_mode(PipelineResult(sample_id="x")) == "layered"  # 无轨迹

        d = cli._result_to_dict(mk({"all_cards_mode": True}), model="m")
        assert d["prompt_mode"] == "all_cards", d
        assert d["protocol_version"] == importlib.import_module("nl2opt").PROTOCOL_VERSION
    check("prompt_mode recorded for ablation pairing", _prompt_mode_recorded)

    # 12) 语义不变性：约束顺序/变量重命名无关，能检出漏约束与定义域变化
    def _semantic_invariance():
        from nl2opt.ir import Constraint, IR, LinearExpr, Objective, Sense, Term, VarType, Variable
        from nl2opt.semantic import has_semantic_gap, semantic_diff

        def mk(cons_order=(0, 1, 2), names=("x", "y"),
               y_type=VarType.continuous, y_upper=None, drop_con=False, obj_y=2):
            nx, ny = names
            cons = [
                Constraint(name="c1", sense=Sense.le,
                           lhs=LinearExpr(terms=[Term(var=nx, coef=2), Term(var=ny, coef=1)]), rhs=10),
                Constraint(name="c2", sense=Sense.ge,
                           lhs=LinearExpr(terms=[Term(var=nx, coef=1), Term(var=ny, coef=3)]), rhs=8),
                Constraint(name="c3", sense=Sense.eq,
                           lhs=LinearExpr(terms=[Term(var=ny, coef=1)]), rhs=2),
            ]
            if drop_con:
                cons = cons[:2]
            return IR(
                variables=[Variable(name=nx, var_type=VarType.continuous, lower=0),
                           Variable(name=ny, var_type=y_type, lower=0, upper=y_upper)],
                objective=Objective(sense=Sense.maximize, expr=LinearExpr(
                    terms=[Term(var=nx, coef=3), Term(var=ny, coef=obj_y)])),
                constraints=[cons[i] for i in cons_order],
            )

        base = mk()
        assert not has_semantic_gap(semantic_diff(base, mk(cons_order=(2, 0, 1))))
        assert not has_semantic_gap(semantic_diff(base, mk(names=("u", "v"))))
        assert not has_semantic_gap(semantic_diff(mk(y_type=VarType.binary, y_upper=1),
                                                  mk(y_type=VarType.integer, y_upper=1)))
        assert has_semantic_gap(semantic_diff(base, mk(drop_con=True, cons_order=(0, 1))))
        assert semantic_diff(base, mk(y_type=VarType.integer))["domain_mismatch"] is True
        assert semantic_diff(base, mk(obj_y=5))["objective_mismatch"] is True
    check("semantic: order/rename invariant + detects gaps", _semantic_invariance)

    # 13) 沙箱信封解析（纯函数，不需要 Docker）
    def _sandbox_envelope():
        from nl2opt import sandbox as sb_mod
        raw = ("noise line from docker\n"
               '{"exec_status": "ok", "solver": {"status": "optimal", '
               '"objective_value": 5.0, "variables": {"x": 1.0}}}\n')
        env = sb_mod._parse_envelope(raw)
        assert env is not None and env["exec_status"] == "ok"
        res = sb_mod._to_sandbox_result(env)
        assert res.ok and res.solver_status == "optimal"
        assert res.objective_value == 5.0 and res.variables == {"x": 1.0}
    check("sandbox parses envelope", _sandbox_envelope)

    # 14) 消融判定语义：首轮通过 / 无验证接受条件 / 黄金值比对 / 目标退化
    def _ablation_logic():
        import ablation
        from nl2opt.ir import Constraint, IR, LinearExpr, Objective, Sense, Term, VarType, Variable

        ok_res = {"success": False, "rounds": 2, "trajectory": [
            {"understand_meta": {"source": "llm"}},
            {"round": 1, "checks": [{"ok": True}, {"ok": True}], "route": None},
        ]}
        assert ablation._first_round_ok(ok_res) is True
        bad_res = {"success": True, "rounds": 2, "trajectory": [
            {"understand_meta": {}},
            {"round": 1, "checks": [{"ok": False}, {"ok": True}], "route": "L2(重写IR)"},
            {"round": 2, "checks": [{"ok": True}]},
        ]}
        assert ablation._first_round_ok(bad_res) is False
        assert ablation._first_round_ok({"trajectory": [{"understand_meta": {}}]}) is False

        opt = {"trajectory": [{"round": 1, "status": "optimal", "objective_value": 2.0}]}
        assert ablation._definite_answer(opt) is True
        assert ablation._definite_answer({"trajectory": [{"round": 1, "status": "error"}]}) is False
        assert ablation._definite_answer({"status": "optimal"}) is False
        stuck = {"status": "repeated_error",
                 "trajectory": [{"round": 1, "status": "optimal", "objective_value": 2.0},
                                {"round": 2, "status": "optimal", "objective_value": 2.0}]}
        assert ablation._definite_answer(stuck) is True
        assert ablation._matches_gold(opt, {"objective_value": 2.0}) is True
        assert ablation._matches_gold(opt, {"objective_value": 63.0}) is False
        inf = {"trajectory": [{"round": 1, "status": "infeasible"}]}
        assert ablation._matches_gold(inf, {"status": "infeasible"}) is True
        assert ablation._matches_gold(stuck, {"status": "infeasible"}) is False

        def _ir_with(terms):
            return IR(
                variables=[Variable(name="t", var_type=VarType.continuous, lower=0)],
                objective=Objective(sense=Sense.minimize, expr=LinearExpr(terms=terms)),
                constraints=[Constraint(name="c1", sense=Sense.ge,
                                        lhs=LinearExpr(terms=[Term(var="t", coef=1)]), rhs=1)],
            ).model_dump()
        assert ablation._degenerate_objective({"final_ir": _ir_with([])}) is True
        assert ablation._degenerate_objective({"final_ir": _ir_with([Term(var="t", coef=0.0)])}) is True
        assert ablation._degenerate_objective({"final_ir": _ir_with([Term(var="t", coef=3.0)])}) is False
        assert ablation._degenerate_objective({}) is False
    check("ablation variant semantics", _ablation_logic)

    # 15) 分阶段 token 记账
    def _stage_usage():
        from nl2opt.dataset import Sample
        from nl2opt.llm.fake import ScriptedProvider
        correct = get_gold_ir("knapsack-01").model_dump()
        wrong = json.loads(json.dumps(correct))
        for t in wrong["objective"]["expr"]["terms"]:
            if t["var"] == "x2":
                t["coef"] = 1000.0
        llm = ScriptedProvider([
            json.dumps(wrong, ensure_ascii=False),
            "建模错误：系数错误",
            json.dumps(correct, ensure_ascii=False),
        ])
        result = run_sample(Sample(id="test-stage-usage", nl_text="测试分阶段记账",
                                   problem_type="milp", expected={"objective_value": 220.0}),
                            llm=llm, cfg=load_config())
        assert result.success is True, result.errors
        stages = result.stage_usage
        assert set(stages) == {"understand", "reflect", "rewrite"}, stages
        for name in ("understand", "reflect", "rewrite"):
            assert stages[name]["calls"] == 1, (name, stages[name])
        assert sum(s["calls"] for s in stages.values()) == result.token_usage.get("calls")
    check("stage usage attribution", _stage_usage)

    # 16) 求解超时 / 不限时接线
    def _solve_timeout():
        ir = get_gold_ir("knapsack-01")
        sol = solve(ir, time_limit=1e-9)
        assert sol.status in ("timeout", "optimal"), f"实际状态: {sol.status}"
    check("solve time_limit wired", _solve_timeout)

    def _solve_no_limit():
        sol = solve(get_gold_ir("knapsack-01"))
        assert sol.status == "optimal"
        assert sol.objective_value == 220.0
    check("solve default no limit works", _solve_no_limit)

    # ===================== 汇总 + pytest 派发 =====================

    inline = total - len(failures)
    print(f"\n通过 {inline}/{total}（smoke + 关键历史回归，零依赖）")

    pytest_passed, pytest_total = _run_pytest()
    if pytest_total == -1:
        print("[提示] 未安装 pytest，tests/ 下的分层套件跳过（`pip install -e .[dev]` 后可运行）")
        return 1 if failures else 0

    print(f"\n[pytest 分层套件] {pytest_passed}/{pytest_total} passed")
    if failures or (pytest_total > 0 and pytest_passed < pytest_total):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
