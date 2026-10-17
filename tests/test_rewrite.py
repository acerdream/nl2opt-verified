"""重写闭环测试：用 Fake LLM 模拟「理解错 → 反思 → 重写修复」。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nl2opt.config import load_config  # noqa: E402
from nl2opt.dataset import Sample  # noqa: E402
from nl2opt.gold_ir import get_gold_ir  # noqa: E402
from nl2opt.llm.fake import ScriptedProvider  # noqa: E402
from nl2opt.pipeline import run_sample  # noqa: E402


def _make_sample(correct_ir, wrong_ir) -> tuple[Sample, ScriptedProvider]:
    cfg_ir = correct_ir
    wrong = json.loads(json.dumps(wrong_ir))

    llm = ScriptedProvider(
        responses=[
            json.dumps(wrong, ensure_ascii=False),      # understand → 错
            "建模错误：系数错误，请修正。",                 # reflect
            json.dumps(cfg_ir, ensure_ascii=False),     # rewrite → 对
        ],
    )
    sample = Sample(
        id="test-rewrite",
        nl_text="测试重写闭环",
        problem_type="milp",
        expected={"objective_value": 220.0},
    )
    return sample, llm


def test_rewrite_loop_repairs_ir():
    correct = get_gold_ir("knapsack-01").model_dump()
    wrong = json.loads(json.dumps(correct))
    for t in wrong["objective"]["expr"]["terms"]:
        if t["var"] == "x2":
            t["coef"] = 1000.0

    sample, llm = _make_sample(correct, wrong)
    cfg = load_config()
    result = run_sample(sample, llm=llm, cfg=cfg)

    assert result.success is True
    assert result.rounds == 2
    assert result.objective_value == 220.0
    # 验证 LLM 被调用了 3 次：understand + reflect + rewrite
    assert len(llm.calls) == 3


def test_no_llm_stops_after_first_fail():
    # 无 LLM + 无黄金 IR + 无内嵌 IR → 无法理解，直接失败
    cfg = load_config()
    sample = Sample(
        id="unknown-id",
        nl_text="无相关 IR 的样例",
        problem_type="milp",
        expected={"objective_value": 1.0},
    )
    result = run_sample(sample, llm=None, cfg=cfg)
    assert result.success is False
    assert "无法理解" in "".join(result.errors)


def test_rewrite_failure_terminates():
    # rewrite 返回非法 JSON → 抛 LLMError → 终止并记录错误
    correct = get_gold_ir("knapsack-01").model_dump()
    wrong = json.loads(json.dumps(correct))
    for t in wrong["objective"]["expr"]["terms"]:
        if t["var"] == "x2":
            t["coef"] = 1000.0

    llm = ScriptedProvider(
        responses=[
            json.dumps(wrong, ensure_ascii=False),  # understand
            # v15 起 reflect 输出必须含有效层级；否则抛 ReflectionParseError 转 L4，
            # 不会走到 rewrite。这里显式给 L2，才能真正触发下面的 rewrite 失败。
            json.dumps({"layer": "L2", "advice": "修正"}),
            "这不是 JSON",                            # rewrite → 失败
        ],
    )
    sample = Sample(
        id="test-rewrite-fail",
        nl_text="测试重写失败",
        problem_type="milp",
        expected={"objective_value": 220.0},
    )
    cfg = load_config()
    result = run_sample(sample, llm=llm, cfg=cfg)
    assert result.success is False
    assert any("重写失败" in e for e in result.errors)


def test_L3_route_reunderstand():
    """L3 理解错误 → 重新调用 understand() 修复。"""
    correct = get_gold_ir("knapsack-01").model_dump()
    wrong = json.loads(json.dumps(correct))
    # 制造"漏约束"类理解错误：去掉 capacity 约束（求解会得到 0/1 全取，目标值 280 ≠ 220）
    wrong["constraints"] = []

    llm = ScriptedProvider(
        responses=[
            json.dumps(wrong, ensure_ascii=False),                     # understand(错)
            json.dumps({"layer": "L3", "advice": "漏了容量约束"}),      # reflect → L3
            json.dumps(correct, ensure_ascii=False),                   # understand(对)
        ],
    )
    sample = Sample(
        id="test-L3-route",
        nl_text="测试 L3 重新理解",
        problem_type="milp",
        expected={"objective_value": 220.0},
    )
    cfg = load_config()
    result = run_sample(sample, llm=llm, cfg=cfg)
    assert result.success is True, result.errors
    assert result.rounds == 2
    # 验证路由标记
    assert any("L3" in (t.get("route", "")) for t in result.trajectory)


def test_L4_route_manual():
    """L4 不可自动修复 → 转人工，停止。"""
    correct = get_gold_ir("knapsack-01").model_dump()
    wrong = json.loads(json.dumps(correct))
    wrong["constraints"] = []

    llm = ScriptedProvider(
        responses=[
            json.dumps(wrong, ensure_ascii=False),                     # understand(错)
            json.dumps({"layer": "L4", "advice": "问题信息不足"}),      # reflect → L4
        ],
    )
    sample = Sample(
        id="test-L4-route",
        nl_text="测试 L4 转人工",
        problem_type="milp",
        expected={"objective_value": 220.0},
    )
    cfg = load_config()
    result = run_sample(sample, llm=llm, cfg=cfg)
    assert result.success is False
    assert result.status == "manual_required"
    assert any("L4 转人工" in e for e in result.errors)
