"""语义覆盖诊断可读性 / detail 可落盘 / 执行错误位置的回归测试。

锁定三处「声明与实现脱节」的修复（不改变判定，只补全诊断信息）：
① CheckResult.detail 可 JSON 落盘（此前 semantic_diff 的 gold_objective 是 tuple，json.dumps 会抛 TypeError）；
② describe_gap 给出具体对象（系数/签名）而非仅计数；
④ _exec_error_head 含出错文件:行号，不再只取末行。
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nl2opt import verifier  # noqa: E402
from nl2opt.gold_ir import get_gold_ir  # noqa: E402
from nl2opt.semantic import describe_gap, semantic_diff  # noqa: E402


def _broken(gold):
    broken = copy.deepcopy(gold)
    broken.constraints = broken.constraints[:-1]  # 删掉最后一条约束
    return broken


def test_describe_gap_names_objects():
    """② describe_gap 应给出具体约束签名/变量角色，而非只给数量。"""
    gold = get_gold_ir("knapsack-01")
    diff = semantic_diff(gold, _broken(gold))
    msg = describe_gap(diff)
    assert "漏约束 1 条" in msg
    # 含具体签名（系数 + sense + rhs），而不是只有「1 条」
    assert "{" in msg and "<=" in msg


def test_semantic_coverage_detail_json_serializable():
    """① detail 必须能 json.dumps（此前 gold_objective 是 tuple 会抛 TypeError）。"""
    gold = get_gold_ir("knapsack-01")
    r = verifier.check_semantic_coverage(gold, _broken(gold))
    assert r.ok is False
    json.dumps(r.detail, ensure_ascii=False)  # 不抛异常即通过
    assert r.detail["diff"]["missing_constraints"]


def test_exec_error_head_includes_location():
    """④ 执行错误摘要应含文件:行号，而非只取异常末行。"""
    from nl2opt.pipeline.orchestrator import _exec_error_head

    class _SB:
        stderr = ("Traceback (most recent call last):\n"
                  '  File "/work/target.py", line 3, in <module>\n'
                  "    x4 = x4 + 1\n"
                  "NameError: name 'x4' is not defined\n")
        error = ""
        stdout = ""

    head = _exec_error_head(_SB())
    assert "NameError" in head
    assert "target.py:3" in head
