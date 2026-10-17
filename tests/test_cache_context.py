"""缓存键版本化与语义隔离的单元测试。

对应 scripts/run_tests.py 里的 CacheContext 断言。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest  # noqa: E402

from nl2opt.cache import Cache, CacheContext, build_cache_context, code_hash  # noqa: E402
from nl2opt.config import Config  # noqa: E402
from nl2opt.ir import IR, LinearExpr, Objective, Sense  # noqa: E402


def test_cache_context_fingerprint_changes_per_field():
    base = CacheContext(protocol_version=15, llm_model="deepseek-flash")
    assert base.fingerprint() == CacheContext(protocol_version=15, llm_model="deepseek-flash").fingerprint()
    assert base.fingerprint() != CacheContext(protocol_version=14, llm_model="deepseek-flash").fingerprint()
    assert base.fingerprint() != CacheContext(protocol_version=15, llm_model="deepseek-v4-pro").fingerprint()
    assert base.fingerprint() != CacheContext(protocol_version=15, llm_model="deepseek-flash", cache_version=2).fingerprint()
    assert base.fingerprint() != CacheContext(protocol_version=15, llm_model="deepseek-flash", backend="docker").fingerprint()


def test_cache_context_isolation_between_contexts():
    c1 = Cache(context=CacheContext(protocol_version=15, llm_model="m1"))
    c2 = Cache(context=CacheContext(protocol_version=15, llm_model="m2"))
    ck = code_hash("x = 1")
    c1.set_code(ck, {"code": "x = 1"})
    assert c1.get_code(ck) is not None
    assert c2.get_code(ck) is None  # 不同语义上下文不得交叉命中


def test_cache_no_context_falls_back_to_bare_key():
    c = Cache()
    ck = code_hash("x = 1")
    c.set_code(ck, {"code": "x = 1"})
    assert c.get_code(ck) is not None


def test_build_cache_context_captures_config():
    cfg = Config({
        "llm": {"model": "deepseek-flash", "temperature": 0.0,
                "reasoning_effort": "medium", "max_tokens": 65536},
        "executor": {"backend": "local", "timeout_seconds": 30},
        "cache": {"namespace": "exp1", "version": 3},
    })
    ctx = build_cache_context(cfg, 15)
    assert ctx.llm_model == "deepseek-flash"
    assert ctx.backend == "local"
    assert ctx.llm_sampling["temperature"] == 0.0
    assert ctx.llm_sampling["reasoning_effort"] == "medium"
    assert ctx.solver_config["timeout_seconds"] == 30
    assert ctx.cache_namespace == "exp1" and ctx.cache_version == 3
    assert ctx.prompt_fingerprint  # 提示词内容指纹非空

    cfg2 = Config({"llm": {"model": "deepseek-v4-pro"}, "executor": {"backend": "local"}})
    assert build_cache_context(cfg2, 15).fingerprint() != ctx.fingerprint()


# ============ prompt_fingerprint：完整 prompt-policy（P0.5） ============


def test_prompt_fingerprint_includes_triggers():
    """改触发词（trigger）必须改变 prompt 指纹——否则改「何时加载哪张卡」会静默复用旧 IR。"""
    import os
    from nl2opt.pipeline import prompts
    from nl2opt.pipeline.prompts import prompt_fingerprint

    os.environ.pop("NL2OPT_ALL_CARDS", None)  # 固定模式，隔离变量
    before = prompt_fingerprint()

    # 临时给 ratio 卡追加一条触发词，验证指纹变化
    card = prompts.CARDS["ratio"]
    original = list(card["triggers"])
    try:
        card["triggers"] = original + [r"some-new-trigger-pattern"]
        after = prompt_fingerprint()
    finally:
        card["triggers"] = original

    assert before != after, "改 trigger 未改变 prompt 指纹"


def test_prompt_fingerprint_includes_card_order_and_body():
    """卡片 body 已在旧实现覆盖；顺带确认 title/body/order 都在指纹里（改 body 必变）。"""
    import os
    from nl2opt.pipeline import prompts
    from nl2opt.pipeline.prompts import prompt_fingerprint

    os.environ.pop("NL2OPT_ALL_CARDS", None)
    before = prompt_fingerprint()

    card = prompts.CARDS["scheduling"]
    original = card["body"]
    try:
        card["body"] = original + "\n- 新增一条建模要点"
        after = prompt_fingerprint()
    finally:
        card["body"] = original

    assert before != after, "改卡片 body 未改变 prompt 指纹"


def test_prompt_fingerprint_changes_across_mode():
    """layered vs all-cards 模式切换必须改变 prompt 指纹（消融 A/B 两侧不得共享 IR）。"""
    import os
    from nl2opt.pipeline.prompts import prompt_fingerprint

    os.environ.pop("NL2OPT_ALL_CARDS", None)
    layered = prompt_fingerprint()

    os.environ["NL2OPT_ALL_CARDS"] = "1"
    try:
        all_cards = prompt_fingerprint()
    finally:
        os.environ.pop("NL2OPT_ALL_CARDS", None)

    assert layered != all_cards, "layered 与 all-cards 模式的 prompt 指纹相同"


def test_cross_mode_cache_miss():
    """跨模式：同一问题文本在不同模式下构建的 Cache 不得交叉命中（problem 级）。"""
    import os
    from nl2opt.cache import build_cache_context
    from nl2opt.config import Config

    cfg = Config({"llm": {"model": "deepseek-flash"}, "executor": {"backend": "local"}})

    os.environ.pop("NL2OPT_ALL_CARDS", None)
    ctx_layered = build_cache_context(cfg, 15)

    os.environ["NL2OPT_ALL_CARDS"] = "1"
    try:
        ctx_all = build_cache_context(cfg, 15)
    finally:
        os.environ.pop("NL2OPT_ALL_CARDS", None)

    assert ctx_layered.fingerprint() != ctx_all.fingerprint(), \
        "跨模式 CacheContext 指纹相同，会复用旧 IR"

    c_layered = Cache(context=ctx_layered)
    c_all = Cache(context=ctx_all)
    key = "some-problem-hash"
    bare_ir = IR(variables=[], objective=Objective(sense=Sense.minimize, expr=LinearExpr()))
    c_layered.set_problem(key, bare_ir)
    assert c_layered.get_problem(key) is not None
    assert c_all.get_problem(key) is None, "跨模式 problem 缓存交叉命中"
