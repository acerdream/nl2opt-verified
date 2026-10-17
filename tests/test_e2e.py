"""端到端确定性链路测试：内置样例全部通过（无需 LLM）。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nl2opt.config import load_config  # noqa: E402
from nl2opt.dataset import load_builtin  # noqa: E402
from nl2opt.pipeline import run_dataset  # noqa: E402


def test_builtin_dataset_all_pass():
    cfg = load_config()
    ds = load_builtin()
    results = run_dataset(ds, llm=None, cfg=cfg)
    assert len(results) == len(ds)
    failures = [(r.sample_id, r.errors) for r in results if not r.success]
    assert not failures, f"以下样例未通过: {failures}"
