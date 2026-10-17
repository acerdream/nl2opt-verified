"""数据集：内置样例集 + 统一加载器。

数据格式（JSONL，每行一条）：
{
  "id": "knapsack-01",
  "problem_type": "milp",
  "nl_text": "……自然语言描述……",
  "expected": {
    "objective_value": 220.0,
    "variables": {"x1": 1.0, "x2": 0.0}   // 可选，键为变量实例名
  }
}

内置样例集位于 data/builtin/samples.jsonl，全部自带参考答案，
用于在无网络、无 LLM 的情况下跑通「建模→生成→执行→验证」的确定性链路。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from pydantic import BaseModel, Field

BUILTIN_FILE = Path(__file__).resolve().parents[2] / "data" / "builtin" / "samples.jsonl"


class Sample(BaseModel):
    """单条样例（训练/评测通用）。"""

    id: str
    nl_text: str
    problem_type: str = Field(default="milp", description="lp | milp（仅供参考）")
    expected: dict[str, Any] = Field(default_factory=dict)
    extra: dict[str, Any] = Field(default_factory=dict)
    trusted_reference: bool = Field(
        default=True,
        description="参考答案是否可信。内置样例(有黄金IR)为 True；外部数据集(如NL4OPT)为 False，"
        "此时若解可行但目标值不符，容忍为'疑似标注错误'而非触发重写。",
    )


class Dataset:
    """样例集合，可切片、可迭代。"""

    def __init__(self, samples: list[Sample], name: str = "") -> None:
        self.samples = samples
        self.name = name

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self) -> Iterator[Sample]:
        return iter(self.samples)

    def __getitem__(self, idx: int | slice) -> Sample | "Dataset":
        if isinstance(idx, slice):
            return Dataset(self.samples[idx], name=self.name)
        return self.samples[idx]


def load_builtin() -> Dataset:
    """加载内置样例集（含参考答案）。"""
    return load_jsonl(BUILTIN_FILE, name="builtin")


def load_jsonl(path: str | Path, name: str = "") -> Dataset:
    """从 JSONL 文件加载样例集。"""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"数据集不存在: {p}")
    samples: list[Sample] = []
    with p.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                data = json.loads(line)
                samples.append(Sample.model_validate(data))
            except Exception as exc:
                raise ValueError(f"{p} 第 {lineno} 行解析失败: {exc}") from exc
    return Dataset(samples, name=name or str(p))


def load_any(path: str | Path | None) -> Dataset:
    """智能加载：None → 内置样例集；否则按 JSONL 加载。"""
    if path is None:
        return load_builtin()
    return load_jsonl(path)
