"""预处理：文本清洗、单位归一、语言检测。

在「理解」之前对自然语言问题做轻量归一化，降低 LLM 对格式噪声的敏感度，
同时产出 language 元信息供下游参考。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# 单位归一映射：常见别名 → 规范写法（仅做轻量归一，不改变数值语义）
_UNIT_ALIASES = [
    (r"\bhrs?\b", "hours"),
    (r"\bhr\b", "hour"),
    (r"\bmins?\b", "minutes"),
    (r"\bmin\b", "minute"),
    (r"\bkgs?\b", "kilograms"),
    (r"\bkg\b", "kilogram"),
    (r"\bgms?\b", "grams"),
    (r"\bg\b", "gram"),
    (r"\blbs?\b", "pounds"),
    (r"\bsecs?\b", "seconds"),
    (r"\bsec\b", "second"),
]

# 多余空白 / 标点归一
_MULTI_SPACE = re.compile(r"[ \t]+")
_CJK_SPACE = re.compile(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])")  # 中文之间空格

_CJK_CHARS = re.compile(r"[\u4e00-\u9fff]")


@dataclass
class Preprocessed:
    """预处理结果。"""

    text: str
    language: str  # zh | en | mixed | unknown
    changed: list[str] = field(default_factory=list)  # 应用的归一动作（可观测）


def detect_language(text: str) -> str:
    """粗粒度语言检测：中文 / 英文 / 混合 / 未知。"""
    cjk = len(_CJK_CHARS.findall(text))
    # 粗略英文判定：拉丁字母词数
    latin_words = len(re.findall(r"[A-Za-z]{2,}", text))
    if cjk > 0 and latin_words > 0:
        return "mixed"
    if cjk > 0:
        return "zh"
    if latin_words > 0:
        return "en"
    return "unknown"


def clean_text(text: str) -> tuple[str, list[str]]:
    """文本清洗：压缩空白、去首尾、归一中文间空格。返回 (清洗后文本, 变更记录)。"""
    changed: list[str] = []
    t = text.strip()
    if t != text:
        changed.append("strip")
    t2 = _MULTI_SPACE.sub(" ", t)
    if t2 != t:
        changed.append("collapse_whitespace")
    t = t2
    t3 = _CJK_SPACE.sub("", t)
    if t3 != t:
        changed.append("remove_cjk_inner_space")
        t = t3
    return t, changed


def normalize_units(text: str) -> tuple[str, list[str]]:
    """单位别名归一。返回 (归一后文本, 变更记录)。"""
    changed: list[str] = []
    t = text
    for pattern, repl in _UNIT_ALIASES:
        t2, n = re.subn(pattern, repl, t)
        if n > 0:
            changed.append(f"normalize_unit:{repl}")
            t = t2
    return t, changed


def preprocess(text: str) -> Preprocessed:
    """完整预处理流水线：清洗 → 单位归一 → 语言检测。"""
    text, changed = clean_text(text)
    text, unit_changes = normalize_units(text)
    changed.extend(unit_changes)
    return Preprocessed(
        text=text,
        language=detect_language(text),
        changed=changed,
    )
