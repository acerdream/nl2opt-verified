"""反思环节：失败归因 + 生成修复反馈。

按「错误层级回退」原则，把失败归类到 L1~L4，供编排器路由：
- L1 代码错误：语法错 / 运行时错（沙箱执行生成代码后才有意义，当前为占位）
- L2 建模错误：约束系数/方向/类型错、目标函数错（IR 结构对，但内容错）
- L3 理解错误：漏变量 / 漏约束 / 变量引用不合法 / IR 校验失败（IR 结构就错）
- L4 不可自动修复：无法归因、或反复失败、需人工介入

层级越小越接近"表面修复"，越大越需要"回退重新理解"。
"""
from __future__ import annotations

from dataclasses import dataclass

from ..llm import LLMMessage, LLMProvider, complete_structured

# 层级常量
L1 = "L1"  # 代码错误
L2 = "L2"  # 建模错误
L3 = "L3"  # 理解错误
L4 = "L4"  # 不可自动修复（转人工）

LAYER_LABELS = {
    L1: "代码错误",
    L2: "建模错误",
    L3: "理解错误",
    L4: "不可自动修复",
}


class ReflectionParseError(RuntimeError):
    """反思输出无法解析出有效错误层级（响应完整但 JSON 非法且无关键词命中）。

    由编排器捕获后路由到 L4（转人工），**绝不盲目落 L2**——把「解析失败」误归为
    「建模错误」会让闭环在错误的方向上反复重写，空烧 token。
    """

# 代码层错误标记：命中即归为 L1（只改代码，不动 IR）。
# 注意：仅列「代码本身出错」的信号，环境不可用（Docker/镜像）不在此列，避免误路由。
_L1_MARKERS = (
    "语法", "SyntaxError", "IndentationError", "TabError",
    "运行时", "RuntimeError",
    "NameError", "TypeError", "AttributeError", "ImportError", "ModuleNotFoundError",
    "ZeroDivisionError", "KeyError", "IndexError", "NotImplementedError",
    "RecursionError", "MemoryError",
)

REFLECT_PROMPT = """你是一名数学建模诊断专家。下面是自然语言问题、上一轮生成的 IR、以及验证失败的原因。
请诊断失败根因，并归类到下列错误层级之一，给出"如何修复"的具体建议。

错误层级定义：
- L1 代码错误：生成的求解代码有语法错误或运行时异常（与数学模型无关）。
- L2 建模错误：IR 结构正确，但内容错（约束系数/方向写错、目标函数错、变量类型错、rhs 错）。
- L3 理解错误：IR 结构本身错（漏了变量、漏了约束、引用了未声明的变量、变量无法正确实例化）。
- L4 不可自动修复：问题信息不足、或无明确可修复点，需人工介入。

## 自然语言问题
{nl_text}

## 上一轮 IR（JSON）
{ir_json}

## 验证失败原因
{errors}

请只输出一行 JSON：{{"layer": "L2", "advice": "具体修复点，如：约束 c1 的 rhs 应为 100 而非 90"}}
不要输出其它文字。"""


@dataclass
class Reflection:
    """反思结果。"""

    category: str  # L1 / L2 / L3 / L4
    advice: str
    label: str = ""  # 人类可读标签，如 "建模错误"

    def __post_init__(self) -> None:
        if not self.label:
            self.label = LAYER_LABELS.get(self.category, self.category)


def reflect(
    nl_text: str,
    ir_json: str,
    errors: str,
    llm: LLMProvider | None,
) -> Reflection:
    """生成反思。若无 LLM（未配置 key），退化为基于错误文本的启发式归因。

    有 LLM 时：截断/空输出经共享响应保护层拦截（抛带 stage 的 LLMError）；
    响应完整但 JSON 非法且无关键词命中 → 抛 `ReflectionParseError`（由编排器转 L4），
    **不**盲目默认 L2。
    """
    if llm is not None:
        prompt = REFLECT_PROMPT.format(nl_text=nl_text, ir_json=ir_json, errors=errors)
        messages = [
            LLMMessage(role="system", content="你是数学建模诊断专家。"),
            LLMMessage(role="user", content=prompt),
        ]
        resp = complete_structured(llm, messages, stage="reflect")
        return _parse_reflection(resp.text)
    return _heuristic_reflection(errors)


def _parse_reflection(text: str) -> Reflection:
    """解析 LLM 输出，优先提取 JSON 里的 layer；失败则回退到文本关键词匹配。

    关键安全性质：**任何一条路径都不允许「什么都不匹配就落 L2」**。
    L2 是「建模错误」——若反思输出根本解析不出层级，把它默认成 L2 会让闭环
    在错误方向上反复重写。故：
    - JSON 解析成功但 `layer` 非法，或 JSON 解析失败且无关键词命中 → 抛
      `ReflectionParseError`（编排器据此转 L4 人工）。
    - 关键词命中（兼容旧格式的散文输出）仍属**合法回退**，不抛。
    """
    from ..llm import extract_json

    # 尝试按 JSON 解析
    try:
        data = extract_json(text)
        if isinstance(data, dict):
            layer = data.get("layer", "")
            advice = data.get("advice", text.strip())
            if layer in LAYER_LABELS:
                return Reflection(category=layer, advice=advice)
    except Exception:
        pass

    # 回退：文本关键词匹配（兼容旧格式）
    for kw, layer in [
        ("L1", L1), ("L2", L2), ("L3", L3), ("L4", L4),
        ("代码错误", L1), ("语法", L1), ("运行时", L1),
        ("建模错误", L2),
        ("理解错误", L3),
        ("不可自动修复", L4), ("人工", L4),
    ]:
        if kw in text:
            return Reflection(category=layer, advice=text.strip())

    # 什么都没匹配到 → 显式解析失败，绝不默认 L2
    raise ReflectionParseError(
        f"反思输出无法解析出有效错误层级（非 JSON 且无关键词命中）: {text[:200]!r}"
    )


def _heuristic_reflection(errors: str) -> Reflection:
    """无 LLM 时的启发式归因，基于错误文本判断层级。"""
    # L1 代码错误：语法 / 运行时 / Python 异常名
    # （沙箱执行生成代码失败时，stderr 末行即异常类型，据此与「环境不可用」区分）
    if any(marker in errors for marker in _L1_MARKERS):
        return Reflection(category=L1, advice=errors)
    # L4 不可自动修复：求解超时（资源限制，重写 IR 无效，需调大超时或人工介入）
    if "timeout" in errors or "超时" in errors:
        return Reflection(category=L4, advice=f"求解超时：{errors}")
    # L3 理解错误：IR 结构问题（漏变量/漏约束/引用未声明/解析失败）
    if ("JSON" in errors or "解析" in errors or "未声明变量" in errors
            or "漏" in errors or "引用" in errors):
        return Reflection(category=L3, advice=errors)
    # 求解状态类（不可行/无界）→ 建模错误（约束写错导致）
    if "不可行" in errors or "infeasible" in errors or "无界" in errors or "unbounded" in errors:
        return Reflection(category=L2, advice=errors)
    # 目标值不符 / 约束违反 → 建模错误
    if "目标值" in errors or "违反" in errors or "可行" in errors or "状态" in errors:
        return Reflection(category=L2, advice=errors)
    # 求解未成功 → 建模错误（默认）
    if "求解未成功" in errors:
        return Reflection(category=L2, advice=errors)
    # 兜底：不可自动修复
    return Reflection(category=L4, advice=errors)
