"""L1 代码层修复：沙箱执行失败（语法错误 / 运行时异常 / API 误用）时修正「生成代码」。

与 L2 的区别：**IR 不变，只改代码本身**。

- 有 LLM：把「原始问题 + 当前 IR + 当前代码 + 执行错误」交给 LLM，产出修正后的完整代码；
- 无 LLM：确定性回退——按 IR 重新生成代码（记为 `regenerate`）。

产出代码必须通过 `compile()` 语法校验，否则视为修复失败（交由编排器升级到 L2/L4）。
"""
from __future__ import annotations

from ..ir import IR
from ..llm import LLMMessage, LLMProvider

REPAIR_PROMPT = """你是一名 Python 求解代码修复专家。下面是一个线性/整数规划的求解脚本，它在受限沙箱中执行时失败了。

请**只修正代码层面的错误**（语法错误、变量名未定义、API 误用、缩进问题、输出格式不符等）。

硬性要求：
1. 只输出修正后的**完整 Python 代码**，不要任何解释文字，不要 Markdown 代码块围栏。
2. 建模内容不得改动：变量、目标函数、约束的数学含义必须与下面的 IR 完全一致——你只修代码。
3. 必须用 PuLP 建模（from pulp import ...），并确保脚本可独立运行。
4. 最后必须输出一行结构化结果（供沙箱解析），格式严格如下：
   print('__NL2OPT_RESULT__' + json.dumps({{'status': _st, 'objective_value': _obj, 'variables': _vars}}))
   其中 status 取值为 optimal / infeasible / unbounded / error；variables 的键必须与 IR 的变量实例名一致。

## 原始问题
{nl_text}

## 当前 IR（JSON）
{ir_json}

## 当前代码
{code}

## 执行错误
{error}

现在只输出修正后的完整 Python 代码："""


def repair_code(
    nl_text: str,
    code: str,
    error: str,
    ir: IR,
    llm: LLMProvider | None,
) -> tuple[str, str]:
    """修复生成代码，返回 (new_code, how)；how ∈ {"llm", "regenerate"}。

    无 LLM 或 LLM 产出不可用时，回退为按 IR 重新生成（确定性投影）。
    """
    if llm is not None:
        prompt = REPAIR_PROMPT.format(
            nl_text=nl_text,
            ir_json=ir.model_dump_json(indent=2),
            code=code,
            error=error,
        )
        messages = [
            LLMMessage(role="system", content="你是 Python 求解代码修复专家。"),
            LLMMessage(role="user", content=prompt),
        ]
        try:
            resp = llm.complete(messages)
            candidate = _extract_code(resp.text)
        except Exception:
            candidate = ""
        if candidate.strip() and _compiles(candidate):
            return candidate, "llm"

    # 确定性回退：按 IR 重新生成
    from .. import codegen

    return codegen.generate_code(ir), "regenerate"


def _extract_code(text: str) -> str:
    """从 LLM 输出中提取代码：容忍 Markdown 代码块围栏与前后说明文字。"""
    if not text:
        return ""
    t = text.strip()
    if "```" in t:
        # 取第一个围栏块的内容
        parts = t.split("```")
        # parts[0] 是围栏前的说明，parts[1] 是围栏内内容（可能带 python 语言标注）
        if len(parts) >= 3:
            body = parts[1]
            first_nl = body.find("\n")
            if first_nl != -1 and body[:first_nl].strip().lower() in ("python", "py", "python3"):
                body = body[first_nl + 1:]
            return body.strip()
    return t


def _compiles(code: str) -> bool:
    """语法校验：能通过 compile() 才认为代码可用于执行。"""
    try:
        compile(code, "<l1-repaired>", "exec")
        return True
    except SyntaxError:
        return False
