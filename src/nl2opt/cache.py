"""三级哈希缓存：问题哈希 / IR 哈希 / 代码哈希，减少重复 token 与计算。

- 问题哈希：对「预处理后的自然语言问题」做 sha256，命中则直接复用之前
  LLM 理解出的 IR，跳过 understand() 的 token 消耗（最大收益）。
- IR 哈希：对 IR 的规范化 JSON 做 sha256，命中则复用之前的求解/验证结果。
- 代码哈希：对生成代码做 sha256，命中则复用执行结果（沙箱启用后有意义）。

持久化：JSON 文件（默认 artifacts/cache.json），可跨进程复用。

## 缓存键版本化与语义隔离（P1）

三级 key 以前只哈希「内容本身」（问题文本 / IR / 代码），**没纳入产生该内容的上下文**：
IR schema 版本、prompt/cards 版本、LLM model/temperature/reasoning、solver 库/后端版本、
timeout 及其它 solver 选项、codegen 协议版本。后果：改 prompt、换模型、改 codegen 或
求解器参数后，可能**静默命中旧结果**（旧 IR 或旧解被当成新配置下的产物）。

修法：引入 `CacheContext`，把上述语义上下文合并成一个确定性 `fingerprint`，
**参与所有三级 key**（作为 key 前缀）。任何一项变化 → 指纹变化 → 旧 key 天然 miss，
不会误命中。另提供 config 级 `cache.namespace` / `cache.version` 作为**人工逃生阀**：
遇到本机制没覆盖的语义变化时，手动递增 version 即可整体失效。
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .ir import IR


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def problem_hash(text: str) -> str:
    """问题文本（应为预处理后的规范文本）→ 哈希。"""
    return _sha256(text.strip())


def ir_hash(ir: IR) -> str:
    """IR → 规范化 JSON → 哈希。"""
    # sort_keys 保证字段顺序无关；默认 model_dump 已稳定
    return _sha256(json.dumps(ir.model_dump(), ensure_ascii=False, sort_keys=True))


def code_hash(code: str) -> str:
    """生成代码 → 哈希。"""
    return _sha256(code.strip())


@dataclass(frozen=True)
class CacheContext:
    """缓存键的语义上下文：任何一项变化都使旧缓存失效。

    字段覆盖了「内容哈希」没纳入的、会影响产物正确性的上下文：
    - protocol_version：IR schema / 验证器判据 / 反思重写路由 / 影响 LLM 输出形态的
      prompt / codegen 协议的**主版本**（单一事实来源，见 `nl2opt.__init__`）；
    - prompt_fingerprint：核心 + 全部卡片的**内容哈希**（任何 prompt 文本改动，
      即使漏 bump PROTOCOL_VERSION，也自动失效 problem 级缓存）；
    - llm_model / llm_sampling：换模型、改 temperature/reasoning/max_tokens 都会影响
      LLM 产出，须隔离；
    - backend / solver_config：本地 scipy 与 docker PuLP/CBC 语义可能不同，timeout 等
      求解器选项影响解；
    - cache_namespace / cache_version：人工逃生阀（本机制未覆盖的语义变化时手动递增）。
    """

    protocol_version: int
    prompt_fingerprint: str = ""
    llm_model: str = ""
    llm_sampling: dict[str, Any] = field(default_factory=dict)
    backend: str = ""
    solver_config: dict[str, Any] = field(default_factory=dict)
    cache_namespace: str = "default"
    cache_version: int = 1

    def fingerprint(self) -> str:
        """把上下文序列化为规范 JSON 再 sha256，得到稳定指纹。"""
        payload = {
            "cache_namespace": self.cache_namespace,
            "cache_version": self.cache_version,
            "protocol_version": self.protocol_version,
            "prompt_fingerprint": self.prompt_fingerprint,
            "llm_model": self.llm_model,
            "llm_sampling": self.llm_sampling,
            "backend": self.backend,
            "solver_config": self.solver_config,
        }
        return _sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def build_cache_context(cfg: Any, protocol_version: int) -> CacheContext:
    """从配置 + 主协议版本构造 CacheContext。

    cfg 为 Config（支持属性/下标访问）。提示词指纹从 prompts 模块取（避免循环导入，
    此处延迟导入）。未配置的字段取空/默认，保证确定性。
    """
    from .pipeline.prompts import prompt_fingerprint

    llm = getattr(cfg, "llm", None) or {}
    executor = getattr(cfg, "executor", None) or {}
    cache_cfg = getattr(cfg, "cache", None) or {}

    def _g(container: Any, key: str, default: Any = None) -> Any:
        if isinstance(container, dict):
            return container.get(key, default)
        return getattr(container, key, default) if container is not None else default

    llm_sampling = {
        "temperature": _g(llm, "temperature", 0.0),
        "reasoning_effort": _g(llm, "reasoning_effort", None),
        "max_tokens": _g(llm, "max_tokens", None),
    }
    # 只纳入影响「求解结果」的求解器选项；docker_image 等纯环境项不参与
    solver_config = {
        "timeout_seconds": _g(executor, "timeout_seconds", None),
        "memory": _g(executor, "memory", None),
        "cpus": _g(executor, "cpus", None),
        "pids_limit": _g(executor, "pids_limit", None),
    }

    return CacheContext(
        protocol_version=protocol_version,
        prompt_fingerprint=prompt_fingerprint(),
        llm_model=_g(llm, "model", ""),
        llm_sampling=llm_sampling,
        backend=_g(executor, "backend", ""),
        solver_config=solver_config,
        cache_namespace=str(_g(cache_cfg, "namespace", "default")),
        cache_version=int(_g(cache_cfg, "version", 1)),
    )


def _namespaced(context_fp: str | None, base_key: str) -> str:
    """把上下文指纹作为前缀并入 key；无上下文时退化为裸 key（向后兼容）。"""
    if not context_fp:
        return base_key
    # 取指纹前 12 位即可区分，且保持 key 可读
    return f"{context_fp[:12]}:{base_key}"


class Cache:
    """三级哈希缓存，带 JSON 持久化。

    `context` 为 CacheContext（可选）：传入时其 `fingerprint()` 会作为前缀并入
    所有三级 key，使「上下文变化」自动隔离缓存（见模块 docstring 的版本化说明）。
    不传则退化为旧的裸 key 行为（向后兼容）。
    """

    def __init__(self, path: str | Path | None = None, context: CacheContext | None = None) -> None:
        self._path = Path(path) if path else None
        self._ctx_fp = context.fingerprint() if context is not None else ""
        self._data: dict[str, dict[str, Any]] = {"problem": {}, "ir": {}, "code": {}}
        if self._path and self._path.exists():
            self._load()

    def _k(self, base_key: str) -> str:
        return _namespaced(self._ctx_fp, base_key)

    # ---- 持久化 ----
    def _load(self) -> None:
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            for level in ("problem", "ir", "code"):
                if level in data and isinstance(data[level], dict):
                    self._data[level] = data[level]
        except Exception:
            # 缓存损坏则当作空缓存，不阻断主流程
            self._data = {"problem": {}, "ir": {}, "code": {}}

    def save(self) -> None:
        if not self._path:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(self._data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # ---- 三级 get/set ----
    def get_problem(self, key: str) -> dict[str, Any] | None:
        return self._data["problem"].get(self._k(key))

    def set_problem(self, key: str, ir: IR) -> None:
        self._data["problem"][self._k(key)] = {"ir": ir.model_dump()}

    def get_ir(self, key: str) -> dict[str, Any] | None:
        return self._data["ir"].get(self._k(key))

    def set_ir(self, key: str, result: dict[str, Any]) -> None:
        self._data["ir"][self._k(key)] = result

    def get_code(self, key: str) -> dict[str, Any] | None:
        return self._data["code"].get(self._k(key))

    def set_code(self, key: str, result: dict[str, Any]) -> None:
        self._data["code"][self._k(key)] = result

    # ---- 统计 ----
    def stats(self) -> dict[str, int]:
        return {k: len(v) for k, v in self._data.items()}
