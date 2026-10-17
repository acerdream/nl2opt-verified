"""配置加载：YAML 为主，环境变量覆盖（用于注入 API Key 等敏感信息）。"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yaml"


class Config:
    """轻量配置对象：既可按属性访问，也可按 dict 访问。

    支持 `cfg.llm.base_url` 与 `cfg["llm"]["base_url"]` 两种风格。
    """

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    def __getattr__(self, key: str) -> Any:
        try:
            return self._wrap(self._data[key])
        except KeyError as exc:  # pragma: no cover - 防御性
            raise AttributeError(key) from exc

    def __getitem__(self, key: str) -> Any:
        return self._wrap(self._data[key])

    def get(self, key: str, default: Any = None) -> Any:
        return self._wrap(self._data.get(key, default))

    def __contains__(self, key: str) -> bool:
        """支持 `"k" in cfg.section`。

        必须显式实现：否则 Python 会退回旧的序列迭代协议，尝试 `cfg[0]`、`cfg[1]`…，
        以 `KeyError: 0` 的形式崩溃（而非返回 False）——这是个容易被忽视的陷阱。
        """
        return key in self._data

    def __iter__(self):
        """支持遍历配置键（与 `__contains__` 同理，避免退回序列协议）。"""
        return iter(self._data)

    def keys(self):
        return self._data.keys()

    def items(self):
        return self._data.items()

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:  # pragma: no cover - 便于调试
        return f"Config({self._data!r})"

    def _wrap(self, value: Any) -> Any:
        if isinstance(value, dict):
            return Config(value)
        return value

    def to_dict(self) -> dict[str, Any]:
        return self._data


def _env_overrides() -> dict[str, Any]:
    """将关键环境变量映射为配置覆盖项。"""
    overrides: dict[str, Any] = {}
    mapping = {
        "DEEPSEEK_BASE_URL": ("llm", "base_url"),
        "DEEPSEEK_MODEL": ("llm", "model"),
        "DEEPSEEK_API_KEY": ("llm", "api_key"),
    }
    for env_key, (section, field) in mapping.items():
        if env_key in os.environ:
            overrides.setdefault(section, {})[field] = os.environ[env_key]
    return overrides


def load_config(path: str | Path | None = None) -> Config:
    """加载配置：YAML 文件 + 环境变量覆盖。

    未指定路径时按顺序查找：
    1. 环境变量 NL2OPT_CONFIG
    2. 项目根目录 config.yaml
    """
    config_path = Path(path) if path else Path(os.environ.get("NL2OPT_CONFIG", DEFAULT_CONFIG_PATH))
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with config_path.open("r", encoding="utf-8") as fh:
        data: dict[str, Any] = yaml.safe_load(fh) or {}

    # 环境变量覆盖（浅层合并，仅覆盖已映射的标量项）
    overrides = _env_overrides()
    for section, fields in overrides.items():
        data.setdefault(section, {})
        data[section].update(fields)

    return Config(data)


def resolve_api_key(cfg: Config) -> str | None:
    """优先读环境变量，其次读 .env 文件，最后读 YAML 中的明文 api_key（不推荐）。"""
    env_name = cfg.llm.get("api_key_env", "DEEPSEEK_API_KEY")
    key = os.environ.get(env_name)
    if key:
        return key

    # 尽力读取项目根 .env（不引入 python-dotenv 依赖）
    env_file = Path(__file__).resolve().parents[2] / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            if k.strip() == env_name and v.strip():
                return v.strip().strip('"').strip("'")

    # 兜底：YAML 明文（不推荐，仅便于本地调试）
    return cfg.llm.get("api_key", None)
