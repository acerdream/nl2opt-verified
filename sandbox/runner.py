"""沙箱内执行器（容器侧，由宿主 sandbox.py 通过 docker run 调用）。

职责：
1. 执行只读挂载的 /work/target.py（子进程 + 墙钟超时）；
2. 捕获其 stdout/stderr，解析 `__NL2OPT_RESULT__<json>` 结构化结果；
3. 向容器 stdout 打印**单行 JSON 信封**，供宿主解析。

约定：
- 目标脚本路径：/work/target.py
- 超时：环境变量 NL2OPT_TIMEOUT（秒，默认 25）
- 信封字段：exec_status(ok|timeout|error) / solver / stdout / stderr / runtime_ms / error

注意：本文件在容器内运行，不 import 本项目任何模块。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

TARGET = "/work/target.py"
MARKER = "__NL2OPT_RESULT__"
_MAX_CAPTURE = 8000  # stdout/stderr 回传上限（字符）


def _tail(text: str) -> str:
    """截断长输出，保留尾部（错误信息通常在尾部）。"""
    if text is None:
        return ""
    if len(text) <= _MAX_CAPTURE:
        return text
    return "...[截断]...\n" + text[-_MAX_CAPTURE:]


def _emit(envelope: dict) -> None:
    sys.stdout.write(json.dumps(envelope, ensure_ascii=False))
    sys.stdout.write("\n")
    sys.stdout.flush()


def _parse_solver(stdout: str) -> dict | None:
    """从 stdout 中提取最后一个 __NL2OPT_RESULT__ 标记行并解析。"""
    found = None
    for line in stdout.splitlines():
        line = line.strip()
        if line.startswith(MARKER):
            found = line[len(MARKER):]
    if not found:
        return None
    try:
        return json.loads(found)
    except json.JSONDecodeError:
        return None


def main() -> int:
    timeout = float(os.environ.get("NL2OPT_TIMEOUT", "25"))
    t0 = time.time()

    if not os.path.exists(TARGET):
        _emit({
            "exec_status": "error",
            "error": f"目标脚本不存在: {TARGET}",
            "solver": None,
            "stdout": "",
            "stderr": "",
            "runtime_ms": 0,
        })
        return 0

    try:
        proc = subprocess.run(
            [sys.executable, "-I", TARGET],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
        runtime_ms = int((time.time() - t0) * 1000)
        stdout = _tail(proc.stdout)
        stderr = _tail(proc.stderr)

        if proc.returncode != 0:
            _emit({
                "exec_status": "error",
                "error": f"目标脚本退出码 {proc.returncode}",
                "solver": None,
                "stdout": stdout,
                "stderr": stderr,
                "runtime_ms": runtime_ms,
            })
            return 0

        solver = _parse_solver(proc.stdout)
        _emit({
            "exec_status": "ok" if solver is not None else "error",
            "error": "" if solver is not None else "未找到 __NL2OPT_RESULT__ 结构化结果",
            "solver": solver,
            "stdout": stdout,
            "stderr": stderr,
            "runtime_ms": runtime_ms,
        })
        return 0

    except subprocess.TimeoutExpired as exc:
        runtime_ms = int((time.time() - t0) * 1000)
        _emit({
            "exec_status": "timeout",
            "error": f"目标脚本执行超时（>{timeout}s）",
            "solver": None,
            "stdout": _tail(_as_text(exc.stdout)),
            "stderr": _tail(_as_text(exc.stderr)),
            "runtime_ms": runtime_ms,
        })
        return 0


def _as_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
