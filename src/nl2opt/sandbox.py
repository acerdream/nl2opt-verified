"""Docker 沙箱执行器（宿主机侧）：在受限容器内执行生成代码。

安全边界（由 docker run 参数强制）：
- `--network=none`      无网络
- `--memory` / `--cpus` / `--pids-limit`  内存 / CPU / 进程数限制
- `--read-only` + `--tmpfs /tmp`          只读根文件系统，仅 /tmp 可写
- `-v <workdir>:/work:ro`                 目标代码只读挂载
- 容器内非 root（镜像内 useradd sandbox）
- 宿主侧墙钟超时：超时则 `docker kill` 容器

对外接口：
- `docker_available()`                Docker 守护进程是否可用
- `image_exists(cfg)`                 沙箱镜像是否已构建
- `run_in_sandbox(code, cfg, timeout)` 执行代码，返回 `SandboxResult`

`SandboxResult.to_solver_result()` 让编排器能以统一方式消费沙箱结果。
"""
from __future__ import annotations

import json
import subprocess
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from .solver import SolverResult

DEFAULT_IMAGE = "nl2opt-sandbox"
# 容器启动 + 镜像解压的宽限时间（不计入被测代码的执行预算）
_STARTUP_GRACE_SECONDS = 10.0

# 容器输出恒为 UTF-8；显式指定编码，避免宿主 locale（如 Windows GBK）解码失败，
# 否则读取线程会因 UnicodeDecodeError 崩溃并导致 stdout 丢失。
_TEXT_KWARGS = {"text": True, "encoding": "utf-8", "errors": "replace"}


@dataclass
class SandboxResult:
    """沙箱执行结果。"""

    exec_status: str  # ok | timeout | error
    solver_status: str = "error"  # optimal | infeasible | unbounded | error | timeout
    objective_value: float | None = None
    variables: dict[str, float] = field(default_factory=dict)
    stdout: str = ""
    stderr: str = ""
    runtime_ms: int = 0
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.exec_status == "ok"

    def to_solver_result(self) -> SolverResult:
        """转换为统一的 SolverResult，供 verifier / 编排器消费。"""
        return SolverResult(
            status=self.solver_status,
            objective_value=self.objective_value,
            variables=self.variables,
            message=self.error or f"sandbox:{self.exec_status}",
            raw={"exec_status": self.exec_status, "runtime_ms": self.runtime_ms, "sandbox": True},
        )


def docker_available() -> bool:
    """Docker 守护进程是否可用（不可用时调用方应给出可执行的指引）。"""
    try:
        proc = subprocess.run(
            ["docker", "version", "--format", "{{.Server.Version}}"],
            capture_output=True, **_TEXT_KWARGS, timeout=20,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return proc.returncode == 0


def image_exists(cfg) -> bool:
    """沙箱镜像是否已构建。"""
    try:
        proc = subprocess.run(
            ["docker", "image", "inspect", _image(cfg)],
            capture_output=True, **_TEXT_KWARGS, timeout=20,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return proc.returncode == 0


def preflight(cfg) -> str | None:
    """沙箱可用性预检：返回 None 表示可用，否则返回人类可读的原因。"""
    if not docker_available():
        return "Docker 不可用（守护进程未运行或当前环境禁止访问 Docker 管道）"
    if not image_exists(cfg):
        return f"沙箱镜像 {_image(cfg)} 未构建，请先运行：docker build -t {_image(cfg)} sandbox/"
    return None


def run_in_sandbox(code: str, cfg, timeout: float | None = None) -> SandboxResult:
    """在 Docker 沙箱中执行 `code`（应为可独立运行的求解脚本）。"""
    reason = preflight(cfg)
    if reason is not None:
        return SandboxResult(exec_status="error", error=reason)

    limit = float(timeout) if timeout else float(_setting(cfg, "timeout_seconds", 30))
    name = f"nl2opt-sbx-{uuid.uuid4().hex[:12]}"
    workdir = Path(tempfile.mkdtemp(prefix="nl2opt-sbx-"))
    try:
        (workdir / "target.py").write_text(code, encoding="utf-8")
        cmd = _build_command(cfg, workdir, name, limit)
        try:
            proc = subprocess.run(
                cmd, capture_output=True, **_TEXT_KWARGS,
                timeout=limit + _STARTUP_GRACE_SECONDS,
            )
        except subprocess.TimeoutExpired:
            _kill_container(name)
            return SandboxResult(
                exec_status="timeout",
                solver_status="timeout",
                error=f"沙箱执行超时（>{limit:g}s，已终止容器）",
            )
        except OSError as exc:
            return SandboxResult(exec_status="error", error=f"启动沙箱失败: {exc}")

        if proc.returncode != 0:
            return SandboxResult(
                exec_status="error",
                error=f"docker run 退出码 {proc.returncode}",
                stderr=(proc.stderr or "")[-2000:],
            )

        envelope = _parse_envelope(proc.stdout)
        if envelope is None:
            return SandboxResult(
                exec_status="error",
                error="无法解析沙箱输出信封",
                stdout=(proc.stdout or "")[-2000:],
                stderr=(proc.stderr or "")[-2000:],
            )
        return _to_sandbox_result(envelope)
    finally:
        # 尽力清理临时目录（Windows 上容器句柄可能短暂占用，忽略失败）
        try:
            for p in workdir.glob("*"):
                p.unlink(missing_ok=True)
            workdir.rmdir()
        except OSError:
            pass


# ---------------- 内部工具 ----------------

def _image(cfg) -> str:
    return str(_setting(cfg, "docker_image", DEFAULT_IMAGE))


def _setting(cfg, key: str, default):
    """从 cfg.executor 读取配置；cfg 无 executor 段时返回默认值。"""
    try:
        return cfg.executor.get(key, default)
    except AttributeError:
        return default


def _build_command(cfg, workdir: Path, name: str, limit: float) -> list[str]:
    mount = str(workdir.resolve()).replace("\\", "/")
    return [
        "docker", "run", "--rm",
        "--name", name,
        "--network=none",
        f"--memory={_setting(cfg, 'memory', '512m')}",
        f"--cpus={_setting(cfg, 'cpus', '1')}",
        f"--pids-limit={_setting(cfg, 'pids_limit', 128)}",
        "--read-only",
        "--tmpfs", "/tmp:rw,size=64m",
        "-e", f"NL2OPT_TIMEOUT={limit:g}",
        "-e", "HOME=/tmp",
        "-v", f"{mount}:/work:ro",
        _image(cfg),
    ]


def _kill_container(name: str) -> None:
    """超时后尽力终止容器（--rm 会在退出后自动清理）。"""
    try:
        subprocess.run(["docker", "kill", name], capture_output=True, **_TEXT_KWARGS, timeout=20)
    except (OSError, subprocess.SubprocessError):
        pass


def _parse_envelope(stdout: str) -> dict | None:
    """取 stdout 最后一个非空行解析为 JSON 信封（docker 可能掺入其它输出）。"""
    for line in reversed((stdout or "").splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and "exec_status" in data:
            return data
    return None


def _to_sandbox_result(env: dict) -> SandboxResult:
    solver = env.get("solver") or {}
    exec_status = str(env.get("exec_status", "error"))
    if exec_status == "timeout":
        solver_status = "timeout"
    else:
        solver_status = str(solver.get("status", "error"))
    variables = solver.get("variables") or {}
    return SandboxResult(
        exec_status=exec_status,
        solver_status=solver_status,
        objective_value=solver.get("objective_value"),
        variables={k: float(v) for k, v in variables.items() if v is not None},
        stdout=env.get("stdout", "") or "",
        stderr=env.get("stderr", "") or "",
        runtime_ms=int(env.get("runtime_ms", 0) or 0),
        error=str(env.get("error", "") or ""),
    )
