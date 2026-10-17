"""Docker 沙箱端到端验证：结果对拍 + 资源限制实证 + L1 信号。

用法：python scripts/check_sandbox.py
前置：docker build -t nl2opt-sandbox sandbox/

验证项：
1. 交叉对拍   生成代码在沙箱内（PuLP/CBC）求解 vs 本地（scipy/HiGHS），目标值需一致；
              沙箱解还需通过 verifier 的独立可行性核验（含整数性/边界）。
2. 无网络     容器内 urlopen 必须失败（--network=none）。
3. 超时       死循环脚本被墙钟超时终止。
4. 内存限制   超额分配被 OOM 终止（--memory）。
5. L1 信号    未定义变量回传 NameError → reflect 归因为 L1（代码层可修复）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Windows 控制台可能是 GBK：避免符号导致的 UnicodeEncodeError
try:
    sys.stdout.reconfigure(errors="replace") # type: ignore
except (AttributeError, ValueError):
    pass

from nl2opt import codegen, sandbox, verifier  # noqa: E402
from nl2opt.config import load_config  # noqa: E402
from nl2opt.gold_ir import get_gold_ir  # noqa: E402
from nl2opt.pipeline.reflect import reflect  # noqa: E402
from nl2opt.solver import solve  # noqa: E402

# 有数值最优解的样例（含 0-1、连续、整数、等式约束）
CROSS_CHECK = ["knapsack-01", "lp-01", "diet-01", "assignment-01", "integer-01", "multi-index-01"]
# 状态类样例：沙箱求解状态需与本地一致
STATUS_CHECK = ["infeasible-01", "unbounded-01"]

_TOL = 1e-4


def main() -> int:
    cfg = load_config()
    reason = sandbox.preflight(cfg)
    if reason is not None:
        print(f"[FAIL] 沙箱不可用：{reason}")
        return 1

    print(f"镜像: {cfg.executor.get('docker_image')} | "
          f"内存: {cfg.executor.get('memory')} | CPU: {cfg.executor.get('cpus')} | "
          f"超时: {cfg.executor.get('timeout_seconds')}s\n")

    failures: list[str] = []

    def report(name: str, ok: bool, detail: str) -> None:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
        if not ok:
            failures.append(name)

    # ---------- 1) 交叉对拍 ----------
    print("--- 1) 交叉对拍（沙箱 PuLP/CBC vs 本地 scipy/HiGHS）---")
    for sid in CROSS_CHECK:
        ir = get_gold_ir(sid)
        local = solve(ir)
        sb = sandbox.run_in_sandbox(codegen.generate_code(ir), cfg)
        both = sb.ok and local.success and sb.objective_value is not None
        gap = abs(sb.objective_value - local.objective_value) if both else None # pyright: ignore[reportOperatorIssue]
        agreed = both and gap is not None and gap <= _TOL
        # 沙箱解必须通过独立验证（约束/边界/整数性）
        feas = verifier.check_feasibility(ir, sb.to_solver_result())
        report(
            f"对拍 {sid}",
            agreed and feas.ok,
            f"local={_fmt(local.objective_value)} sandbox={_fmt(sb.objective_value)} "
            f"gap={_fmt(gap)} 可行={feas.ok} 耗时={sb.runtime_ms}ms"
            + ("" if not sb.error else f" err={sb.error}"),
        )

    for sid in STATUS_CHECK:
        ir = get_gold_ir(sid)
        local = solve(ir)
        sb = sandbox.run_in_sandbox(codegen.generate_code(ir), cfg)
        report(
            f"状态 {sid}",
            sb.solver_status == local.status,
            f"local={local.status} sandbox={sb.solver_status}",
        )

    # ---------- 2) 无网络 ----------
    print("\n--- 2) 隔离性：无网络 ---")
    net_code = (
        "import json, urllib.request\n"
        "try:\n"
        "    urllib.request.urlopen('http://example.com', timeout=5)\n"
        "    print('NETWORK_REACHABLE=1')\n"
        "except Exception as exc:\n"
        "    print('NETWORK_BLOCKED=' + type(exc).__name__)\n"
        "print('__NL2OPT_RESULT__' + json.dumps("
        "{'status': 'error', 'objective_value': None, 'variables': {}}))\n"
    )
    sb = sandbox.run_in_sandbox(net_code, cfg, timeout=20)
    blocked = "NETWORK_BLOCKED=" in sb.stdout
    report("无网络", blocked and "NETWORK_REACHABLE" not in sb.stdout,
           _first_line(sb.stdout, "NETWORK_") or sb.error)

    # ---------- 3) 超时 ----------
    print("\n--- 3) 资源限制：墙钟超时 ---")
    sb = sandbox.run_in_sandbox("while True:\n    pass\n", cfg, timeout=4)
    report("超时终止", sb.exec_status == "timeout",
           f"exec_status={sb.exec_status} ({sb.error or '无错误信息'})")

    # ---------- 4) 内存限制 ----------
    print("\n--- 4) 资源限制：内存上限 ---")
    mem_code = "big = []\nwhile True:\n    big.append(bytearray(50 * 1024 * 1024))\n"
    sb = sandbox.run_in_sandbox(mem_code, cfg, timeout=25)
    report("内存上限", sb.exec_status == "error",
           f"exec_status={sb.exec_status} err={sb.error or _last_line(sb.stderr)}")

    # ---------- 5) L1 代码错误信号 ----------
    print("\n--- 5) L1 代码层错误信号 ---")
    sb = sandbox.run_in_sandbox("x4 = undefined_name + 1\nprint(x4)\n", cfg, timeout=10)
    err_text = f"沙箱执行失败（{sb.exec_status}）：{_last_line(sb.stderr) or sb.error}"
    refl = reflect("测试", "{}", err_text, None)  # 无 LLM → 启发式归因
    report("L1 归因", sb.exec_status == "error" and refl.category == "L1",
           f"stderr={_last_line(sb.stderr)} → reflect={refl.category}({refl.label})")

    print("\n" + "=" * 60)
    if failures:
        print(f"[FAIL] {len(failures)} 项失败: {failures}")
        return 1
    print("[OK] 全部通过：沙箱执行、隔离性、资源限制与 L1 信号均符合预期")
    return 0


def _fmt(v) -> str:
    return "-" if v is None else f"{v:.6f}"


def _first_line(text: str, prefix: str) -> str:
    for line in (text or "").splitlines():
        if line.strip().startswith(prefix):
            return line.strip()
    return ""


def _last_line(text: str) -> str:
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    return lines[-1] if lines else ""


if __name__ == "__main__":
    raise SystemExit(main())
