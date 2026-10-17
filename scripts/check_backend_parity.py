"""跨后端边界样本对拍套件：本地 HiGHS 与 docker/CBC 必须对同一 IR 给出一致结果。

**为什么需要它（一族同根缺陷）**：`codegen`（docker 后端）与 `solver.engine`
（local 后端）是**两条独立实现**，对同一 IR 的语义处理一旦不一致，就是静默错误。
历史三次踩坑，全部只能靠「主动构造边界形态的 IR + 两后端对拍」才抓得到——
自然分布的 LLM 输出几乎不产生这些形态：

| 样本 | 缺陷 | 版本 |
|---|---|---|
| 目标常数 | engine 丢弃 `expr.constant`，codegen 输出它 → 目标值不同 | v8 |
| 重复项 | engine 赋值覆盖（取最后一项），codegen 累加 → 模型不同 | v11 |
| 非法变量名 | codegen 把 IR 名当 Python 名 → SyntaxError 落 L1 | v13 |

新增边界样本时，只需往 `SAMPLES` 表里加一个构造器，无需改主流程。

零 LLM 成本（样例内嵌 IR 走 `extra.ir`）。docker 后端需更宽权限（Docker 管道）。
用法：
    python scripts/check_backend_parity.py            # 只跑本地
    python scripts/check_backend_parity.py --docker   # 两后端都跑并逐样本比对
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from nl2opt.gold_ir import get_gold_ir  # noqa: E402


# ---- 边界样本构造器：返回 {id, ir, expected_objective, required_vars, forbidden_vars} ----

def _const_offset():
    """目标常数：knapsack-01 最优 220，人为 +7 → 期望 227。"""
    ir = get_gold_ir("knapsack-01")
    ir.objective.expr.constant = 7.0
    return {
        "id": "const-offset",
        "ir": ir.model_dump(),
        "expected_objective": 227.0,
        "required_vars": {"x1", "x2", "x3"},
        "forbidden_vars": set(),
    }


def _duplicate_terms():
    """重复项：maximize 2x+3x s.t. x+4x<=10 → 规范化 5x<=10 → x=2，目标 10。
    （若 engine 赋值覆盖会得 7.5，与 codegen 累加的 10 分叉。）"""
    return {
        "id": "duplicate-terms",
        "ir": {
            "variables": [{"name": "x", "var_type": "continuous", "lower": 0.0, "upper": 10.0}],
            "objective": {"sense": "maximize", "expr": {
                "terms": [{"var": "x", "coef": 2.0}, {"var": "x", "coef": 3.0}],
                "constant": 0.0}},
            "constraints": [{"name": "c1", "sense": "<=", "lhs": {
                "terms": [{"var": "x", "coef": 1.0}, {"var": "x", "coef": 4.0}],
                "constant": 0.0}, "rhs": 10.0}],
        },
        "expected_objective": 10.0,
        "required_vars": {"x"},
        "forbidden_vars": set(),
    }


def _non_identifier_names():
    """非法 Python 标识符变量名：maximize 2*(x-y) + 3*start_J1，start_J1<=3 → 目标 23。
    （旧 codegen 会生成 `x-y = ...` SyntaxError；且结果 key 必须是原始名 x-y，非安全名 _v0。）"""
    return {
        "id": "non-identifier-names",
        "ir": {
            "variables": [
                {"name": "x-y", "var_type": "continuous", "lower": 0.0, "upper": None},
                {"name": "start", "var_type": "integer", "index_set": ["J1"],
                 "lower": 0.0, "upper": 3.0},
            ],
            "objective": {"sense": "maximize", "expr": {
                "terms": [{"var": "x-y", "coef": 2.0}, {"var": "start_J1", "coef": 3.0}],
                "constant": 0.0}},
            "constraints": [
                {"name": "c1", "sense": "<=", "lhs": {
                    "terms": [{"var": "x-y", "coef": 1.0}, {"var": "start_J1", "coef": 1.0}],
                    "constant": 0.0}, "rhs": 10.0},
            ],
        },
        "expected_objective": 23.0,
        "required_vars": {"x-y", "start_J1"},
        "forbidden_vars": {"_v0", "_v1"},          # 安全名不得泄漏进结果
    }


def _free_lower_bound():
    """`lower=None`（无下界）的变量：minimize c + k s.t. c >= -7, k >= -5 → 最优 -12。

    抓的缺陷（v22 修复）：旧 codegen 在 `lower is None` 且类型**非 continuous** 时
    强制 `lowBound=0`，而 engine 用 `-inf` → 同一 IR 两后端得到**不同模型**：
    本地给 -12（c=-7, k=-5），docker 给 -7（k 被锁在 0）。
    此前 `Variable.lower` 默认 0.0，该分支极少触发；默认值改为 None 后成为常见情形。
    """
    return {
        "id": "free-lower-bound",
        "ir": {
            "variables": [
                {"name": "c", "var_type": "continuous", "lower": None, "upper": None},
                {"name": "k", "var_type": "integer", "lower": None, "upper": None},
            ],
            "objective": {"sense": "minimize", "expr": {
                "terms": [{"var": "c", "coef": 1.0}, {"var": "k", "coef": 1.0}],
                "constant": 0.0}},
            "constraints": [
                {"name": "c_floor", "sense": ">=", "lhs": {
                    "terms": [{"var": "c", "coef": 1.0}], "constant": 0.0}, "rhs": -7.0},
                {"name": "k_floor", "sense": ">=", "lhs": {
                    "terms": [{"var": "k", "coef": 1.0}], "constant": 0.0}, "rhs": -5.0},
            ],
        },
        "expected_objective": -12.0,
        "required_vars": {"c", "k"},
        "forbidden_vars": set(),
    }


SAMPLES = [_const_offset, _duplicate_terms, _non_identifier_names, _free_lower_bound]


# ---- 执行 ----

def _run_eval(data: Path, out: Path, backend: str) -> dict | None:
    out.unlink(missing_ok=True)
    cmd = [sys.executable, "-m", "nl2opt", "eval", "--data", str(data),
           "--no-cache", "--out", str(out)]
    if backend == "docker":
        cmd += ["--backend", "docker"]
    subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                   # 子进程输出 UTF-8，Windows 默认 GBK 解码 → 显式指定，否则 reader 抛错
                   encoding="utf-8", errors="replace",
                   env={**os.environ, "PYTHONPATH": str(ROOT / "src"),
                        "PYTHONIOENCODING": "utf-8"})
    if not out.exists():
        return None
    rows = json.loads(out.read_text(encoding="utf-8"))
    r = rows[0]
    return {"status": r.get("status"), "objective_value": r.get("objective_value"),
            "variables": r.get("variables"), "success": r.get("success")}


def _verify(spec: dict, res: dict | None) -> bool:
    """单个后端结果是否满足该样本的期望。"""
    if res is None or res["objective_value"] is None:
        return False
    if abs(res["objective_value"] - spec["expected_objective"]) > 1e-6:
        return False
    vars_ = res["variables"] or {}
    if not spec["required_vars"].issubset(vars_):
        return False
    if spec["forbidden_vars"] & vars_.keys():
        return False
    return True


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="跨后端边界样本对拍套件")
    ap.add_argument("--docker", action="store_true", help="同时跑 docker 后端并逐样本比对")
    args = ap.parse_args(argv)

    specs = [mk() for mk in SAMPLES]
    all_ok = True
    local_res: dict[str, dict | None] = {}
    docker_res: dict[str, dict | None] = {}

    for spec in specs:
        p = ROOT / "data" / f"_parity_{spec['id']}.jsonl"
        p.write_text(json.dumps({
            "id": spec["id"], "problem_type": "milp",
            "nl_text": f"边界样本对拍：{spec['id']}",
            "expected": {"objective_value": spec["expected_objective"]},
            "trusted_reference": True, "extra": {"ir": spec["ir"]},
        }, ensure_ascii=False) + "\n", encoding="utf-8")

        out_local = ROOT / "artifacts" / f"_parity_{spec['id']}_local.json"
        lr = _run_eval(p, out_local, "local")
        local_res[spec["id"]] = lr

        dr = None
        if args.docker:
            out_docker = ROOT / "artifacts" / f"_parity_{spec['id']}_docker.json"
            dr = _run_eval(p, out_docker, "docker")
            docker_res[spec["id"]] = dr

        p.unlink(missing_ok=True)
        out_local.unlink(missing_ok=True)
        (ROOT / "artifacts" / f"_parity_{spec['id']}_docker.json").unlink(missing_ok=True)

    # 汇总表
    print(f"{'样本':22s} {'期望':>10s} {'本地 HiGHS':>22s} {'docker/CBC':>22s} 判定")
    print("-" * 90)
    for spec in specs:
        lr, dr = local_res[spec["id"]], docker_res.get(spec["id"])
        l_ok = _verify(spec, lr)
        d_ok = _verify(spec, dr) if dr is not None else None
        same = (lr is not None and dr is not None
                and lr["objective_value"] == dr["objective_value"]
                and lr["variables"] == dr["variables"]
                and lr["status"] == dr["status"])
        verdict = ""
        if args.docker:
            verdict = "✓ 一致" if (l_ok and d_ok and same) else "✗ 不一致"
        else:
            verdict = "✓" if l_ok else "✗"
        all_ok = all_ok and l_ok and (not args.docker or (d_ok and same))
        lv = f"{lr['objective_value']}" if lr else "None"
        dv = f"{dr['objective_value']}" if dr else ("—" if args.docker else "（未跑）")
        print(f"{spec['id']:22s} {spec['expected_objective']:>10g} {lv:>22s} {dv:>22s} {verdict}")

    # 额外断言：非法名样本的结果 key 必须是原始 IR 名（已在 _verify 的 forbidden_vars 覆盖）
    print()
    print("结论:", "全部通过 ✓" if all_ok else "存在失败 ✗")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
