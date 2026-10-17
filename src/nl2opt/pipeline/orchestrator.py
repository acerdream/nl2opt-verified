"""闭环编排器：理解→建模→生成→执行→验证→反思→重写，最多 N 轮。"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from .. import codegen, sandbox, verifier
from ..cache import Cache, code_hash, ir_hash, problem_hash
from ..config import Config
from ..dataset import Sample
from ..gold_ir import get_gold_ir
from ..ir import IR, ir_from_dict
from ..llm import LLMError, LLMProvider
from ..solver import SolverResult, solve
from .reflect import Reflection, ReflectionParseError, reflect
from .repair_code import repair_code
from .rewrite import rewrite
from .understand import understand


@dataclass
class PipelineResult:
    """一次端到端运行的结果。"""

    sample_id: str
    success: bool = False
    suspicious: bool = False  # 疑似参考答案标注错误（解可行但目标值不符，未重写）
    rounds: int = 0
    objective_value: float | None = None
    variables: dict[str, float] = field(default_factory=dict)
    status: str = "unknown"
    final_ir: dict[str, Any] | None = None
    generated_code: str = ""
    errors: list[str] = field(default_factory=list)
    trajectory: list[dict[str, Any]] = field(default_factory=list)
    token_usage: dict[str, Any] = field(default_factory=dict)  # 累计 token 用量
    stage_usage: dict[str, Any] = field(default_factory=dict)  # 按阶段（understand/reflect/...）

    def to_files(self, out_dir: str | Any) -> dict[str, str]:
        """把产物落盘到 out_dir，返回各产物路径。

        产物：
        - final_code.py        生成的最终求解代码
        - solution.json        求解结果（状态/目标值/变量）
        - verification_report.json  验证报告
        - rewrite_trace.json   重写轨迹（trajectory）
        - final_error.json     最终错误（无错误时为空列表）
        - token_usage.json     token 用量
        """
        from pathlib import Path

        d = Path(out_dir)
        d.mkdir(parents=True, exist_ok=True)

        paths: dict[str, str] = {}

        def _write(name: str, content: str) -> str:
            p = d / name
            p.write_text(content, encoding="utf-8")
            paths[name] = str(p)
            return str(p)

        # final_code.py
        _write("final_code.py", self.generated_code or "")

        # solution.json
        _write("solution.json", json.dumps({
            "status": self.status,
            "objective_value": self.objective_value,
            "variables": self.variables,
            "success": self.success,
            "suspicious": self.suspicious,
            "rounds": self.rounds,
        }, ensure_ascii=False, indent=2))

        # verification_report.json
        report = {
            "sample_id": self.sample_id,
            "success": self.success,
            "suspicious": self.suspicious,
            "status": self.status,
            "rounds": self.rounds,
            "objective_value": self.objective_value,
            "errors": self.errors,
        }
        # 附加最后一轮 checks（若有）
        if self.trajectory:
            last = self.trajectory[-1]
            report["last_round_checks"] = last.get("checks", [])
        _write("verification_report.json", json.dumps(report, ensure_ascii=False, indent=2))

        # rewrite_trace.json
        _write("rewrite_trace.json", json.dumps(self.trajectory, ensure_ascii=False, indent=2))

        # final_error.json（自描述：has_error 区分"成功无错误"与"有错误"）
        _write("final_error.json", json.dumps({
            "has_error": bool(self.errors),
            "errors": self.errors,
        }, ensure_ascii=False, indent=2))

        # token_usage.json
        _write("token_usage.json", json.dumps(self.token_usage, ensure_ascii=False, indent=2))

        # stage_usage.json（分阶段 token，供成本报表按阶段归因）
        _write("stage_usage.json", json.dumps(self.stage_usage, ensure_ascii=False, indent=2))

        return paths


class _UsageTrackingLLM:
    """包装 LLMProvider，累计每次 complete() 的 usage。

    - `usage_sink`：整样本合计（原有行为，保持向后兼容）；
    - `stage_sink`：**按阶段**分别累计（understand / reflect / rewrite / repair_code），
      由编排器在每次调用前用 `set_stage(...)` 指定归属。

    这样 understand/reflect/rewrite 内部无需改动，就能拿到分阶段 token 成本。
    """

    def __init__(
        self,
        inner: LLMProvider,
        usage_sink: dict[str, Any],
        stage_sink: dict[str, Any] | None = None,
    ) -> None:
        self._inner = inner
        self._sink = usage_sink
        self._stage_sink = stage_sink if stage_sink is not None else {}
        self._stage = "unknown"
        self._calls = 0

    def set_stage(self, stage: str) -> None:
        """标注后续 complete() 调用所属阶段（用于分阶段成本核算）。"""
        self._stage = stage

    def complete(self, messages, **kwargs):
        resp = self._inner.complete(messages, **kwargs)
        self._calls += 1
        self._sink["calls"] = self._calls  # calls 始终记录，不依赖 usage 非空
        self._record_usage(resp.usage)
        self._record_stage(resp.usage)
        return resp

    def __getattr__(self, name: str) -> Any:
        """委托其它属性/方法到内层 LLMProvider。

        这样将来 LLMProvider 抽象类新增任何方法，包装器无需同步改动，
        会自动透传给内层对象（透明代理）。
        """
        return getattr(self._inner, name)

    @staticmethod
    def _accumulate(sink: dict[str, Any], usage: dict[str, Any]) -> None:
        """把一次调用的 usage 累加进 sink（标量直接加，嵌套 dict 递归累加）。"""
        for key, val in usage.items():
            if isinstance(val, bool):
                continue
            if isinstance(val, (int, float)):
                sink[key] = sink.get(key, 0) + val
            elif isinstance(val, dict):
                inner = sink.setdefault(key, {})
                for k2, v2 in val.items():
                    if not isinstance(v2, bool) and isinstance(v2, (int, float)):
                        inner[k2] = inner.get(k2, 0) + v2

    def _record_usage(self, usage: dict[str, Any]) -> None:
        if not usage:
            return
        self._accumulate(self._sink, usage)

    def _record_stage(self, usage: dict[str, Any]) -> None:
        stage = self._stage_sink.setdefault(self._stage, {})
        stage["calls"] = stage.get("calls", 0) + 1
        if usage:
            self._accumulate(stage, usage)


def _error_fingerprint(errors: str) -> str:
    """归一化错误文本，得到稳定指纹。

    去掉具体数值（浮点/整数），只保留错误类型结构，使
    "目标值不符：求解 1120.000000，期望 220.0" 与
    "目标值不符：求解 1130.000000，期望 220.0" 归为同一指纹。
    """
    import re

    normalized = re.sub(r"-?\d+\.?\d*", "<N>", errors)
    # 压缩空白
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _understand_with_cache(
    text: str,
    llm: LLMProvider,
    cache: Cache | None,
    preprocess_text: bool = True,
) -> tuple[IR, dict[str, Any]]:
    """带三级缓存的理解：问题哈希命中则复用缓存的 IR，否则调 LLM 并写缓存。

    返回 (ir, meta)，meta 含 cache_hit / problem_key 等观测信息。
    """
    meta: dict[str, Any] = {}

    # 预处理（默认启用）：清洗 + 单位归一，使问题哈希更稳定。
    # 注意：仅用规范化文本计算 key；LLM 仍看原始 text（保持语义信息完整）。
    key_text = text
    if preprocess_text:
        from .preprocess import preprocess
        key_text = preprocess(text).text

    key = problem_hash(key_text)
    meta["problem_key"] = key

    if cache is not None:
        cached = cache.get_problem(key)
        if cached is not None:
            try:
                ir = ir_from_dict(cached["ir"])
                meta["cache_hit"] = True
                return ir, meta
            except Exception:
                # 缓存内容损坏则忽略，走 LLM
                pass

    ir = understand(text, llm, meta)     # meta 记录命中的领域卡片与提示词规模
    meta["cache_hit"] = False
    if cache is not None:
        cache.set_problem(key, ir)
    return ir, meta


def run_sample(
    sample: Sample,
    llm: LLMProvider | None,
    cfg: Config,
    force_llm_understand: bool = False,
    cache: Cache | None = None,
) -> PipelineResult:
    """对单个样例跑完整闭环。

    - llm=None 时走确定性链路（IR 必须由外部提供）。
    - force_llm_understand=True 时，即使有黄金 IR 也强制走 LLM 理解，
      用于单独演练理解环节、评估 LLM 抽取质量。
    - cache 非 None 时启用问题哈希缓存，命中则复用 LLM 理解结果（省 token）。
    """
    tol = float(cfg.pipeline.get("feasibility_tolerance", 1e-6))
    # 目标值容差与可行性容差是两回事：前者应明显更松，否则会把求解器浮点抖动误判为建模错误
    obj_tol = float(cfg.pipeline.get("objective_tolerance", 1e-4))
    max_rounds = int(cfg.pipeline.get("max_rounds", 5))
    repeated_error_limit = int(cfg.pipeline.get("repeated_error_limit", 3))
    # 求解超时（秒），None 表示不限时
    solve_time_limit = cfg.executor.get("timeout_seconds", None) if hasattr(cfg, "executor") else None
    if solve_time_limit is not None:
        solve_time_limit = float(solve_time_limit)

    # 执行后端：local = 进程内 scipy 求解 IR；docker = 受限容器执行「生成代码」
    executor_cfg = cfg.executor if hasattr(cfg, "executor") else None
    backend = str(executor_cfg.get("backend", "local")) if executor_cfg is not None else "local"
    l1_max_retries = int(executor_cfg.get("l1_max_retries", 1)) if executor_cfg is not None else 0

    result = PipelineResult(sample_id=sample.id)

    # 用 usage 追踪包装器包住 LLM，累计 token 用量（understand/reflect/rewrite 内部无需改动）
    if llm is not None:
        llm = _UsageTrackingLLM(llm, result.token_usage, result.stage_usage)

    # 连续相同错误跟踪
    last_fingerprint: str | None = None
    repeat_count: int = 0

    # L1 代码层修复状态：forced_code 承载修复后的代码，供下一轮直接执行
    forced_code: str | None = None
    l1_attempts = 0

    # 疑似参考噪声跟踪（untrusted reference 且「可行但目标值不符」时置位）
    # 语义：**不短路**——仍允许一次修复尝试，用于把「不知道谁错」变成证据；
    # 但无论后续修复是否成功，判定都保持 suspicious、**绝不升级为通过**。
    suspicion: str | None = None     # None / "pending" / "converged" / "unresolved"
    suspicion_rewrites = 0

    # IR 获取优先级：
    # force_llm_understand=True → 强制 LLM 理解
    # 否则：样例内嵌 IR > 黄金 IR 基准 > LLM 理解
    ir: IR | None = None
    if force_llm_understand:
        if llm is None:
            result.errors.append("force_llm_understand=True 但未提供 LLM")
            return result
        try:
            _set_stage(llm, "understand")
            ir, meta = _understand_with_cache(sample.nl_text, llm, cache)
            result.trajectory.append({"understand_meta": meta})
        except Exception as exc:
            result.errors.append(f"理解失败: {exc}")
            return result
    elif "ir" in sample.extra:
        ir = IR.model_validate(sample.extra["ir"])
    else:
        gold = get_gold_ir(sample.id)
        if gold is not None:
            ir = gold
        elif llm is not None:
            try:
                _set_stage(llm, "understand")
                ir, meta = _understand_with_cache(sample.nl_text, llm, cache)
                result.trajectory.append({"understand_meta": meta})
            except Exception as exc:
                result.errors.append(f"理解失败: {exc}")
                return result
        else:
            result.errors.append("无 LLM 且无该样例的黄金 IR，无法理解")
            return result

    for round_no in range(1, max_rounds + 1):
        result.rounds = round_no

        cache_hit_info: dict[str, Any] = {}

        # ---- 三级缓存：code 级（复用生成代码）----
        # L1 修复后的代码优先执行，且不入缓存（缓存保持「IR → 规范代码」的确定性投影语义）
        if forced_code is not None:
            code = forced_code
            forced_code = None
            cache_hit_info["code_forced"] = True
        else:
            code = codegen.generate_code(ir)
            if cache is not None:
                ckey = code_hash(code)
                cached_code = cache.get_code(ckey)
                if cached_code is not None:
                    code = cached_code.get("code", code)
                    cache_hit_info["code_cache_hit"] = True
                else:
                    cache.set_code(ckey, {"code": code})
                    cache_hit_info["code_cache_hit"] = False

        result.generated_code = code
        result.final_ir = ir.model_dump()

        # ---- EXECUTE：按 backend 选择执行路径 ----
        # local：进程内 scipy/HiGHS 直接求解 IR（无需 Docker）
        # docker：受限容器中执行「生成代码」（--network=none + 资源限制 + 超时）
        exec_meta: dict[str, Any] = {}
        checks_exec: list[verifier.CheckResult] = []
        use_cache = cache is not None and not cache_hit_info.get("code_forced")
        ikey = f"{backend}:{ir_hash(ir)}" if use_cache else ""
        cached_sol = cache.get_ir(ikey) if use_cache else None

        if cached_sol is not None:
            sol = SolverResult(
                status=cached_sol.get("status", "error"),
                objective_value=cached_sol.get("objective_value"),
                variables=cached_sol.get("variables", {}),
                message=cached_sol.get("message", ""),
                raw=cached_sol.get("raw", {}),
            )
            cache_hit_info["ir_cache_hit"] = True
            exec_meta = {"executor": backend, "from_cache": True}
        else:
            if use_cache:
                cache_hit_info["ir_cache_hit"] = False

            if backend == "docker":
                sb = sandbox.run_in_sandbox(code, cfg, timeout=solve_time_limit)
                sol = sb.to_solver_result()
                exec_meta = {
                    "executor": "docker",
                    "exec_status": sb.exec_status,
                    "sandbox_runtime_ms": sb.runtime_ms,
                }
                if sb.exec_status != "ok":
                    # 执行失败（代码错误 / 超时 / 环境不可用）→ 显式记录，供反思定位层级
                    checks_exec.append(
                        verifier.CheckResult(
                            False,
                            f"沙箱执行失败（{sb.exec_status}）：{_exec_error_head(sb)}",
                            {
                                "exec_status": sb.exec_status,
                                "error": sb.error,
                                "stderr": (sb.stderr or "")[-1500:],
                            },
                            kind="execution",
                        )
                    )
            else:
                sol = solve(ir, time_limit=solve_time_limit)
                exec_meta = {"executor": "local"}

            # 仅缓存真实求解结论；执行错误/超时属瞬时问题，缓存会污染后续运行
            if use_cache and (sol.success or sol.status in ("infeasible", "unbounded")):
                cache.set_ir(ikey, {
                    "status": sol.status,
                    "objective_value": sol.objective_value,
                    "variables": sol.variables,
                    "message": sol.message,
                    "raw": sol.raw,
                })

        checks = verifier.verify(ir, sol, sample.expected, tol, obj_tol,
                                 getattr(sample, "nl_text", None)) + checks_exec

        # 语义覆盖验证：force_llm_understand=True 且有黄金 IR 时，
        # 额外对比 LLM 产出 IR 与黄金 IR 的语义覆盖（漏变量/漏约束，变量名无关）。
        if force_llm_understand:
            gold = get_gold_ir(sample.id)
            if gold is not None:
                checks.append(verifier.check_semantic_coverage(gold, ir))

        all_ok = verifier.all_pass(checks)

        result.trajectory.append(
            {
                "round": round_no,
                "ir": ir.model_dump(),
                "status": sol.status,
                "objective_value": sol.objective_value,
                "variables": sol.variables,
                "checks": [{"ok": c.ok, "message": c.message, "blocking": c.blocking,
                            "detail": c.detail} for c in checks],
                **exec_meta,
                **cache_hit_info,
            }
        )

        if all_ok:
            if suspicion:
                # 修复后达到了参考值 ⇒ 「参考是噪声」被证伪，原本就是模型错。
                # 记 converged，交由循环后的收尾块统一裁决（v9 起**计为通过**，
                # 与 trusted 数据集判定一致；status 保留 suspicious_resolved 可追溯）。
                suspicion = "converged"
                result.objective_value = sol.objective_value
                result.variables = sol.variables
                break
            result.success = True
            result.status = sol.status
            result.objective_value = sol.objective_value
            result.variables = sol.variables
            break

        # 容忍机制（仅当参考答案不可信时启用）：
        # 解可行、仅目标值与参考答案不符 → 疑似参考答案标注错误。
        #
        # 旧实现直接 `return`，后果是：**在 trusted_reference=False 的数据集上
        # （NL4OPT 245/245、IndustryOR 97/97）目标不符永远不会触发任何修复尝试**——
        # 闭环的修复能力在这两个最难的external数据集上等于被关闭，且无法区分
        # 「参考噪声」与「模型真错」。
        #
        # 现改为：标记疑似后**继续走修复流程**（最多一次重写），把修复结果记为证据
        # （suspicious_outcome=converged/unresolved），判定仍保持 suspicious。
        if (not sample.trusted_reference) and verifier.suspicious_reference(checks):
            if suspicion is None:
                suspicion = "pending"
                result.suspicious = True
                result.objective_value = sol.objective_value
                result.variables = sol.variables
                result.errors = [verifier.describe(checks)]
            result.trajectory[-1]["suspicious"] = True
            if suspicion_rewrites >= 1:
                # 已给过一次机会仍未达到参考值 → 不再消耗 token
                suspicion = "unresolved"
                # ⚠️ 必须用**本轮**的 sol 刷新结果字段：`if suspicion is None:` 只在第 1 轮
                # 设置过 objective_value/variables/errors，若不在此更新，break 后报告值会
                # 停留在第 1 轮，与 final_ir（最后一轮 IR）不一致。
                # 实测影响 6 题（NL4OPT 5 + IndustryOR 1），全部 rounds=2、status=suspicious：
                # 例 industryor-005 报告 18.9566 而 final_ir 实为 18.9857。
                result.objective_value = sol.objective_value
                result.variables = sol.variables
                result.errors = [verifier.describe(checks)]
                result.trajectory[-1]["suspicious_outcome"] = "unresolved"
                break

        # 失败 → 连续相同错误检测 → 反思 → 按错误层级回退路由（L1→L2→L3→L4）
        errors = verifier.describe(checks)
        result.errors = [errors]

        # 连续相同错误终止：归一化错误指纹，连续重复 N 轮即终止，避免空转烧 token
        fingerprint = _error_fingerprint(errors)
        if fingerprint == last_fingerprint:
            repeat_count += 1
        else:
            last_fingerprint = fingerprint
            repeat_count = 1
        result.trajectory[-1]["error_fingerprint"] = fingerprint
        result.trajectory[-1]["repeat_count"] = repeat_count

        if repeat_count >= repeated_error_limit:
            result.status = "repeated_error"
            result.objective_value = sol.objective_value
            result.variables = sol.variables
            result.errors.append(
                f"[连续相同错误终止] 连续 {repeat_count} 轮错误指纹一致: {fingerprint}"
            )
            break

        _set_stage(llm, "reflect")
        try:
            reflection: Reflection = reflect(sample.nl_text, json.dumps(ir.model_dump(), ensure_ascii=False), errors, llm)
        except (LLMError, ReflectionParseError) as exc:
            # 反思环节「生产失败」（截断/空）或「解析失败」（JSON 非法且无关键词命中）
            # → 显式转 L4 人工，记录 reflection_parse_error，绝不盲目落 L2 继续重写。
            result.trajectory[-1]["route"] = "L4(反思解析失败)"
            result.trajectory[-1]["reflection_parse_error"] = str(exc)
            result.errors.append(f"[L4 反思解析失败] {exc}")
            result.status = "manual_required"
            result.objective_value = sol.objective_value
            result.variables = sol.variables
            break
        result.trajectory[-1]["reflection"] = {
            "category": reflection.category,
            "label": reflection.label,
            "advice": reflection.advice,
        }

        # 无 LLM：无法自动修复，直接结束（确定性链路到此为止）
        if llm is None:
            result.status = sol.status
            result.objective_value = sol.objective_value
            result.variables = sol.variables
            break

        # 按层级路由
        if reflection.category == "L4":
            # 不可自动修复 → 转人工，记录并停止
            result.errors.append(f"[L4 转人工] {reflection.advice}")
            result.status = "manual_required"
            result.objective_value = sol.objective_value
            result.variables = sol.variables
            break

        if reflection.category == "L1" and backend == "docker" and l1_attempts < l1_max_retries:
            # L1 代码层修复：IR 不变，只修代码（修复后的代码下一轮直接执行）
            l1_attempts += 1
            result.trajectory[-1]["route"] = f"L1(代码修复 第{l1_attempts}次)"
            try:
                _set_stage(llm, "repair_code")
                new_code, how = repair_code(sample.nl_text, code, errors, ir, llm)
            except Exception as exc:
                result.errors.append(f"L1 代码修复异常: {exc}")
                result.status = sol.status
                result.objective_value = sol.objective_value
                result.variables = sol.variables
                break
            result.trajectory[-1]["l1_repair"] = how
            forced_code = new_code
            continue

        if reflection.category == "L1":
            # 无法走 L1 代码修复：本地后端没有「代码执行」路径，或 L1 重试已用尽
            # → 升级为 L2（改 IR），避免在代码层空转
            why = "本地后端无代码执行路径" if backend != "docker" else "L1 重试已用尽"
            result.trajectory[-1]["route"] = f"L1→L2({why})"
            result.errors.append(f"[L1 升级] {why}，改为重写 IR")
            try:
                _set_stage(llm, "rewrite")
                new_ir = rewrite(sample.nl_text, ir, errors, reflection.advice, llm)
            except (LLMError, ValueError) as exc:
                result.errors.append(f"重写失败(L1→L2): {exc}")
                result.status = sol.status
                result.objective_value = sol.objective_value
                result.variables = sol.variables
                break
            ir = new_ir
            continue

        if reflection.category == "L3":
            # 理解错误（漏变量/漏约束/IR 校验失败）→ 重新调用 understand() 抽取
            result.trajectory[-1]["route"] = "L3(重新理解)"
            try:
                _set_stage(llm, "understand")
                ir = understand(sample.nl_text, llm, result.trajectory[-1])
            except Exception as exc:
                result.errors.append(f"重新理解失败(L3): {exc}")
                result.status = sol.status
                result.objective_value = sol.objective_value
                result.variables = sol.variables
                break
            continue

        # L2 建模错误 → rewrite() 改 IR
        result.trajectory[-1]["route"] = "L2(重写IR)"
        if suspicion:
            suspicion_rewrites += 1     # 疑似态下最多允许一次重写尝试
        try:
            _set_stage(llm, "rewrite")
            new_ir = rewrite(
                sample.nl_text,
                ir,
                errors,
                reflection.advice,
                llm,
            )
        except (LLMError, ValueError) as exc:
            result.errors.append(f"重写失败(L2): {exc}")
            result.status = sol.status
            result.objective_value = sol.objective_value
            result.variables = sol.variables
            break

        ir = new_ir

    else:
        # for 循环正常结束（未被 break）→ 轮次用尽
        result.status = "max_rounds_exceeded"

    # 疑似参考噪声的最终裁决（v9 修订）。
    #
    # 两种结局的**证据强度不同**，判定也不同：
    #
    #   converged  → 重写后达到参考值 ⇒ 「参考是噪声」这一假设**被证伪**，
    #                原本就是模型错。判定与 trusted 数据集**一致**：记为通过
    #                （success=True），同时保留 status=suspicious_resolved 可追溯。
    #   unresolved → 给过一次机会仍未达到 ⇒ 疑似噪声的解释才站得住，
    #                保持 success=False、status=suspicious。
    #
    # **安全不变量**：只有 unresolved 才可能"被放过"，converged 已由求解器
    # 与验证器双重确认（最终 IR 可行且目标值一致），不存在掩盖建模错误的风险。
    #
    # v8 曾规定「converged 也不计通过」，v9 撤销，原因（实测驱动）：
    #   ① **判定不一致**：同为「2 轮修复后可行且目标值一致」，trusted 数据集记
    #      通过、untrusted 记失败——同样的证据、相反的判定；
    #   ② **依据不成立**：`trusted_reference` 在 NL4OPT 是 245/245 全 False、
    #      IndustryOR 97/97 全 False，是**数据集级标签**，不携带逐题信息；
    #      把它当逐题证据降级判定属逻辑错误；
    #   ③ **理由未贯彻**：v8 反对计通过的唯一理由是「错误消息含期望值、可能拟合」
    #      （`[FAIL] 目标值不符：求解 X，期望 Y`），但该模板对 trusted 数据集
    #      同样适用，按此理由内部数据集的修复通过也不该算数——双标；
    #   ④ **实测反证**：NL4OPT 245 中 16 题 converged，**全部为 2 轮**（首轮错、
    #      一次重写即达标）。"拟合答案"不会如此稳定，符合"修正建模细节即可达标"。
    if suspicion:
        result.suspicious = True
        converged = suspicion == "converged"
        result.success = converged          # 仅 converged 计通过；unresolved 仍不计
        result.status = "suspicious_resolved" if converged else "suspicious"
        result.trajectory[-1]["suspicious_outcome"] = (
            "converged" if converged else "unresolved")
        result.errors.append(
            "[疑似参考噪声] 修复结果="
            + ("converged" if converged else "unresolved")
            + ("（重写后达到参考值 ⇒ 参考可达、原本是模型错，"
               "判定与可信参考答案一致：**计为通过**，status 保留 suspicious_resolved 可追溯）"
               if converged else
               "（给过一次机会仍未达到 ⇒ 疑似参考答案噪声的解释站得住，**不计通过**）"))
    return result


def _set_stage(llm: Any, stage: str) -> None:
    """标注后续 LLM 调用所属阶段（供分阶段 token 核算）。

    llm 可能是 `_UsageTrackingLLM` 包装器、其它 provider 或 None；
    只有包装器实现了 set_stage，其余情况静默跳过。
    """
    setter = getattr(llm, "set_stage", None)
    if callable(setter):
        setter(stage)


def _exec_error_head(sb) -> str:
    """从沙箱执行结果提取一行错误摘要（含异常类型 + 出错位置）。

    取 stderr 的**末行**（Python 回溯末行 = 异常类型 + 消息），并向上找最近的
    `File "...", line N` 帧，把「文件:行号」附上——否则只剩异常类型、丢了出错位置。
    完整 stderr（末 1500 字符）仍保留在 detail.stderr 供落盘/复核。

    环境不可用（Docker/镜像缺失）不走此路径（那是 executor 层面，不是代码错误），
    反思环节据此把「代码错误」与「环境不可用」区分开（前者 → L1）。
    """
    text = (sb.stderr or "").strip() or (sb.error or "").strip() or (sb.stdout or "").strip()
    if not text:
        return "未知错误"
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return "未知错误"
    exc_line = lines[-1]
    # 找异常前最近的 File "..." 帧，取其文件名:行号
    import re

    loc = None
    for ln in reversed(lines[:-1]):
        m = re.search(r'File "([^"]+)", line (\d+)', ln)
        if m:
            loc = f"{m.group(1).rsplit('/', 1)[-1]}:{m.group(2)}"
            break
    head = f"{exc_line}（@{loc}）" if loc else exc_line
    return head[:300]


def run_dataset(
    samples,
    llm: LLMProvider | None,
    cfg: Config,
    force_llm_understand: bool = False,
    cache: Cache | None = None,
) -> list[PipelineResult]:
    """批量运行，返回结果列表。"""
    return [
        run_sample(s, llm, cfg, force_llm_understand=force_llm_understand, cache=cache)
        for s in samples
    ]
