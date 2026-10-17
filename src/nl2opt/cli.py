"""命令行入口：支持 list / solve / eval 子命令。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import PROTOCOL_VERSION
from .cache import Cache
from .config import load_config
from .dataset import load_any
from .llm import make_provider
from .pipeline import run_dataset, run_sample


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="nl2opt", description="自然语言 → 优化模型求解闭环")
    p.add_argument("--config", help="配置文件路径（默认项目根 config.yaml）")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="列出内置样例")

    sp = sub.add_parser("solve", help="求解单条自然语言问题")
    sp.add_argument("text", help="自然语言问题（或内置样例 id）")
    sp.add_argument("--data", help="外部数据集 JSONL 路径（可选）")
    sp.add_argument("--use-llm", action="store_true", help="用 LLM 做理解/反思（需配置 API key）")
    sp.add_argument("--force-llm-understand", action="store_true",
                    help="即使有黄金 IR 也强制走 LLM 理解（需 --use-llm）")
    sp.add_argument("--no-cache", action="store_true", help="禁用三级哈希缓存（默认启用）")
    sp.add_argument("--backend", choices=["local", "docker"],
                    help="执行后端：local=进程内求解 IR；docker=受限容器执行生成代码")

    ep = sub.add_parser("eval", help="在数据集上批量评测")
    ep.add_argument("--data", help="数据集 JSONL 路径；缺省用内置样例集")
    ep.add_argument("--use-llm", action="store_true", help="用 LLM 做理解/反思")
    ep.add_argument("--out", help="结果 JSON 输出路径（可选）")
    ep.add_argument("--force-llm-understand", action="store_true",
                    help="即使有黄金 IR 也强制走 LLM 理解（需 --use-llm）")
    ep.add_argument("--no-cache", action="store_true", help="禁用三级哈希缓存（默认启用）")
    ep.add_argument("--backend", choices=["local", "docker"],
                    help="执行后端：local=进程内求解 IR；docker=受限容器执行生成代码")
    return p


def _make_llm(cfg, use_llm: bool):
    if not use_llm:
        return None
    return make_provider(cfg)


def _apply_backend(cfg, backend: str | None) -> None:
    """命令行 --backend 覆盖配置中的 executor.backend，并在 docker 不可用时给出提示。"""
    if not backend:
        return
    cfg.to_dict().setdefault("executor", {})["backend"] = backend
    if backend == "docker":
        from . import sandbox

        reason = sandbox.preflight(cfg)
        if reason:
            print(f"[警告] 执行后端设为 docker，但沙箱不可用：{reason}", file=sys.stderr)


def _make_cache(cfg, enabled: bool) -> Cache | None:
    """构造三级缓存。enabled=False 返回 None（禁用）。路径取 paths.cache_file。

    附带 CacheContext：把协议版本 / prompt 指纹 / LLM 采样 / 后端 / 求解器选项等
    语义上下文并入缓存键，避免改 prompt、换模型、改 codegen/求解器参数后静默命中旧缓存。
    """
    if not enabled:
        return None
    cache_file = cfg.paths.get("cache_file", "artifacts/cache.json")
    cache_path = Path(cache_file)
    cache_path = cache_path if cache_path.is_absolute() else Path.cwd() / cache_path
    from .cache import build_cache_context

    context = build_cache_context(cfg, PROTOCOL_VERSION)
    return Cache(cache_path, context=context)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    cfg = load_config(args.config)

    if args.command == "list":
        ds = load_any(None)
        for s in ds:
            print(f"{s.id:18s} [{s.problem_type:5s}] {s.nl_text[:60]}")
        return 0

    if args.command == "solve":
        ds = load_any(getattr(args, "data", None))
        llm = _make_llm(cfg, args.use_llm)
        cache = _make_cache(cfg, not getattr(args, "no_cache", False))
        _apply_backend(cfg, getattr(args, "backend", None))

        # 先按 id 匹配，否则把 text 当自然语言问题（此时需要 LLM）
        sample = next((s for s in ds if s.id == args.text), None)
        if sample is None:
            if not args.use_llm:
                print("错误：无 LLM 且非内置样例 id。请提供内置 id，或加 --use-llm 配置 API key。", file=sys.stderr)
                return 2
            from .dataset import Sample as _S
            sample = _S(id="adhoc", nl_text=args.text, problem_type="milp", expected={})

        result = run_sample(sample, llm, cfg, force_llm_understand=args.force_llm_understand, cache=cache)
        if cache is not None:
            cache.save()
        _print_result(result)
        return 0 if result.success else 1

    if args.command == "eval":
        ds = load_any(getattr(args, "data", None))
        llm = _make_llm(cfg, args.use_llm)
        cache = _make_cache(cfg, not getattr(args, "no_cache", False))
        _apply_backend(cfg, getattr(args, "backend", None))
        results = run_dataset(ds, llm, cfg, force_llm_understand=args.force_llm_understand, cache=cache)
        if cache is not None:
            cache.save()
        _print_eval(results)
        if getattr(args, "out", None):
            _write_eval_out(results, args.out, cfg.llm.get("model"))
            print(f"\n结果已写入: {args.out}")
        # 落盘产物 + token 聚合报告到 artifacts_dir
        _dump_artifacts(results, cfg)
        _dump_token_report(results, cfg)
        return 0

    return 0


def _prompt_mode(r) -> str:
    """从 trajectory 推断提示词模式：`all_cards`（≈原单体）/ `layered`（按需卡片）。

    为什么必须记录：卡片策略**不体现在 model/protocol_version 上**，
    两个批次因此会被消融脚本误判为「同一配置的重复批次」而跳过对比——
    实际它们是提示词分层的 A/B 两侧。
    """
    for step in (r.trajectory or []):
        meta = step.get("understand_meta") if isinstance(step, dict) else None
        if meta and meta.get("all_cards_mode") is not None:
            return "all_cards" if meta["all_cards_mode"] else "layered"
    import os

    return "all_cards" if os.environ.get("NL2OPT_ALL_CARDS") == "1" else "layered"


def _result_to_dict(r, model: str | None = None) -> dict:
    """把 PipelineResult 序列化为完整 dict（含 trajectory 和 token_usage）。

    记录 `model` 让结果文件**自描述**——成本报表据此选择正确的单价档，
    避免换模型后仍按默认模型计价。`prompt_mode` 同理，用于提示词消融的可比性判定。
    """
    return {
        "id": r.sample_id,
        "model": model,
        "protocol_version": PROTOCOL_VERSION,
        "prompt_mode": _prompt_mode(r),
        "success": r.success,
        "suspicious": r.suspicious,
        # 疑似参考噪声的两种结局（v9 起判定不同）：
        #   True  = 重写后达到参考值 ⇒「参考是噪声」被证伪、原本是模型错；
        #           **计为通过**（success=True），但 status 保留 suspicious_resolved 可追溯
        #   False = 给过一次机会仍未达到 ⇒ 疑似噪声的解释站得住，**不计通过**
        "suspicious_resolved": (getattr(r, "status", "") == "suspicious_resolved"),
        "rounds": r.rounds,
        "status": r.status,
        "objective_value": r.objective_value,
        "variables": r.variables,
        "errors": r.errors,
        "final_ir": r.final_ir,
        "trajectory": r.trajectory,
        "token_usage": r.token_usage,
        "stage_usage": getattr(r, "stage_usage", {}),
    }


def _write_eval_out(results, out_path: str, model: str | None = None) -> None:
    """写完整评测结果（含 trajectory 和 token_usage）。"""
    Path(out_path).write_text(
        json.dumps([_result_to_dict(r, model) for r in results], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _dump_artifacts(results, cfg) -> None:
    """把每个样本的产物落盘到 artifacts_dir/<id>/ 目录。"""
    artifacts_dir = Path(cfg.paths.get("artifacts_dir", "artifacts"))
    artifacts_dir = artifacts_dir if artifacts_dir.is_absolute() else Path.cwd() / artifacts_dir
    for r in results:
        sample_dir = artifacts_dir / r.sample_id
        try:
            r.to_files(sample_dir)
        except Exception as exc:  # 单样本落盘失败不阻断整体
            print(f"  [警告] 落盘 {r.sample_id} 产物失败: {exc}", file=sys.stderr)
    print(f"\n产物已落盘到: {artifacts_dir}")


def _dump_token_report(results, cfg) -> None:
    """聚合所有样本的 token 用量，落盘 token_report.json。"""
    artifacts_dir = Path(cfg.paths.get("artifacts_dir", "artifacts"))
    artifacts_dir = artifacts_dir if artifacts_dir.is_absolute() else Path.cwd() / artifacts_dir
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    agg: dict[str, float] = {}
    n_with_usage = 0
    for r in results:
        u = r.token_usage
        if not u:
            continue
        n_with_usage += 1
        for key, val in u.items():
            if isinstance(val, (int, float)):
                agg[key] = agg.get(key, 0.0) + val
            elif isinstance(val, dict):
                for k2, v2 in val.items():
                    if isinstance(v2, (int, float)):
                        agg[f"{key}.{k2}"] = agg.get(f"{key}.{k2}", 0.0) + v2

    report = {
        "samples_with_usage": n_with_usage,
        "total_samples": len(results),
        "aggregate": agg,
    }
    out = artifacts_dir / "token_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Token 聚合报告已写入: {out}")


def _print_result(r) -> None:
    print(f"\n=== {r.sample_id} ===")
    print(f"成功: {r.success} | 轮次: {r.rounds} | 状态: {r.status}")
    if r.objective_value is not None:
        print(f"目标值: {r.objective_value}")
    if r.variables:
        print("解:", json.dumps(r.variables, ensure_ascii=False))
    if r.errors:
        print("错误:", r.errors)


def _print_eval(results) -> None:
    n = len(results)
    ok = sum(1 for r in results if r.success)
    print(f"\n=== 评测结果 ===")
    print(f"总题数: {n} | 通过: {ok} | 通过率: {ok/n*100:.1f}%")
    print(f"{'id':18s} {'成功':6s} {'轮次':4s} {'目标值':>12s}")
    for r in results:
        obj = f"{r.objective_value:.4f}" if r.objective_value is not None else "-"
        print(f"{r.sample_id:18s} {str(r.success):6s} {r.rounds:<4d} {obj:>12s}")


if __name__ == "__main__":
    raise SystemExit(main())
