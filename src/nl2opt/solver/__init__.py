"""求解器子包：对外暴露 solve 与 SolverResult。"""

from .engine import SolverResult, solve

__all__ = ["solve", "SolverResult"]
