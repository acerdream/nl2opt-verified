# nl2opt

> 把自然语言描述的线性规划（LP）/ 混合整数规划（MILP）问题，转换成**可运行、可独立验证、可闭环修复**的求解模型与代码。

**核心主张**：LLM 负责「理解与建模」，但不被信任——每一次生成都要经过**独立的求解器 + 验证器**复核；验证失败时按错误层级定位并回退修复，而不是无脑重试。

```
NL4OPT 245 题    239/245 = 97.6%
OR-Bench 100 题   98/100  = 98.0%   （development 58/60 + test 40/40）
IndustryOR 97 题  90/93   = 96.8%   （可计分口径；4 题因参考歧义/题面矛盾排除）
```

---

## 它解决什么问题

教科书式的「LLM 直接写代码解优化题」有三个结构性缺陷：

| 缺陷 | 后果 |
|---|---|
| LLM 生成的代码**无法独立校验** | 答案错了不知道错在哪，也无法判断是「模型错」还是「参考答案错」 |
| 一次生成、无反馈 | 错过一条约束就永久失败 |
| 参考答案本身可能错 | 正确解被判错，甚至被"修"成错的 |

本项目针对这三点分别给出机制：**IR 中间表示 + 独立验证器**、**按错误层级的闭环重写**、**参考答案噪声容忍 + 人工核定**。

## 设计哲学：三权分立

| 角色 | 职责 | 实现 |
|---|---|---|
| **生成方** | 理解题面、抽取 IR、反思失败原因、重写 | LLM（DeepSeek，OpenAI 兼容协议） |
| **裁判方 ①** | 求解 | scipy / HiGHS（本地）或 PuLP / CBC（Docker 沙箱） |
| **裁判方 ②** | **不依赖生成代码**，直接基于 IR 逐条复核可行性、整数性、目标值 | `verifier.py` |

关键点：验证器**不看 LLM 生成的代码**，只拿 IR 重新算一遍。这使「代码写错」与「建模错」可区分。

## 闭环流程

```
INIT → UNDERSTAND → MODEL → GENERATE → EXECUTE → VERIFY → REFLECT → REWRITE → DONE/FAILED
         (NL→IR)      (校验)   (IR→代码)   (求解)    (独立复核) (归因)     (分层回退)
                                                       │
                                                     失败 → 按错误层级路由
                                                            L1 代码层：只补代码
                                                            L2 建模层：重写 IR
                                                            L3 理解层：重新读题面
                                                            L4 不可自动修复：转人工
```

**终止条件**：验证通过 / 达最大轮数 / 连续相同错误 / token 预算耗尽 / 转人工。

## 快速开始

```powershell
# Windows PowerShell
$env:PYTHONPATH = "$PWD\src"

python scripts\run_tests.py                # ① 自检（24/24 零依赖 smoke，无需 Docker/网络）
python -m nl2opt list                      # ② 列出内置样例
python -m nl2opt solve knapsack-01         # ③ 求解单题（黄金 IR，确定性链路）
python -m nl2opt eval                      # ④ 批量评测内置样例（local 后端）
```

```bash
# 启用 LLM（需在 .env 或环境变量中配置 DEEPSEEK_API_KEY）
python -m nl2opt solve "某工厂生产两种产品……使总利润最大？" --use-llm
python -m nl2opt eval --data data/nl4opt.jsonl --use-llm
```

```bash
# 可选：Docker 沙箱后端（隔离 + 资源限制）
docker build -t nl2opt-sandbox sandbox/
python scripts/check_sandbox.py
python -m nl2opt eval --backend docker
```

> 无 API Key 时，理解阶段降级到黄金 IR、反思退化为启发式归因——**确定性链路仍可完整跑通**。

## 实测结果

### NL4OPT（245 题 · 连续 LP · 参考答案经独立核验修正 27 处）

| 版本 | 结果 | 说明 |
|---|---|---|
| **v20（当前）** | **239/245 = 97.6%** | 首轮 227 + 重写修复 12；未通过 6 题均为 `suspicious`（参考不可信）|
| v8（历史） | 242/245 = 98.8% | ⚠️ 跑在 v10 之前，**错误消息含期望值**（存在答案泄露路径），不可与 v20 直接比较 |

### OR-Bench（100 题 · 库存/指派/排班 · 官方 CP-SAT 黄金值）

| 口径 | 结果 |
|---|---|
| `structured`（schema → IR，双后端对拍） | **100/100** 命中 CP-SAT 黄金值（HiGHS 与 CBC 一致）|
| `development` 60（题面内嵌 schema） | 58/60 = 96.7% |
| `test` 40（散文题面 + 补入 JSON companion） | **40/40 = 100%**（对官方 gold 逐题核对 39/40 完全相同，余 1 为浮点抖动）|
| **合并** | **98/100 = 98.0%** |

> ⚠️ test 40 若**不喂** companion（旧版输入），通过率只有 27.5% —— 但那是**输入缺失**，不是模型能力。此结论曾导致「模型在库存/指派上偏弱」的误判，已撤回。

### IndustryOR（100 题原始 → 97 题线性 · 真实工业场景）

| 口径 | 结果 |
|---|---|
| **可计分（93 题）** | **90/93 = 96.8%** |
| 总（97 题） | 91/97 = 93.8%（首轮 84 + 重写修复 6、平均 1.08 轮）|

**为什么是 93 而不是 97**：该数据集的 `en_answer` 只有一个裸数值（无最优性证书、无变量解），人工逐题核定后发现 5 题参考值有问题：

| 题 | 判定 | 依据 |
|---|---|---|
| 005 | **已改正** → 1329/70 ≈ 18.9857 | 原答案对应的解**预算还剩 $0.816 未花**，不是最优 |
| 002 / 011 / 040 | `ambiguous`（不计分）| 目标口径歧义 / 反解解不在可行域顶点 / 12 种 GDP 口径组合均不符 |
| 065 | `inconsistent`（不计分）| 题面约束自相矛盾（A/B 均满负荷 + 产量下限无法同时满足）|

剩余 3 题未通过也**不是建模能力问题**：001 差 0.13%（时序口径）、036 的 VRP 需 420 条弧（超出单次输出预算）、048 存在非确定性。

## 关键设计

### 1. 中间表示 IR（而非直接生成代码）

LLM 只输出结构化 IR（变量 / 目标 / 约束），代码由**确定性的模板投影**生成：

```json
{
  "variables": [{"name": "x", "var_type": "continuous", "lower": 0}],
  "objective": {"sense": "maximize", "expr": {"terms": [{"var": "x", "coef": 2.0}]}},
  "constraints": [...],
  "assumptions": ["题面未给系数，按…处理"]
}
```

好处：结构上消除语法/API/未定义变量类错误；错误集中在**可诊断的建模层**；且 IR 可被独立验证器直接复核。

### 2. 双执行后端

- `local`：进程内 scipy/HiGHS 直接求解 IR（快，无需 Docker）
- `docker`：把 IR 投影成 PuLP 代码，在受限容器中执行（`--network=none`、内存/CPU/PID 限制、双重超时）

两后端在 100 题上**逐题对拍 0 处不一致**。

### 3. 验证器：不信任生成代码

逐条复核：① 代码是否无异常执行 ② 求解状态（optimal/infeasible/unbounded）③ 解代回约束、变量域、整数性 ④ 目标值（**相对容差**）⑤ 目标非退化（不能空目标）⑥ 语义覆盖（变量/约束是否漏）⑦ 目标一致性（由 IR 重算 vs 报告值）。

### 4. 按错误层级重写（而非重试）

反思阶段把失败归因到 L1–L4，分别路由到：代码补丁 / 重写 IR / 重新理解题面 / 转人工。避免「同一个错反复犯」。

### 5. 答案泄露防护

失败消息里**不含参考答案数值**（只留 `detail` 字段供人复核）。否则 LLM 能"反推"出能算出该值的模型——那是拟合答案，不是建模。

### 6. 参考答案噪声容忍

`trusted_reference=false` 且「解可行但目标不符」时，标记 `suspicious` 但仍走一次重写：
- 达到参考值 → `suspicious_resolved`，**计为通过**（参考可达 ⇒ 原本是模型错）
- 仍未达到 → `suspicious`，不计通过（可能是参考错）

再加上**人工核定层**：`reference_status` 为 `corrected` / `ambiguous` / `inconsistent`，后两者排除在能力分母外。

## 目录结构（精简）

```
src/nl2opt/
├── ir.py             # IR Schema（Pydantic 严格校验）
├── verifier.py       # 独立验证器（裁判方）
├── codegen.py        # IR → PuLP 代码
├── solver/engine.py  # scipy/HiGHS
├── semantic.py       # 语义覆盖比对（变量名/顺序无关）
├── cache.py          # 三级哈希缓存
└── pipeline/         # understand / reflect / rewrite / orchestrator / prompts
scripts/              # 数据转换、评测、报表（run_tests 为零依赖自检入口）
data/                 # 内置样例 + 三个外部数据集（见 DATA.md）
artifacts/            # 评测轨迹与报告
tests/                # pytest 分层测试（82 用例）
```

## 已知限制（诚实报告）

| 限制 | 说明 |
|---|---|
| **变量族规模** | 像 VRP 那样需要 420 条弧的题，超出 LLM 单次输出预算，IR 会被截断（036）|
| **非确定性** | 相同提示词下个别题的建模选择会变（如连续 vs 整数），048 因此时对时错 |
| **参考值可信度** | IndustryOR 的参考值可靠性低，已核定 5 题；NL4OPT 修正过 27 处 |
| **统计效力** | 所有消融对比的样本量偏小（p 值均不显著），结论应读作**方向性证据** |
| **仅线性模型** | IR 不支持非线性（3 道 NLP 题已从 IndustryOR 剔除）|

## 文档

| 文档 | 内容 |
|---|---|
| [`项目.md`](项目.md) | 完整实现文档：状态机、各阶段设计、协议版本历史、目录结构、配置 |
| [`METHODOLOGY.md`](METHODOLOGY.md) | 方法论：为什么这么做、凭什么相信、踩过的坑（含 IndustryOR 参考值核定 §2.7）|
| [`DATA.md`](DATA.md) | 数据集下载/格式/转换、参考答案核定口径 |
| `artifacts/*.md` | 单次评测的原始报告 |

## 许可

- 项目代码：MIT
- NL4OPT 数据：CC-BY-NC-4.0（**非商业**）
- IndustryOR 数据：CC-BY-NC-4.0（**非商业**）
- OR-Bench 数据：CC-BY-4.0

> ⚠️ 注意：HF 上的 "ORBench" 多为 Over-Refusal Benchmark（LLM 安全基准），与运筹学无关。本项目用的是 `AriaAICompany/or-bench-problems`。
