# 数据集下载说明（DATA.md）

本项目用统一 JSONL 格式存放数据集。内置样例集（`data/builtin/samples.jsonl`）已自带参考答案，
**无需下载、无需网络**即可跑通完整确定性链路。

若要进行真实「自然语言 → 模型」的 LLM 评测，建议下载以下公开数据集。

---

## 一、数据格式（JSONL，每行一条）

```json
{"id": "knapsack-01", "problem_type": "milp", "nl_text": "……自然语言描述……", "expected": {"objective_value": 220.0, "variables": {"x1": 0.0, "x2": 1.0, "x3": 1.0}}}
```

| 字段 | 说明 |
|---|---|
| `id` | 唯一标识 |
| `problem_type` | `lp` / `milp` / `auto`（仅供参考，实际以 IR 变量类型判定） |
| `nl_text` | 自然语言问题描述 |
| `expected.objective_value` | 参考答案目标值（无答案可设为 `null`）。**若存在 `answer_variable`，则此值是那个变量的期望值**，不是目标函数值 |
| `expected.variables` | 参考答案解向量（可选，键为变量实例名） |
| `expected.status` | 可选，`infeasible` / `unbounded` / `no_solution` |
| `expected.answer_variable` | **可选（v27）**：当答案**不是目标函数值**而是某个变量的值时填该变量名。目标规划 / 字典序题常见（如 IndustryOR-029 的答案是 P3 偏差变量 `d3_minus`）。验证器 `check_optimality` 会改为比较**该变量在解中的值**；变量缺失时报可操作错误（提示 `d{k}_minus` 命名约定） |
| `expected.reference_status` | **可选（v28）**：参考答案**人工核定**状态 —— `corrected`（原值有误、已改正，正常计分）/ `ambiguous`（来源不明/口径歧义，**不计入能力**）/ `inconsistent`（题面自相矛盾，**不计入能力**） |
| `expected.reference_original` | **可选（v28）**：核定前的原始值（可追溯，便于复现历史结论） |
| `expected.reference_note` | **可选（v28）**：核定依据（为什么改 / 为什么判歧义），人可读 |
| `trusted_reference` | 参考答案是否可信。内置样例为 `true`（有黄金 IR，手工核验）；外部数据集（NL4OPT / OR-Bench / IndustryOR）为 `false`（未独立核验，可能有标注噪声）。 |

> 若样例内嵌了手工标注的 IR，可放在 `extra.ir` 字段；加载器会优先使用它，
> 否则走「黄金 IR 基准 → LLM 理解」的降级顺序。

### 参考答案人工核定（v28 引入）

`trusted_reference=false` 只表示「整批数据未独立核验」，**不携带逐题信息**。
逐题人工核对后，用 `expected.reference_status` 记录结论：

| 状态 | 含义 | 是否计入能力分母 |
|---|---|---|
| （无字段）| 未核对，按原值使用 | ✅ 计入 |
| `corrected` | 原值与题面/最优性不符，已按题面推导改正 | ✅ 计入（用改正后的值）|
| `ambiguous` | 参考值来源不明（多种自然口径都不符）| ❌ **排除** |
| `inconsistent` | 题面本身自相矛盾（约束无法同时满足）| ❌ **排除** |

**当前 IndustryOR 的核定结果**（97 题 → **可计分 93 题**）：

| id | 原值 | 状态 | 依据 |
|---|---|---|---|
| 005 | 18.6943 | `corrected` → **1329/70 ≈ 18.9857** | 原答案对应解预算还剩 $0.816 未花，**非最优** |
| 002 | 125.0 | `ambiguous` | 目标口径歧义（「每年 5 人×2 年」vs「每架 5 人」）|
| 011 | 3867.0 | `ambiguous` | 反解解不在可行域顶点上；p3 最优点与利润最优点重合 |
| 040 | 43090833.43 | `ambiguous` | 12 种 GDP 口径 × labor 换算组合均不符 |
| 065 | 30500.0 | `inconsistent` | A/B 均满负荷 + x≥80 + y≥50 时 **infeasible** |

> 核定源头在 `scripts/convert_industryor.py` 的 `REFERENCE_FIXES`，重新生成数据即生效。
> 详细推导见 `METHODOLOGY.md` §2.7。

### 容忍机制（参考答案噪声容忍）

当 `trusted_reference=false` 且「解可行、但目标值与参考答案不符」时，
判定为**疑似参考答案标注错误**，标记为 `suspicious`；**v8 起仍走一次重写**，据结果分两档（v9 判定）：达到参考值记 `suspicious_resolved` 并**计通过**，仍未达到记 `suspicious` 且不计通过（v9；实测 NL4OPT 245 中 16 题经此路径达标）——
避免「正确解被误判 → 触发重写 → 把对改错」的恶性循环。

- `trusted_reference=true`（内置样例）：目标值不符 → 触发反思→重写（参考答案可信）。
- `trusted_reference=false`（NL4OPT / OR-Bench / IndustryOR）：目标值不符 → 标记 `suspicious`，交人工复核。
- 若该题带 `reference_status: ambiguous/inconsistent`：**直接排除在能力统计外**（不是模型错）。

---

## 二、推荐下载源

以下数据集是「自然语言 → 优化问题」方向的公开资源。下载后请**转换成上面的 JSONL 格式**再放入 `data/` 目录。

### 1. CardinalOperations/NL4OPT（首选，带最优解 ground truth）

- HuggingFace：<https://huggingface.co/datasets/CardinalOperations/NL4OPT>
- 国内镜像：<https://hf-mirror.com/datasets/CardinalOperations/NL4OPT>
- 简介：NL4OPT 测试集转换版，已用求解器算出并校验最优解作为 ground truth。
  规模 n<1K，JSON 格式，**最适合**本项目「理解→建模→求解→验证」闭环评测。
- 数据文件：`NL4OPT/NL4OPT_with_optimal_solution.json`（245 条，含 `optimal_solution`）。
- 许可：CC-BY-NC-4.0（非商业）。

**下载与转换**：

```bash
# 下载（hf-mirror 可达）
git clone --depth 1 https://hf-mirror.com/datasets/CardinalOperations/NL4OPT data/NL4OPT

# 转换 → data/nl4opt.jsonl（245 条；15 条无最优解标 expected.status="no_solution"）
python scripts/convert_nl4opt.py

# 参考答案修正（累计 27 处，幂等）
python scripts/fix_nl4opt_full.py
```

**实测结果（v20）**：**239/245 = 97.6%**（首轮 227 + 重写修复 12）；
未通过 6 题均为 `suspicious`（解可行但与参考不符，而参考本身不可信）。
*历史 v1：首轮 213、校正参考答案后 232/245 = 94.7%。*
详见 §三「NL4OPT 详情」与 `项目.md` §8.2。

### 2. AriaAICompany/or-bench-problems（100 道，带 schema + 黄金目标值）✅ 已下载并转换

- HuggingFace：<https://huggingface.co/datasets/AriaAICompany/or-bench-problems>
- 国内镜像：<https://hf-mirror.com/datasets/AriaAICompany/or-bench-problems>
- 简介：100 道库存（34）/ 指派（33）/ 排班（33）问题，每题含结构化 `schema_payload`
  **和**自然语言题面 `prompt_en`，黄金目标值来自 OR-Tools CP-SAT（官方 1 线程 6–8 s）。
  数据文件为 `data/or-bench-problems/data/problems.jsonl`（另有 `protocol.md` 评测协议、`splits.json` 划分）。
- 许可：**CC-BY-4.0**（可商用，比 NL4OPT 的 CC-BY-NC 宽松）。
- 规模明细：development 60 / test 40；difficulty easy 53 / medium 27 / hard 12 / infeasible 8；
  `gold_status` = optimal 92 + **infeasible 8**。
- 注意：这里的 "ORBench" 是运筹学（Operations Research）评测，**不是** HF 上大量出现的
  "Over-Refusal Benchmark"（LLM 安全基准，二者无关，勿混淆）。

**下载与转换**（已实测）：

```bash
# 下载（hf-mirror 可达；官方域名在国内通常直连不通）
git clone --depth 1 https://hf-mirror.com/datasets/AriaAICompany/or-bench-problems data/or-bench-problems

# 转换：schema_payload → IR → 本项目 JSONL（三题族分别建模，语义对齐官方 CP-SAT 参考实现）
python scripts/convert_orbench.py                    # 全量 100 题 → data/orbench.jsonl
python scripts/convert_orbench.py --split development --family inventory --verbose
```

**实测结果**（详见 `artifacts/ORBENCH_REPORT.md`）：

| 指标 | 值 |
|---|---|
| 本地 scipy/HiGHS 命中黄金值 | **100/100** |
| 沙箱 PuLP/CBC 命中黄金值 | **100/100** |
| 两后端互相一致 | **100/100** |
| 不可行题状态核对 | 8/8 一致 |

> 该数据集**每行自带 schema**，因此走「schema → IR」直接映射（等价于官方 `structured-solver`
> 基线），用于压测 T2–T5（建模→生成→执行→验证）；`prompt_en` 保留在 `nl_text` 中，
> 可用于「自然语言 → IR」的 T1 对照实验。

**转换产物**：

| 文件 | 题数 | 说明 |
|---|---|---|
| `data/orbench.jsonl` | 100 | schema → IR（**内嵌 IR**，走确定性链路 T2–T5） |
| `data/orbench_nl.jsonl` | 100 | **散文题面；test 40 题在题面后追加 `prompt_json` companion**（走 LLM 理解 T1）。见下方 ⚠️ |
| `data/orbench_nl_nocompanion.jsonl` | 100 | 旧版（**无 companion**），保留用于复现 v20 的历史结论（27.5%）|
| `data/orbench_nl_sub12.jsonl` | 12 | 分层抽样（`D-*` / `T-*` / 三题族各覆盖） |
| `data/orbench_nl_sub30.jsonl` | 30 | 分层抽样（更大规模） |
| `data/orbench_nl_sched.jsonl` | 33 | 排班题族全量（专批） |

> ⚠️ **`orbench_nl.jsonl` 的信息完整性问题（2026-10 查证，已修复）**
>
> 原始 `problems.jsonl` 每条记录携带**三份输入**：`prompt_en`（散文）、`prompt_json`（完整 JSON）、
> `schema_payload`。**100/100 题都有 `prompt_json`，目标系数（`weight`/`cost`/`penalty`）从不缺失。**
>
> 但初版 `convert_orbench.py` 只取 `prompt_en`（第 376 行），`--strip-ir` 又去掉了内嵌 IR ——
> 于是 **test split（40 题）的散文里没有目标系数，答案在给定输入下不可解**。
>
> 后果（v20 实测）：`schema_embedded` 60 题通过率 **96.7%**，`text_only` 40 题仅 **27.5%**；
> 后者 40 题中 **23 题属"缺必需数值"**，剔除后模型在可答题上的通过率为 **64.7%**。
>
> **这不是 OR-Bench 的数据缺陷，是转换口径造成的**。**现已修复**：`convert_orbench.py` 的
> `attach_companion=True`（默认）会把 `prompt_json` 作为**独立段落附在题面后**
> （格式 `\n\n--- JSON companion ---\n{...}`），而不是塞进 `nl_text` 的散文里。
>
> **修复后实测**：test 40 **40/40 = 100%**（修复 29、回退 0）；逐题对官方 `gold_objective`
> 核对 **39/40 完全一致**（余 1 为浮点抖动）、**8/8 不可行题正确识别**、34 题首轮通过。
> 合并 development 后 OR-Bench 全 100 题为 **98/100 = 98.0%**。
> 旧版保留为 `orbench_nl_nocompanion.jsonl`（`--no-companion` 生成）。
> 详见 [`METHODOLOGY.md`](METHODOLOGY.md) §2.6。

**自然语言链路（T1）对照实验**：

```bash
# 生成不含内嵌 IR 的版本（走 LLM 理解）；默认会把 test 40 的 prompt_json 作为
# 独立段落附在题面后（attach_companion=True）。加 --no-companion 可生成旧版。
python scripts/convert_orbench.py --strip-ir --out data/orbench_nl.jsonl
python scripts/convert_orbench.py --strip-ir --no-companion --out data/orbench_nl_nocompanion.jsonl

# 评测（需 API Key；OR-Bench 模型规模大，务必给足 max_tokens）
python -m nl2opt eval --data data/orbench_nl.jsonl --use-llm --no-cache \
    --out artifacts/eval_orbench_nl.json
python scripts/report_orbench_nl.py artifacts/eval_orbench_nl.json
```

- prompt 有天然对照组：`D-*` 模板（60 题，`development` split）**内嵌完整 JSON schema**；
  `T-*` 模板（40 题，`test` split）是**同一问题的散文版**。**现已把配套 `prompt_json` 作为
  companion 附在散文后**（v28 起，`attach_companion=True` 默认），报告按
  `schema_embedded` / `text_only` 分别统计。
- ⚠️ **`max_tokens` 必须给足**（`config.yaml` 已设 65536）：这些题的 IR 规模远超 NL4OPT 的小 LP，
  reasoning 会吃掉大半输出预算，预算不足会让正文 JSON 被截断。
  （v6 由 32768 提到 65536——32768 对最大的排班实例是临界值。）
- ⚠️ **`max_tokens` 必须给足**（`config.yaml` 已设 32768）：这些题的 IR 规模远超 NL4OPT 的小 LP，
  reasoning 会吃掉大半输出预算，预算不足会让正文 JSON 被截断。
- ⚠️ **需要 IR 表达力支持字符串/多维索引**：排班题的自然写法（`start_J1`、`assign_J1_M1`）
  曾因 `index_set` 仅支持一维整数而被 Schema 拒绝；现已扩展支持（见 `项目.md` §14.2），
  排班题产出率因此从受阻变为 **33/33**。

**实测结果**：

| 批次 | 规模 | 结果 | 结果文件 |
|---|---|---|---|
| structured（schema → IR） | 100 | 本地 HiGHS 与沙箱 CBC **均 100/100** 命中 CP-SAT 黄金值；8/8 不可行状态一致 | `artifacts/eval_orbench_local.json` / `eval_orbench_docker.json` |
| 自然语言 12 题抽样 | 12 | 产出 IR **10/12**、目标值命中 **9/12**、目标签名一致 3/12、重写修复 3 题 | `artifacts/eval_orbench_nl_sub12.json` |
| 自然语言排班专批 | 33 | 产出 IR **33/33**、名义命中 31/33、**实质通过 29/33 (87.9%)**、签名一致 18/33 | `artifacts/eval_orbench_nl_sched.json` |
| **自然语言全量 100（v20）** | 100 | 总 **69/100**；拆开：development **58/60 = 96.7%**、test **11/40 = 27.5%**（缺 companion）| `artifacts/eval_orbench_nl_v20.json` |
| **test 40（补入 companion）** | 40 | **40/40 = 100%**（修复 29、回退 0）；对官方 `gold_objective` **39/40 完全一致**、**8/8 不可行正确识别**、34 题首轮通过 | `artifacts/eval_orbench_test40_companion.json` |

> **「判别力分层」洞察**（排班专批）：当黄金目标值为 0 时，退化模型（目标函数无变量项）
> 也会靠巧合命中，无法区分真假建模 → 需按「可判别层」统计真实正确率（该批可判别层仅 3/5）。
> 为此验证器新增**目标定义检查**，拦截「目标无项 / 系数全 0」的退化模型。
> 报告见 `artifacts/ORBENCH_NL_REPORT.md`。

### 3. CardinalOperations/IndustryOR（100 道真实工业题）✅ 已下载并转换

- HuggingFace：<https://huggingface.co/datasets/CardinalOperations/IndustryOR>
- 国内镜像：<https://hf-mirror.com/datasets/CardinalOperations/IndustryOR>
- 简介：**首个工业运筹基准**，100 道真实工业场景问题（生产排程、库存、运输、选址、人员排班、
  可靠性设计等），覆盖 5 类（线性规划 / 整数规划 / 混合整数规划 / **非线性规划** / 其他）
  与 3 个难度（Easy 39 / Medium 41 / Hard 20）。与 NL4OPT 同属 CardinalOperations。
- 数据文件：**单文件** `IndustryOR.json`（顶层为 list，100 条）。
- 许可：**CC-BY-NC-4.0**（非商业，与 NL4OPT 相同）。
- 字段：`id`（1–100）、`en_question`（题面）、`en_answer`（最优目标值，字符串）、`difficulty`。
  **无黄金 IR、无变量解** → 走 LLM 理解链路。

**下载**（已实测，二选一）：

```bash
# 方式 A：git clone
git clone --depth 1 https://hf-mirror.com/datasets/CardinalOperations/IndustryOR data/IndustryOR

# 方式 B：只下数据文件（更轻，推荐）
curl.exe -L -o data/IndustryOR.json https://hf-mirror.com/datasets/CardinalOperations/IndustryOR/resolve/main/IndustryOR.json
```

> ⚠️ **必须先剔除非线性题，否则 IR Schema 会拒绝**。本项目 IR 只支持**线性**模型，
> 而 IndustryOR 含非线性规划（NLP）题（如概率/可靠性**乘积**、二次分配 QAP）。
>
> ⚠️ **不能用关键词筛选**：题面里的 `square meters` / `cubic meters` / `cm^3` 只是**计量单位**。
> 实测初扫命中 14 道"几何题"，逐题核对后**大多是线性的**（面积/体积只是给定参数）。

**转换**（两步，已实测）：

```bash
# ① 用 LLM 逐题判定线性性（需 API Key；100 题约 13 批调用）
python scripts/classify_industryor.py --batch 8
#    → data/industryor_classification.json（含每题判定 + 理由，可审查）

# ② 剔除非线性并转换
python scripts/convert_industryor.py                       # 全部线性题 → data/industryor.jsonl
python scripts/convert_industryor.py --limit 10 --out data/industryor10.jsonl
python scripts/convert_industryor.py --keep-unknown        # 保留未判定的题（默认剔除，保守）
```

**实测分类结果**：100 题中判定 **linear 97 / nonlinear 3**（未判定 0）。
剔除的 3 道经独立复核确认：

| 题号 | 判定理由 | 复核 |
|---|---|---|
| `id=17` | 成功概率含变量乘积 | 摧毁概率 `1-Π(1-p)^x`，非线性 |
| `id=71` | 分配变量相乘为二次 | 二次分配问题（QAP），含 `x_ij·x_kl` |
| `id=99` | 系统可靠性为可靠性乘积 | 可靠性连乘，非线性 |

**转换产物**：

| 文件 | 题数 | 说明 |
|---|---|---|
| `data/IndustryOR/` | — | **完整 git 仓库**（clone 结果，含 `.git`/`.gitattributes`/`README.md`/`IndustryOR.json`） |
| `data/IndustryOR.json` | 100 | 原始下载文件（**未改动**，`data/IndustryOR/IndustryOR.json` 的副本；脚本硬编码读此路径） |
| `data/industryor_classification.json` | 100 | 线性性判定结果 + 理由（含 `provenance` 字段说明来源） |
| `data/industryor.jsonl` | **97** | 全量线性题（Easy 39 / Medium 40 / Hard 18） |
| `data/industryor10.jsonl` | **10** | 分层抽样（Easy 4 / Medium 4 / Hard 2）；题号与既有评测 artifact 一致 |

> **目录布局说明**：`git clone` 落在 `data/IndustryOR/`（子目录），而
> `classify_industryor.py:29` 与 `convert_industryor.py:32` 硬编码 `data/IndustryOR.json`（平铺路径）。
> 因此 clone 后需**复制**（不是移动，以保留仓库完整）：
> ```bash
> cp data/IndustryOR/IndustryOR.json data/IndustryOR.json
> ```
>
> **分类结果的来源**（2026-10 恢复时记录）：`industryor_classification.json` 沿用此前的 LLM 逐题判定
> （linear 97 / nonlinear 3），其中 3 道非线性题（`17`/`71`/`99`）已对当前文件**逐题复核原文确认**；
> 文件内 `provenance` 字段标明"未重新调用 LLM"。若需重新分类：
> `python scripts/classify_industryor.py --batch 8`（⚠️ 有 API 成本，100 题约 13 批）。
>
> **10 题抽样为何不用 `--limit 10`**：该选项取"按 id 排序后的前 10 道"，
> 而既有评测用的是**分层抽样**（Easy 4 / Medium 4 / Hard 2）。为保证与
> `eval_industryor10_v7/v8/v22` 评的是同一批题，恢复时按 artifact 里的 10 个 id 精确取：
> `005 / 009 / 015 / 022 / 023 / 040 / 045 / 052 / 081 / 090`。

> 评测命令：
> ```bash
> python -m nl2opt eval --data data/industryor10.jsonl --use-llm \
>     --out artifacts/eval_industryor10_v8.json
> ```
>
> ⚠️ IndustryOR 题面显著长于 NL4OPT（抽样 486–1305 字符，全量最长 5990），且含 Markdown 表格，
> 单题耗时与 token 消耗明显更高；`config.yaml` 的 `max_tokens: 32768` 已够用。

**题面自足性（2026-10 查证，与 NL4OPT 同类，无 OR-Bench 的问题）**：

| 检查项 | 结果 |
|---|---|
| 字段 | `id` / `en_question` / `en_answer` / `difficulty` —— **无结构化 companion**（上游单文件，未改动） |
| 「数据在别处」指针式表述 | **0/100** |
| 目标语义显式覆盖 | 97/100；另 3 题（`id` 2 / 51 / 74）目标为**隐含但可推断**（如 "how many … to contribute to air defense"、"should hire and how many" + 费率、"formulate and solve" + 日租成本） |
| `en_answer` 为空 | **0** |
| **「空目标」失败（行为证据）** | **0** ← 与 NL4OPT 相同；OR-Bench test 有 19 例 |

> ⚠️ 注意 `id=55` 题面的数字**全用英文单词拼写**（`two courses`、`seven courses`），
> 数字正则返回 0 个 —— 但它其实完全自足（`en_answer=4`）。
> **数字检测不能单独作为自足性判据**（详见 `METHODOLOGY.md` §2.6 的两个检测陷阱）。

**参考答案核定（v28 完成，取代此前的「待办」）**

> ⚠️ 本节曾记录「4 题 `suspicious` **尚未独立核验**」。该待办**已完成**，
> 且结论与当时的初步判断**部分相反**——必须更正：

| 题 | 当时的初步判断 | 核定结论（v28 / v29）|
|---|---|---|
| `industryor-005` | 「r3 收敛到 18.6943 → 参考值可达、属模型错」 | ❌ **当时判断错误**。18.6943 对应的解 **预算还剩 $0.816 未花**——把钱换成 okra 还能提高纤维，所以它**不是最优**。正确答案是 **1329/70 ≈ 18.9857**（已改 `corrected`）|
| `industryor-023` | 「r2 收敛到 21500 → 参考值可达」 | ✅ 参考值对，但**模型错在固定成本语义**：题面的「weekly fixed costs」是每期必付，不应建成开关成本（y + big-M）。修提示词后 **1 轮通过** |
| `industryor-015` | 「未收敛，模型够不到 168」 | ✅ 参考值对，模型不稳定；**v24 起已通过**（其 168 曾被某版本精确命中）|
| `industryor-040` | 「未收敛，模型够不到 43.09M」 | ⚠️ **判为歧义**（`ambiguous`）：12 种「GDP 口径 × 人年→人月换算」组合均复现不了 43,090,833；含投入产出平衡得 36,289,567、去掉平衡则**无界**或 83,000,000 |

**核心教训**：「某版本能收敛到参考值」**不能**证明参考值正确。
005 就是反例——当时的 v8 能"算到" 18.6943（因为 v10 之前错误消息里含期望值），
但那个值**数学上不是最优**（预算没用满）。判断参考值对错只能靠**题面推导或独立求解**，
不能靠"某个模型能凑到它"。

**现已完成全量 97 题的人工核定**（5 题）：方法与逐题依据见
[`METHODOLOGY.md`](METHODOLOGY.md) §2.7；字段说明见本文 §一「参考答案人工核定」。

### 4. NL4Opt 官方原始数据（数学建模标注）

- 主页：<https://nl4opt.github.io>
- 简介：NLP4LP（自然语言 → 线性规划）竞赛原始数据，含目标/约束标注。
  原始版只给数学模型、无最优解，需自行求解补 ground truth。

> ⚠️ 重要更正：网上常被引用的 `microsoft/ORBench` 在 HuggingFace 上并不存在运筹学数据集；
> 搜索 "OR-Bench" 命中的几乎全是「Over-Refusal Benchmark」（LLM 过度拒绝安全基准），与优化无关。
> 请以上述 1、2、3 三个带最优解的数据集为准。
>
> 下载提示：本机无法直连 huggingface.co，但 **hf-mirror.com 可达**。
> 可直接用镜像下载，例如：
> `git clone https://hf-mirror.com/datasets/CardinalOperations/NL4OPT`

---

## 三、已转换数据集汇总 ✅

三个外部数据集均已下载并转换完成（详见 §二 各自的「下载与转换」小节）：

| 数据集 | 原始规模 | 转换产物 | 链路 | structured 对拍 | 最新评测结果 |
|---|---|---|---|---|---|
| **NL4OPT** | 245 | `data/nl4opt.jsonl`（245） | LLM 理解 | — | **v20：239/245 = 97.6%**（首轮 227 + 重写修复 12）<br>*历史 v1：首轮 213、校正标注后 232/245 = 94.7%* |
| **OR-Bench Problems** | 100 | `data/orbench.jsonl`（100，内嵌 IR）<br>`data/orbench_nl.jsonl`（100，无 IR；test 40 带 JSON companion）| schema→IR **及** LLM 理解 | **100/100**（本地 HiGHS 与沙箱 CBC 均命中 CP-SAT 黄金值）| development 58/60 = **96.7%** + test 40（补 companion）**40/40 = 100%**<br>⇒ **合并 98/100 = 98.0%** |
| **IndustryOR** | 100 | `data/industryor.jsonl`（**97**，已剔除非线性 3 题）<br>`data/industryor10.jsonl`（10 抽样）| LLM 理解 | — | **v29 全量：可计分 90/93 = 96.8%**（总 91/97 = 93.8%；首轮 84 + 重写修复 6）<br>4 题因参考歧义/题面矛盾排除，见 §一「参考答案人工核定」 |

> **IndustryOR 的 97 题里有 5 题参考值经人工核定**：005 已改正（`corrected`，计入）、
> 002/011/040 标 `ambiguous`、065 标 `inconsistent`（后 4 题**不计入能力分母**）。
> 因此**可计分集合是 93 题**，不是 97 题。各题依据见 §一 的核定表与 `METHODOLOGY.md` §2.7。

> 三者共同点：均**无黄金 IR、无变量解**（OR-Bench 除外，它自带 schema），
> 因此 `trusted_reference=false`，走「LLM 理解 + 容忍机制」链路。
>
> ⚠️ **但三者的「输入完整性」不同**：NL4OPT 与 IndustryOR 题面**自足**；
> OR-Bench 的 `test` split（40 题）**散文题面缺目标系数**，而配套 `prompt_json` 未被喂入 ——
> 导致该 40 题中 23 题在给定输入下不可解；现已通过 `convert_orbench.py` 的
> `attach_companion=True` 把 companion 作为**独立段落附在题面后**修复（40/40 = 100%）。
> 详见 [`METHODOLOGY.md`](METHODOLOGY.md) §2.6。

### NL4OPT 详情

| 项 | 值 |
|---|---|
| 原始文件 | `NL4OPT/NL4OPT_with_optimal_solution.json` |
| 转换后 | `data/nl4opt.jsonl` |
| 总条数 | 245 条 |
| 数值目标值 | 230 条 |
| 无最优解（`No Best Solution` / `-99999`） | 15 条 → 标记为 `expected.status = "no_solution"` |
| 转换脚本 | `scripts/convert_nl4opt.py` |
| 参考答案修正 | 累计 27 处 → `scripts/fix_nl4opt_full.py`（幂等，覆盖全部 27 处） |

> **参考答案修正说明**（累计 27 处，两个来源，均由 `fix_nl4opt_full.py` 覆盖）：
>
> 1. **全量核验 19 处** — 12 处数值答案被独立求解器证伪并修正，7 条 `no_solution` 误标
>    （实际有可行解）改为数值答案。
> 2. **早期抽样 8 处** — `nl4opt-015 / 178 / 057 / 201 / 232 / 031 / 079 / 181`
>    （原 `fix_nl4opt_answers.py` 已并入本脚本，避免重转后丢失）。
>
> ⚠️ **重新转换的正确流程**：`convert_nl4opt.py` 从原始文件重新生成时会**覆盖全部修正**
> （实测重转后 26/27 处偏离正确值，仅 `nl4opt-015` 因已同步进原始文件而保留）。因此重转后**必须**执行：
>
> ```bash
> python scripts/convert_nl4opt.py      # 重转（会重置参考答案）
> python scripts/fix_nl4opt_full.py     # 恢复 27 处修正（幂等，可重复运行）
> ```
>
> 已验证：**重转 + 修正后的数据与修正前逐字段一致（0 差异）**。
>
> 修正只作用于本项目转换产物；原始下载文件保持原样以追溯上游事实（`nl4opt-015` 除外，该处已同步）。

重新转换：

```bash
python scripts/convert_nl4opt.py
```

字段说明：NL4OPT 每条只含 `en_question`（英文问题）+ `en_answer`（最优目标值），
**无黄金 IR、无变量解**，因此它走「LLM 理解链路」——需要配 `DEEPSEEK_API_KEY` 才能评测。

---

## 四、转换后如何接入

把转换好的文件放到任意位置（本项目已转换的都在 `data/`），然后：

```bash
# 列出（仍用内置样例）
python -m nl2opt list

# 在外部数据集上批量评测（确定性链路，用黄金 IR / 内嵌 IR）—— 仅 OR-Bench structured 适用
python -m nl2opt eval --data data/orbench.jsonl

# 启用 LLM 理解（需配置 DEEPSEEK_API_KEY）—— NL4OPT / IndustryOR / OR-Bench NL 必须用这个
python -m nl2opt eval --data data/nl4opt.jsonl --use-llm
python -m nl2opt eval --data data/industryor.jsonl --use-llm      # 全量 97 题（可计分 93，见 §一）
python -m nl2opt eval --data data/industryor10.jsonl --use-llm    # 10 题抽样（快速冒烟）
python -m nl2opt eval --data data/orbench_nl_sub12.jsonl --use-llm

# 能力汇总（OR-Bench 按 split 拆分 + IndustryOR 可计分口径）
python scripts/report_capability.py

# 常用开关
#   --out <path>        结果 JSON（含完整 trajectory 与 token_usage）
#   --no-cache          禁用三级哈希缓存（默认启用）
#   --backend docker    用沙箱执行生成代码（默认 local 进程内求解 IR）
#   --force-llm-understand  即使有黄金 IR 也强制走 LLM 理解（用于评估理解质量）
```

评测后会自动落盘到 `artifacts/`：每个样本一个产物目录（`final_code.py` / `solution.json` /
`verification_report.json` / `rewrite_trace.json` / `final_error.json` / `token_usage.json`），
以及聚合的 `token_report.json`。成本折算用 `python scripts/report_cost.py`。

---

## 五、关于参考答案

### 内置样例（`data/builtin/samples.jsonl`）

参考答案均为**手工推导并逐条核对**的值（`expected.objective_value` 与 `expected.variables`）。
若你发现某条答案有误，欢迎在 `data/builtin/samples.jsonl` 中修正并提交。

### 外部数据集（`trusted_reference=false`）

三批外部数据的参考答案**都未经独立核验**，但可靠性差异很大：

| 数据集 | 答案来源 | 可信度 |
|---|---|---|
| **OR-Bench** | 官方 CP-SAT 黄金值 | **高**——structured 100/100 双后端命中；test 40 的 `gold_objective` 逐题核对 **39/40 完全一致**（余 1 为浮点抖动）|
| **NL4OPT** | 官方用求解器算出 | **中**——已累计修正 **27 处**（19 处全量核验 + 8 处早期抽样，见 `scripts/fix_nl4opt_full.py`）；修正后仍有个别 `suspicious` |
| **IndustryOR** | 原始 `en_answer` **只给一个裸数值**（无最优性证书、无变量取值）| **低**——已人工核定 5 题：1 改正 / 3 歧义 / 1 题面矛盾（见 §一）|

**因此 IndustryOR 的「未通过」不能直接读作模型错。** 归因顺序应是：

1. 先看 `expected.reference_status` —— `ambiguous`/`inconsistent` 直接排除（不是模型错）；
2. 再看该题是否属**已知限制**（如 VRP 变量族超出单次输出预算、非确定性）；
3. 最后才归因到建模能力。

详细流程与逐题依据见 [`METHODOLOGY.md`](METHODOLOGY.md) §2.7。
