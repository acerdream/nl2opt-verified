"""nl2opt：自然语言线性/整数规划 → 可运行、可验证、可迭代修复的求解模型与代码。

闭环：理解(Understand) → 建模(Model) → 生成(Generate) → 执行(Execute)
      → 验证(Verify) → 反思(Reflect) → 重写(Rewrite) → 回到对应环节。

默认采用“零新增依赖”方案：
- 求解器：已内置的 scipy.optimize（linprog / milp，底层 HiGHS）
- LLM 客户端：已内置的 httpx 直连 DeepSeek 的 OpenAI 兼容接口
"""

__version__ = "0.1.0"

# 评测协议版本：**任何会改变「同一输入 → 同一判定/行为」的改动都必须递增**。
# 例如：IR 可表达的模型范围、验证器的判据、反思/重写的路由规则、Prompt 中影响
# LLM 输出形态的部分。仅重构、注释、报表脚本改动不需要递增。
#
# 为什么需要它：实测踩过两次坑——
#   ① sub12 的 flash 批次跑在「IR 支持字符串/多维索引」之前，排班题失败是能力缺口
#      而非模型差异，与之后 v4-pro 的批次**不可直接对比**；
#   ② 33 题排班批次跑在「验证器新增目标定义检查」之前，空目标不会被拦下，
#      闭环轨迹因此与之后的重跑不同。
# 记录版本号后，对比工具可自动拒绝不可比批次，而不是给出错误结论。
#
# 版本历史：
#   1 — 初版
#   2 — IR 支持字符串索引与多维索引（index_set）
#   3 — 验证器新增目标定义检查；目标值容差与可行性容差解耦（改为相对容差）
#   4 — 验证器新增目标系数题面依据检查（check_coefficient_grounding，无参考可判定）；
#       仅在题面确实给出多个权重时生效，题面未给信息时不判错
#   5 — 提示词分层：常驻核心（1721 字符）+ 按题面命中的领域卡片；
#       原单体提示词（3075 字符）里 5 条数据集专属规则移入卡片，
#       核心补齐实测缺失的通用规则（数值必须来自题面 / 缺失信息记入 assumptions）
#   6 — max_tokens 由 32768 提升到 65536：实测 32768 对最大排班实例是**临界值**
#       （同一题一次成功、一次截断），提升后消除该类截断失败
#   7 — 卡片触发词「宁多勿漏」：NL4OPT 离线审计发现 5 处真实漏检
#       （"N times as much as" / "N times more than" / "half as much as" /
#        "only 30%" / "a minimum of 20%"），已补齐；ratio 卡改为任何百分号都触发，
#       并在正文提醒区分「比例约束」与「收益率」。实测未改变 OR-Bench 的卡片选择。
#   8 — 三项行为变更（依据 IndustryOR 冒烟诊断）：
#       ① 疑似参考噪声不再短路：untrusted + 目标不符仍走**一次**重写尝试，
#          结果记为 suspicious_outcome，但绝不升级为通过；
#       ② 修正本地求解器**丢弃目标常数**（与 codegen/docker 不一致的静默错误）；
#       ③ 新增 fixed_cost 与 piecewise 两张卡片（触发词见 prompts.py）
#   9 — 判定修订：`converged`（重写后达到参考值）**计为通过**，status 保留
#       `suspicious_resolved` 可追溯；仅 `unresolved` 不计通过。
#       撤销 v8 的「converged 也不计通过」——实测 NL4OPT 245 中 16 题 converged
#       且全为 2 轮，判定须与 trusted_reference=True 的数据集一致（同样的证据
#       不应得到相反判定；trusted_reference 是数据集级标签，非逐题证据）。
#   10 — 剥离「答案泄露」：check_optimality 的失败 message 不再含参考答案数值
#       （原为「目标值不符：求解 X，期望 Y」）。该文本经 describe() 进入
#       reflect/rewrite 的 prompt，LLM 据此可「拟合答案」而非「修正建模」——
#       污染**所有**依赖重写的修复（含 trusted_reference=True），比 converged
#       判定更根本。修法：message 只报「目标值与参考不符（求解 X）」，
#       完整数值仍保留在 CheckResult.detail（诊断/报告/人工复核可见，不进 prompt）。
#       状态类错误（infeasible/unbounded/no_solution）保留——那是必要建模事实，
#       不泄露数值。
#   11 — 修复重复线性项导致的双后端语义分叉（P0）：
#       ① ir.py 的 LinearExpr 新增规范化 validator：合并同变量项、删零项、按变量名
#          排序（治本——表达式表示唯一，哈希/日志/codegen/solve 全统一）；
#       ② solver/engine.py 的目标系数与约束行由赋值 `=` 改为累加 `+=`（止血——
#          万一有路径绕过规范化也不分叉）。
#       历史缺陷：`c[idx[var]] = coef` 赋值覆盖，重复项只取最后一项，而 codegen/PuLP
#       累加（2*x+3*x=5*x），同一 IR 两后端得到不同模型。
#   12 — 收紧 IR 数值与边界校验（P0）：契约错误在入口拒绝，而非推迟成执行错误
#       （后者会干扰 L1/L2/L3 归因）：
#       ① 所有数值字段（系数/常数/边界/rhs）必须 math.isfinite；
#       ② lower <= upper；
#       ③ binary 可行域必须与 [0,1] 相交；
#       ④ 约束名唯一；
#       ⑤ 拒绝空白/超长（>200）变量名。
#   13 — codegen 变量名安全（P0）：IR 变量名与 Python 标识符解耦。
#       Python 名改用安全标识符 `_v{i}`，PuLP 显示名保留原始 IR 名，结果回填显式
#       「原始实例名 → 安全表达式」映射（不依赖 PuLP 可能改写的 name）。
#       历史缺陷：声明侧直接 `{var.name} = ...`，变量名若为 `x-y`/`class`/`1x`
#       会生成 SyntaxError 落入 L1，而 local 后端按字符串名照常工作 → 后端分叉。
#   14 — 验证器拒绝「不完整/非有限」的解向量（P1）：可行性检查先做解向量完整性
#       校验——声明变量必须全部出现（缺失明确失败）、不得出现未声明变量（多余明确
#       失败）、每个值必须 math.isfinite（拒绝 NaN/Inf）。不再用 `x.get(var, 0.0)`
#       把缺失变量静默当 0（会漏掉「下界应为 1 却缺值」这类错误），NaN 在比较里
#       不触发失败、`round(nan)` 还会抛异常，均须在约束/边界/整数检验之前拦截。
#   15 — 抽出统一 LLM 响应保护层（P1）：`llm.complete_structured(llm, messages, stage)`
#       统一负责截断/空输出检测 + 一次低推理强度重试 + 带 stage 名的明确错误。
#       understand/rewrite/reflect 三阶段共用（此前 understand 有、rewrite 直接
#       extract_json(resp.text)、reflect 空/异常输出默认落 L2——同一次模型截断在
#       不同阶段表现不对称，污染 L1~L3 归因）。
#       反思 JSON 非法且无关键词命中 → 抛 `ReflectionParseError`，编排器据此转 L4
#       人工（记录 reflection_parse_error），**绝不盲目默认 L2**。
#   16 — 缓存键版本化与语义隔离（P1）：引入 `CacheContext`（protocol_version /
#       prompt 内容指纹 / llm_model / llm_sampling / backend / solver_config /
#       cache_namespace / cache_version），其指纹作为前缀并入三级缓存 key。
#       此前 key 只哈希内容本身，改 prompt、换模型、改 codegen/求解器参数会**静默命中
#       旧结果**；现任何语义上下文变化都使旧 key 天然 miss。config 新增 `cache.namespace`
#       / `cache.version` 作人工逃生阀。注：protocol_version 进入 key 意味着本版本
#       递增会**整体失效一次存量缓存**（清空旧语义，属预期）。
#   17 — 目标系数依据检查改为「告警优先」分级（P2）：`check_coefficient_grounding` 不再
#       一律硬失败，而是按证据强度分级——
#       ① 强证据（`structured_weight_map` 能解析出「对象↔权重」的 JSON 结构）→ 系数全
#          相同仍硬失败（blocking=True）；
#       ② 弱证据（仅 `stated_weights` 关键词邻近扫描，无法建立对象↔权重对应）→ 系数全
#          相同仅告警（blocking=False，不阻断 all_pass）。
#       `CheckResult` 新增 `blocking` 字段；`all_pass` 只计阻断项，`describe` 渲染
#       PASS/FAIL/WARN 三态。目的：弱证据启发式不再「阻止本应接受的模型」。
#   18 — 新增目标值一致性检查（P0，闭合「执行→验证」可信边界）：
#       `verifier.check_objective_consistency` 由 IR 重算 Σ coef·x + constant（先校验
#       有限性），与求解器报告的 objective_value 用独立容差（feas_tol）比对，作为
#       **blocking** 检查接入 verify()，先于 check_optimality。此前 check_optimality 只
#       拿报告值与参考答案比、check_feasibility 只核约束/变量——谁都不重算目标值，于是
#       Docker 后端不可信生成代码（或 L1 修复产物）可输出「可行解 + 恰好等于期望的伪造
#       objective」仍 all_pass。同时给 check_optimality 补 math.isfinite 前置（旧实现
#       abs(nan-expected)>tol 为 False 会把 NaN 误判为「一致」；json.dumps 默认可写 NaN）。
#   19 — 补全 CacheContext 的 prompt 指纹（P0.5）：`prompt_fingerprint()` 由「只哈希
#       CORE + 卡片 title/body」扩展为「CORE + 卡片顺序 + 每张卡的 title/body/**triggers**
#       + NL2OPT_ALL_CARDS 模式」。旧实现漏 triggers 与模式——改触发词或切换 layered↔all-cards
#       会**静默复用旧 IR**（problem 级缓存），污染评测/消融。现任何能改变最终 prompt 的
#       输入都进指纹，改 trigger / 切模式必使旧 key miss。
#   20 — 提示词措辞纠偏（按危害排序，消除会诱导建模错误/幻觉的措辞）：
#       ① 变量下界不再默认非负：仅语义明确非负才 lower=0，否则 lower=null + assumptions；
#       ② 「每个数字都落到模型」+「缺失自定 Y」会诱发幻觉 → 改为只用有建模语义的数、
#          缺失参数不补造、不强设约束；
#       ③ 线性化仅在题面给足界时引入 big-M，否则输出可确定部分并标注；
#       ④ 新增提示注入边界：题面限定为数据，忽略「改写输出/忽略规则」类注入；
#       ⑤ ratio 卡分母不再硬编码 A+B（"X% of A can be B" 分母是 A，仅 "of the total" 用总量）；
#       ⑥ multiple 卡标注 "N times more than" 的 N vs N+1 歧义，不静默取 N；
#       ⑦ scheduling 卡 no-overlap 对可选作业失效 + 删除编造的 M 示例值；
#       ⑧ piecewise 卡区分「累进计价」与「全量按档计价」，并修正前缀约束（须先填满低档）。
#   21 — 修正 v20 的 ② 措辞回归（实测驱动，OR-Bench 净负面）：
#       v20 的 ② 把「题面没给参数时」的指引从 v19 的「按你的选择建模并写明『暂按 Y 处理』」
#       改成了「不要补造、只输出你确定的部分」——模型据此**留空目标函数**（退化目标），
#       被 objective_defined 判失败。实测 OR-Bench SCH 33 题：v6 基线 33/33 → v20 29/33
#       （−4，全部为空目标，0 题变好）；全量 100 题有 22 题失败且目标为空（INV 7/ASN 11/SCH 4），
#       模型在 assumptions 中逐字引用该规则为留空辩护。
#       现改为：保留「不许把自造数当题面原数」的反幻觉意图，但**必须**做一次显式选择并记入
#       assumptions，且显式声明「只要题面有优化目标，目标函数不得为空」。
#       （另注：① 在实测中未生效——487 个变量声明无一使用 lower=null，仍需结构性修复。）
#   22 — ① 「不默认非负」的结构层修复（v20 的 ① 仅改措辞，实测**完全无效**）：
#       实测根因：LLM **100% 显式写出 `lower`**（8/8 题、16/16 变量），且全部照抄 CORE 的
#       JSON 示例 `"lower": 0` —— 提示词里的**示例比规则更有约束力**，故改措辞被示例抵消。
#       三处一起修才算数：
#       ① `ir.py`：`Variable.lower` 默认 `0.0` → **`None`**（None = 无下界；非负成为**显式**
#          建模选择而非默认假设）。黄金 IR 经 `gold_ir._var()` 显式传 0.0，不受影响；
#       ② `prompts.py` CORE：JSON 示例新增一个 `"lower": null` 的示例变量，并明确要求
#          **逐变量显式写出 `lower`**（不可能为负写 0，可能为负写 null）；
#       ③ `codegen._bounds_args`：**修复由此暴露的双后端分叉**——旧实现在 `lower is None`
#          且类型非 continuous 时强制 `lowBound=0`，而 engine 用 `-inf`，同一 IR 两后端得到
#          不同模型（此前因默认值为 0.0 而极少触发）。现统一为「None = 无下界」，
#          并对齐 binary 的 [max(lo,0), min(up,1)] clamp。
#       验证：`check_backend_parity.py` 新增 `free-lower-bound` 样本（最优解必须为负：
#       minimize c+k s.t. c>=-7, k>=-5 → -12），本地 HiGHS 与 docker/CBC **均给 -12**；
#       端到端「b 必须为负」的题正确求解（旧行为下该题 infeasible）。
#   23 — 修复 suspicious 终止路径的「报告值与 final_ir 不一致」bug：
#       旧实现在 `if suspicion is None:` 分支里只用**第 1 轮**的 sol 设置
#       result.objective_value / variables / errors，后续轮次不再更新；而 final_ir
#       始终是最后一轮的 IR。于是 unresolved 终止时，报告的 objective_value 与
#       final_ir 的重解值不一致。实测影响 6 题（NL4OPT v20 5 题、IndustryOR 1 题，
#       全部 rounds=2 + status=suspicious）：
#         industryor-005 报告 18.9566，实际最终模型最优值 18.9857；
#         nl4opt-125 报告 431.41，实际 440.0；nl4opt-138 报告 40.0，实际 43.81；
#         nl4opt-143 报告 143.0，实际 108.0；nl4opt-163 报告 330.0，实际 425.0。
#       **判定不受影响**（每轮 check 都用当轮 sol，success/status 正确），
#       但报告值会误导分析（曾据此误判 NL4OPT 退步题的具体数值）。
#       修复：unresolved break 前用本轮 sol 刷新三字段 + errors；并加回归测试。
#   24 — fixed_cost 卡片补齐「两种固定成本」的语义区分（实测驱动）：
#       旧卡片只教「固定成本是开启即付一次 → 引入 0-1 变量 y + big-M」，把「固定成本」
#       窄化为 setup cost。IndustryOR-023 的题面是「weekly fixed costs for the three
#       special equipment」（每期固定支出，与产不产无关），模型照卡片建出 y 开关、
#       只收实际生产那条线的固定成本 → 得 24000，而参考为 21500。
#       验证：把固定成本全额（4500）作目标常数、去掉 y，则最优解 Shirt 100 + Casual 200
#       （labor 1500 满、material 1600 满）→ **21500 精确命中参考值**。
#       现卡片区分：(a) 开启费/setup → y + big-M；(b) 期间固定成本（weekly/monthly/annual，
#       无「若生产才发生」条件）→ 直接作目标常数，**不引入 y**；并给出误建症状
#       （只开最赚钱的一条线、目标值偏高）。
#   25 — IR 引用完整性保障（VRP / 网络流类变量族，实测驱动）：
#       问题（industryor-036）：VRP 题按节点对展开的变量族规模是 n²，LLM 生成的 index_set
#       漏了部分组合，约束 deg_in_19 引用 x_8_19 未声明 → IR 校验失败；而**首次 understand
#       失败即 return（rounds=0）**，闭环从未启动，也没有第二次机会。
#       两层改进：
#       ① 校验层（ir.py `_check_references`）：把「未声明引用」**一次报全**（旧实现只报
#          第一个），并把每个缺失实例归因到变量族与缺失的索引组合
#          （如「变量族 `x` 缺索引 `8_19`」）并指出引用位置（目标/哪个约束）；
#       ② 提示词（CORE 规则 2）：新增「引用完整性自查」——每个被引用实例都必须已声明；
#          n² 规模的族不要只列一部分；若判断列不全，**不要硬写半份 index_set**，
#          改为只建模能完整列出的部分并写进 assumptions。
#       另：tests/test_ir_validation.py 新增 3 个用例锁住该路径（此前**完全没有测试覆盖**）。
#   26 — 固定支出必须落进 `objective.expr.constant`（实测驱动，IndustryOR-001）：
#       问题：001 的 assumptions 声称「现有 50 名工人工资 50*360*8=144000 已作为常数计入」，
#       但 `objective.expr.constant` 实际为 **0** —— 「声明与实现不一致」。模型大概认为
#       「常数不影响 argmin」就省掉了，但**评测比较的是目标值**：漏常数 → 75600 vs 参考 219888
#       （差 65%）；补上 144000 → 219600，差 0.13%。
#       修复：
#       ① CORE 新增规则 6：与决策无关的必付金额（固定工资/租金/折旧/已发生投入）必须写进
#          `objective.expr.constant`，并显式说明「常数不影响最优解但影响目标值」这一陷阱；
#          要求输出前自查「assumptions 提到的每个固定金额是否真的落到 constant 或约束 rhs」；
#       ② `fixed_cost` 卡片 (b)（期间固定成本）强化为「必须真的写进 constant 字段」并附本案例。
#   27 — 目标规划题的「答案不是目标函数值」（实测驱动，IndustryOR-029）：
#       题面是 p1/p2/p3 优先级 + 「至少多少不能满足 P3」，正确答案是 d3_minus=2500。
#       旧模型：目标 = 10000*d2_minus + p3_unmet。用大 M 实现字典序，**解其实是对的**
#       （d3_minus=2500），但 `objective` 报的是加权和 10,003,000 —— 与参考不可比；
#       且答案变量选错（用 p3_unmet=3000 而非 d3_minus=2500，二者差常数 500）。
#       两层修复：
#       ① 评测层：`expected` 支持可选 `answer_variable`。指定后 `check_optimality` 比较
#          **该变量在解中的值**，而非目标函数值；变量缺失时给出可操作错误
#          （提示偏差变量须命名 `d{k}_minus` / `d{k}_plus`）。
#       ② 提示词 CORE 新增规则 7：目标规划题须用 `d{k}_minus`/`d{k}_plus` 命名偏差变量、
#          用大 M 实现字典序，并明确「答案通常是某个偏差变量，objective 是加权和、无业务
#          含义」；禁止另造 `unmet = 总数 − 达成量`（与 d{k}_minus 差一个常数）。
#       数据：`convert_industryor.py` 给 029 标注 `answer_variable: "d3_minus"`。
#       测试：`tests/test_verifier_solution_integrity.py` 新增 2 个用例。
#   28 — IndustryOR 参考答案人工核定（005 改正、040 标歧义）：
#       005（corrected）：原 en_answer=18.6943 **不是最优**——它对应 okra=498.86g +
#         carrot=101.14g，花费 $14.184，**预算还剩 $0.816 未花**；把这笔钱换成 okra
#         可继续提高纤维。最优解 okra=557.142857g + carrot=42.857143g 正好花满 $15，
#         纤维 = 1329/70 ≈ 18.9857142857（手算复核：0.026·a + 0.012·(600−a) = 15）。
#       040（ambiguous）：参考值来源不明，判「参考口径有误 / 题面 GDP 口径歧义」：
#         含投入产出平衡（题面明写各产品生产消耗，属严格建模）得 36,288,567；
#         去掉平衡后 y 自由则**无界**、加 y≤x 得 83,000,000；4 种 GDP 口径 ×
#         3 种人年→人月换算共 12 组合均复现不了 43,090,833。标记但不改值。
#       同类标记（值不改，一律**不计入模型能力**）：
#       002（ambiguous）目标口径歧义——「每年 5 人 × 2 年」(175) vs「每架 5 人」(125)，
#         以及第 1 年训练的飞行员第 2 年能否立即计入；参考 125 只在最宽松读法下自洽。
#       011（ambiguous）参考 3867 反解对应解不在可行域任何顶点上（LP 最优必在顶点）；
#         且 p3 最优点与利润最优点重合，正确实现 p3 不改变最优解（仍 4078.33）。
#       065（inconsistent）题面自相矛盾——A/B 均满负荷 + x≥80 + y≥50 时 infeasible；
#         参考 30500 对应「B 不设约束」的解 (100,50)，但该解需 B 用 250h > 题面的 150h。
#       ⇒ 可计分集合 = 97 − 4（ambiguous/inconsistent）= **93 题**；corrected 的 005 计入。
#       两项都保留 `expected.reference_original` 供追溯，依据写在 `reference_note`；
#       源头是 `scripts/convert_industryor.py` 的 `REFERENCE_FIXES`，重新生成数据即生效。
#   29 — 规则副作用修正（v28 全量跑暴露的 3 处回退，均非模型能力问题）：
#       ① 027（v24 通过 → v28 失败，1146.4142 → 1146.5665）：题面未明说整数，模型选了
#          continuous。修：CORE 规则 1 增补「**计数类量默认 integer**（件/台/人/辆/批次），
#          除非题面明说可分割」。验证：027 → 1146.4142 通过（1 轮）。
#       ② 048（v24 通过 → v28 失败，11250 → 25000 再 → 0）：题面用 p1–p4 **编号**但未声明
#          优先级，且另有明确优化方向（最大化利润）；规则 7 误判成目标规划。修：
#          (a) 触发条件改为「明确声明了优先级顺序」；(b) 明说「p1/p2/p3 编号本身不是优先级
#          声明」；(c) 增补「**先判断 p1–p4 是约束还是目标**：若有 maximize/minimize 方向，
#          主目标是那个量、p1–p4 只作约束」。验证：048 → 11250 通过（2 轮）。
#       ③ 036（v25 单题通过 175.374 → v28 失败 rounds=0）：VRP 420 条弧超出单次输出能力，
#          IR 被截断成畸形 JSON。修：规则 2 明确「**禁止用汇总变量代替整个族**」。
#          仍不稳定——属**规模限制**，非规则可解，记为已知限制。
PROTOCOL_VERSION = 29

__all__ = ["__version__", "PROTOCOL_VERSION"]
