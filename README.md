# Embodied AI Literature Assistant

这是一个面向具身智能/多模态/机器人论文调研的多智能体学习助手原型。当前主线是用 LangGraph 编排调研流程，用 PaperQA2 对选中的 PDF 做证据抽取和带引用回答。

## 当前功能

- `Router/PlannerAgent`：判断用户问题类型，并规划后续工作流。
- `Planner memory tool`：Planner/Supervisor 在正式规划前先读 `EPISODE_INDEX.jsonl / PAPER_CARDS.jsonl` 轻量索引，由 LLM 决定是否需要调用本地 `memory_get` 工具读取 `EPISODES.md / PAPERS.md` 的详细记忆块，再基于召回结果规划后续工作流。
- `ResearchPlanningAgent`：根据用户问题、thread memory 和可选 web scouting context 生成信号驱动的调研视角、子问题和搜索 query；每个视角会记录 `signal_sources`，便于检查它来自用户问题、memory、web 线索还是领域知识。
- `ExpertResearchAgent`：统一调度 web search、paper search、PDF 缓存和 PaperQA reader。
- `PaperSearchAgent`：由 LLM 规划工具化 search plan，按需调用 arXiv/OpenAlex/OpenReview/Crossref、作者主页、publication page、Google Scholar profile 查询、DBLP/OpenReview profile 等搜索方式；如果候选或 PDF 不足，会进行一轮 targeted follow-up search。
- `PaperTriageAgent`：从候选论文中选择值得精读的 PDF。
- `PaperQA ReaderAdapter`：默认使用显式 `PaperSearch -> GatherEvidence` 工具链，只让 PaperQA 抽 PDF 证据；最终回答统一交给 `SynthesisAgent` 组织。PaperQA 原生多轮 agent 仍可作为可选对照模式。
- `SynthesisAgent`：综合 web evidence 与 PaperQA evidence，生成最终中文回答。

目前已经可以跑文献调研类问题，例如：

- 2026 年 VLA 机器人操作方向的代表论文和技术趋势。
- 某位老师/团队最近研究方向及代表论文。

## 目录结构

```text
embodiedai_kb/
  data_collection/          # 元数据采集脚本和评分逻辑
  search/                   # 本地 metadata search
  paper_search/             # 在线 multi-source paper search connector
  langgraph_workflow/       # LangGraph 多智能体工作流
scripts/
  ask_literature.py         # 早期 PaperQA/metadata demo 入口
  ask_literature_langgraph.py # 当前主入口
third_party/paper-qa/       # PaperQA2 源码
data/
  metadata/                 # 运行记录和论文 metadata
  pdf_cache/literature/     # 下载后的 PDF 缓存
  paperqa_agent_runs/last/  # 每次 PaperQA reader 的本轮 PDF 目录
```

## 环境激活

当前机器上项目使用 conda `base` 环境即可运行：

```bash
conda activate base
cd /home/karim/python-final-project
```

当前已验证的 Python 环境：

```text
Python 3.13.13
/home/karim/miniconda3/bin/python
```

关键包包括 `langgraph`、`litellm`、`paperqa`、`pydantic`、`httpx`。项目内的 arXiv 搜索使用自写 HTTP connector，不依赖 `arxiv` 这个 pip 包。

## API 配置

如果使用 OpenAI-compatible 中转站：

```bash
export OPENAI_API_KEY="你的 API key"
export OPENAI_BASE_URL="你的 OpenAI-compatible base url"
```

如果使用 Tavily web search：

```bash
export TAVILY_API_KEY="你的 Tavily key"
```

默认情况下，LangGraph 主程序会把 PaperQA embedding 设为 `sparse`，因此不需要额外 embedding API：

```text
--embedding sparse
```

注意：`sparse` 对中文问题检索英文 PDF evidence 不够强。后续可以改成兼容中文/英文的 embedding 模型，或在 PaperQA reader 中加入英文 evidence question。

## 主运行命令

当前通用跑法使用 LangGraph 入口。整体是混合式架构：全局由 LangGraph supervisor 做 plan-and-execute，局部在 PaperSearchAgent 和 PaperQA reader 中使用 search / evidence 的 ReAct-like 观察-补搜/补证循环，以控制 API 成本。
默认链路是：

`Planner -> ResearchPlanningAgent -> ExpertResearchAgent -> PaperSearchAgent -> PaperTriageAgent -> PaperQA evidence extraction -> SynthesisAgent`

```bash
python scripts/ask_literature_langgraph.py "你的问题" \
  --include-frontier \
  --paperqa-k 6 \
  --agent-search-count 6 \
  --synthesis-max-tokens 6000
```

如果问题明确限制年份，再加年份范围：

```bash
python scripts/ask_literature_langgraph.py "帮我调研 2026 年 VLA 机器人操作方向的最新代表性论文，总结主要技术趋势和代表性工作。" \
  --include-frontier \
  --year-from 2024 \
  --year-to 2026 \
  --paperqa-k 6 \
  --agent-search-count 6 \
  --synthesis-max-tokens 6000
```

只测试 planner/search/triage，不下载和精读 PDF：

```bash
python scripts/ask_literature_langgraph.py "你的问题" --dry-run
```

使用轻量 thread memory 记录本轮运行摘要：

```bash
python scripts/ask_literature_langgraph.py "你的问题" --thread-id vla_demo
```

这会读写 `data/memory/vla_demo.jsonl`。JSONL 中保存较完整的运行摘要，方便回看和 debug；回答摘要会保留较长片段，避免长回答只剩一小段无法追溯。
同时会维护一个 OpenClaw-style 的轻量 Markdown memory store。全局记忆始终加载；每个 thread 的 episode/paper 记忆单独隔离：

```text
data/memory/global/
  IDENTITY.md   # 助手角色、项目约束、用户偏好
  MEMORY.md     # 稳定事实/长期语义记忆
  LESSONS.md    # 搜索/调研经验，给 LLM 参考，不当硬规则

data/memory/threads/{thread_id}/store/
  EPISODES.md          # 当前 thread 每轮运行的详细摘要、query、PaperQA 状态与答案片段
  PAPERS.md            # 当前 thread 每轮 Paper Card Collection：论文卡片、链接、来源问题、PaperQA evidence
  EPISODE_INDEX.jsonl  # 当前 thread 机器读 episode 索引：问题、摘要、主题、论文 id、detail_ref
  PAPER_CARDS.jsonl    # 当前 thread 机器读论文卡片索引：问题、核心想法、PDF 状态、证据 id、链接
```

运行时会把全局 Markdown store、当前 thread JSONL 和当前 thread Markdown store 整理成轻量 `memory_packet`，再格式化成很短的 `memory_context`。Planner/Supervisor 会先看当前 thread 的 `EPISODE_INDEX.jsonl / PAPER_CARDS.jsonl` 这些轻量索引，再按需调用本地 `memory_get` 读取当前 thread 的 `EPISODES.md / PAPERS.md` 详细块，并把召回内容追加到 `memory_context` 后再规划后续工作流。它用于解析“这篇论文 / 这个方向 / 继续 / 把刚才论文链接发我”等追问；它不会被当作 PaperQA 或 Web 证据引用。加载 thread 时也会把 JSONL 中已有但尚未进入当前 thread store 的记录回填到 `EPISODES.md` 和 `PAPERS.md`。
`PAPERS.md` 现在按轮次写入 `Paper Card Collection`：每一轮一个集合，集合下每篇论文都是一张 paper card，包含 `paper_id/card_id`、来源问题、摘要/核心想法、链接、PDF 状态、角色、置信度，以及能匹配到的 `PaperQA evidence`。`EPISODE_INDEX.jsonl` 和 `PAPER_CARDS.jsonl` 是后续 memory_search/memory_get 的基础索引：agent 可以先看轻量 index，再按 episode/card 追到 `EPISODES.md` / `PAPERS.md` 的详细内容。
当前 `EPISODE_INDEX.jsonl` 的 `summary` 由 `SynthesisAgent` 在同一次最终回答调用里附带生成：模型输出会被解析成用户可见的 `final_answer` 和只写入记忆的 `episode_summary`，不会为了摘要再发起第二次 LLM 调用；如果解析失败，才退回到截断版回答摘要。可用 `--episode-summary-max-tokens` 或环境变量 `LITERATURE_EPISODE_SUMMARY_MAX_TOKENS` 给摘要长度一个预算提示。`PAPER_CARDS.jsonl` 来自 selected paper metadata、abstract 和 PaperQA evidence contexts。旧 JSONL 记录如果当时没有保存 abstract/evidence，回填出来的 card 会比较稀疏；新运行会自动写入更完整的 card。
如果只是让系统“把刚才/这些/相关的论文链接发给我”，目标是让 Planner 参考 memory 中的上一轮 selected papers 判断是否可以直接回答，避免不必要的重新检索；当前仍需要继续优化 Planner 对 memory 的使用。
ExpertResearchAgent 现在还会先运行一个本地 `memory_paper_tool`：它从当前 thread 的 `PAPER_CARDS.jsonl` 和 JSONL run record 中整理已经选过/读过的论文 PDF，再由 LLM 判断这些记忆论文是否足够回答当前问题。如果足够，就直接复用这些本地 PDF 交给 PaperQA，不再重新跑外部 paper search；如果不够，它会把记忆论文和新搜索候选一起交给 PaperTriageAgent 选择。

如果想在终端里用短对话形式运行，可以启动 REPL：

```bash
python scripts/chat_literature.py \
  --thread-id vla_chat \
  --compact \
  --include-frontier \
  --year-from 2024 \
  --year-to 2026 \
  --paperqa-k 6 \
  --agent-search-count 6
```

REPL 中输入 `:q` 退出，输入 `:reset` 会让下一轮清空当前 thread memory。
运行时默认会打印轻量进度，例如 Planner、ResearchPlanningAgent、PaperSearchAgent、PaperTriageAgent、PaperQAReader 和 SynthesisAgent 的 start/done，方便判断是否卡在搜索、筛选、PDF 缓存还是最终回答。

如果想用网页前端演示，可以启动 Flask Web UI：

```bash
python scripts/web_app.py --host 127.0.0.1 --port 7860
```

然后打开：

```text
http://127.0.0.1:7860
```

这个前端不会重写后端逻辑，而是把 `scripts/ask_literature_langgraph.py` 包成一个聊天窗：主区域是对话，侧边栏可以切换历史 thread 或新建聊天，并折叠展示常用参数和高级参数。网页默认启用 frontier 本地论文库；页面只暴露 arXiv 本地库、普通 web search、Reader 模式等常用开关。每条回答只显示正文，回答下方展示选中论文/PDF、PaperQA evidence 和运行指标；MemoryPaperTool、Expert trace、PaperQA trace、终端日志和 Run JSON 不再默认出现在聊天框里。网页会用轻量 Markdown 渲染和 MathJax 显示公式，默认回答 token 预算是 9000。当前版本是同步请求，长问题会等待数分钟；适合本地演示和 debug。网页默认单轮超时是 420 秒，代理/网络卡死时会比终端默认更快返回错误。

## 常用参数

- `--include-frontier`：加入 frontier/latest 论文 metadata 库。
- `--include-arxiv`：加入本地 arXiv metadata 库。
- `--year-from 2024 --year-to 2026`：限制论文年份。
- `--paperqa-k 6`：最多选多少篇 PDF 给 PaperQA 精读。
- `--paper-triage-candidate-limit 30`：分批筛选后，给最终 PaperTriageAgent 复看的最高分候选数。
- `--paper-triage-abstract-max-chars 400`：最终复看时每篇候选摘要的最长字符数；设为 0 可传完整摘要，但会非常耗 token。
- `--paper-triage-screen-batch-size 20`：PaperTriageAgent 第一阶段每批给多少篇候选打分。
- `--paper-triage-screen-top-n 30`：第一阶段打分后，把前多少篇交给最终选择。
- `--paper-triage-screen-abstract-max-chars 350`：第一阶段打分时每篇候选摘要的最长字符数。
- `--agent-search-count 6`：PaperQA 每次本地 paper search 返回多少篇。
- `--paper-search-web-queries 5`：PaperSearchAgent 每轮最多额外发送多少条 web/profile 查询，用来找作者主页、publication page、Google Scholar profile 入口、项目页等网页线索。
- `--paper-search-loop-iterations 1`：PaperSearchAgent 内部最多搜索几轮；默认只跑初始 search plan 以节省 token。遇到人名/团队/冷门问题时可手动调到 2，让 LLM 根据候选/PDF 缺口生成补搜工具 query。
- `--paper-search-min-pdf-candidates 3`：PaperSearchAgent 至少希望拿到多少篇带 PDF 的在线候选；低于该值会触发补搜，直到达到轮数上限。
- `--paper-search-min-candidates 8`：PaperSearchAgent 至少希望拿到多少篇在线候选；低于该值会触发补搜。
- `--academic-paper-sources arxiv,openalex,openreview`：academic connector 默认搜索源。`crossref` 仍可手动加入，例如 `--academic-paper-sources arxiv,openalex,openreview,crossref`，但默认关闭，因为它更适合精确 title/DOI fallback，做人名/主题泛搜时容易返回无关论文。
- `--academic-paper-request-delay 1.5`：academic connector 请求错峰间隔。遇到 arXiv/OpenAlex/OpenReview 限流时可以调到 2-3，但搜索会更慢。
- `--academic-paper-max-workers 2`：academic connector 最大并发请求数。遇到频繁 429/503/403 时建议保持 1-2。
- `--disable-paper-search-loop`：关闭 PaperSearchAgent 内部补搜循环，只执行第一轮搜索。
- `--paperqa-reader-mode explicit-tools`：默认 reader 模式。显式调用 PaperQA 的 `PaperSearch/GatherEvidence`，让证据抽取更可控。
- `--paperqa-reader-mode paperqa-agent`：可选对照模式。先跑 PaperQA 原生多轮 agent，失败或证据不足再 fallback。
- `--paperqa-reader-mode agent-only`：只跑 PaperQA 原生 agent，不 fallback。
- `--paperqa-answer-mode evidence-only`：默认模式。跳过 PaperQA 的 `GenerateAnswer`，只返回结构化 evidence contexts 给 `SynthesisAgent`。
- `--paperqa-answer-mode answer`：可选对照模式。让 PaperQA 先生成一版带引用草稿，再交给 `SynthesisAgent` 综合。
- `--paperqa-min-evidence-count 4`：原生 agent 至少抽到多少条 positive evidence 才认为证据足够。
- `--paperqa-min-relevant-papers 1`：原生 agent 至少覆盖多少篇相关论文才认为证据足够。
- `--paperqa-title-query-count 6`：先用选中论文标题触发 PaperQA 本地检索，帮助 PDF 进入 evidence 状态。
- `--paperqa-per-paper-evidence-count 4`：对前 N 篇 selected paper 逐篇补一次 `GatherEvidence`；设为 0 可关闭，设成 `--paperqa-k` 可增强覆盖但会增加 LLM 成本。
- `--evidence-k 12`：PaperQA gather evidence 检索多少个 evidence chunk。
- `--answer-max-sources 6`：最终回答最多引用多少个来源。
- `--synthesis-max-tokens 4096`：最终 `SynthesisAgent` 回答的 token 上限；如果长回答在中途截断，可以调大到 6000 或 8000。
- `--llm-timeout 180`：每次 LiteLLM/中转站请求的超时时间，避免代理或 API 卡住后无限等待。
- `--quiet-workflow-progress`：关闭 LangGraph/agent 的实时进度日志。
- `--thread-id default`：轻量 thread memory 的会话名；每个 thread 对应一个 JSONL 文件。
- `--memory-dir data/memory`：thread memory 保存目录。
- `--memory-recent-turns 3`：运行开头加载最近多少条压缩 memory 记录到 Planner / ResearchPlanning / Synthesis。
- `--memory-llm`：Planner memory tool 使用的模型；默认复用 `--router-llm` 或 `--llm`。
- `--memory-recall-max-tokens 900`：Planner memory tool 从索引中选择 episode/card id 时的输出 token 上限。
- `--memory-detail-max-chars 9000`：本地 `memory_get` 召回详细记忆块的字符上限。
- `--disable-memory-paper-tool`：关闭 ExpertResearchAgent 的本地 PDF 记忆复用工具，强制走外部 paper search/metadata。
- `--memory-paper-candidate-limit 50`：最多给 `memory_paper_tool` 查看多少张当前 thread 的论文卡片。
- `--memory-paper-record-limit 30`：最多读取多少条当前 thread 的 JSONL run record，用来给论文卡片补 `cache_path/cache_status`。
- `--memory-paper-max-tokens 1000`：`memory_paper_tool` 选择记忆论文时的 LLM 输出 token 上限。
- `--reset-memory`：运行前清空当前 thread 的 memory，然后写入本轮记录。
- `--no-memory`：关闭 memory 读写。
- `--dry-run`：只跑规划、搜索、候选选择等，不进入 PDF 精读。
- `--download-only`：下载/缓存 PDF，但不运行 PaperQA 生成回答。
- `scripts/chat_literature.py --turn-timeout 900`：终端聊天每一轮的硬超时；设为 0 可关闭。

## 运行输出在哪里

每次 LangGraph 运行记录会保存到：

```bash
data/metadata/ask_literature_langgraph_last.json
```

PaperQA 本轮 PDF 目录：

```bash
data/paperqa_agent_runs/last
```

PDF 缓存目录：

```bash
data/pdf_cache/literature
```

Thread memory 目录：

```bash
data/memory/{thread_id}.jsonl
data/memory/global/*.md
data/memory/threads/{thread_id}/store/*
```

终端中重点关注这些字段：

```text
Planner decision
ResearchPlanningAgent plan
PaperSearchAgent:
  mode=llm
  tool_queries
  search_loop=1/1 stop=...
  platform_queries
PaperTriageAgent:
  mode=llm
  selected_count
[PaperQA] LangGraph reader prepared ...
Relevant Papers=...
Current Evidence=...
```

如果 `PaperTriageAgent selected_count=0`，说明候选 PDF 没有被选中，PaperQA 不会启动。  
如果 `Paper Count > 0` 但 `Relevant Papers` 较少，说明 PDF 已进入 PaperQA，但 `GatherEvidence` 只从部分论文中抽到了和问题相关的 evidence。

## 当前已知问题

- 默认 reader 使用显式 PaperQA 工具链做 evidence extraction；PaperQA 原生多轮 agent 仍可通过 `--paperqa-reader-mode paperqa-agent` 作为对照模式。显式工具链已加入英文 evidence question 和 per-paper evidence pass 来缓解漏读，但它仍不是严格的逐页精读器。
- 在 `sparse` embedding 下，中英混合检索仍可能漏证据；遇到人名/团队/冷门方向问题时，可以提高 `--paperqa-title-query-count` 和 `--paperqa-per-paper-evidence-count`。
- `ResearchPlanningAgent` 已初步去掉固定的 `method/data/evaluation/deployment/safety` 视角模板，改成从用户问题、thread memory、web scouting context 和必要领域知识中生成 perspectives，并记录 `signal_sources`。后续还需要用多类问题验证它是否真的减少模板化角度。
- 当前 memory 已从纯 JSONL 摘要升级为 `global Markdown store + per-thread Markdown store + memory_packet + Planner memory tool`：`IDENTITY.md / MEMORY.md / LESSONS.md` 是全局 source-of-truth；`EPISODES.md / PAPERS.md / EPISODE_INDEX.jsonl / PAPER_CARDS.jsonl` 按 thread 隔离。Planner 会先看当前 thread 索引再按需读取详情。它仍不是向量数据库或知识图谱；记忆只作为上下文，不作为论文证据。
- 后续建议加入：
  - 更强的跨语言 embedding；
  - 更细的 `MemoryExtractor`，从回答中抽取长期稳定事实、用户偏好和项目经验；
  - ResearchPlanningAgent 的多轮 signal-driven perspective discovery，减少固定角度先验；
  - LearningPlannerAgent 和 IdeaThinkAgent。
