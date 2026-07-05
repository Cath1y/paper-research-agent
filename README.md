# 多智能体 arXiv 调研系统

本项目是一个面向论文调研与论文精读的多智能体系统。系统使用 LangGraph 编排多个 agent，用 arXiv / OpenAlex / OpenReview / Web Search / 本地 metadata 等来源检索论文，并调用 PaperQA 阅读 PDF、抽取证据，最后由 SynthesisAgent 生成带引用的中文回答。

老师可以通过终端或本地网页界面运行。推荐先使用终端命令验证，再打开网页演示。

## 1. 系统功能

系统支持以下任务：

- 根据用户问题自动规划调研流程；
- 从 arXiv、OpenAlex、OpenReview、网页搜索和本地论文库中发现候选论文；
- 对候选论文进行筛选，选择值得精读的 PDF；
- 下载并缓存 PDF；
- 使用 PaperQA 从 PDF 中抽取证据；
- 综合论文证据和网页证据生成最终回答；
- 支持 thread 级记忆，能够复用之前读过的论文和历史回答；
- 提供终端问答和本地网页聊天界面。

典型问题示例：

```text
帮我调研 2026 年 VLA 机器人操作方向的最新代表性论文，总结主要技术趋势和代表性工作。

帮我精读 OpenVLA 这篇论文，并调研它的相关方向。

帮我推荐几篇上海交通大学赵波老师值得读的代表论文，并说明推荐理由。
```

## 2. 工作流概览

核心链路如下：

```text
用户问题
  -> Planner / RouterAgent 判断任务需求
  -> ResearchPlanningAgent 生成调研视角和搜索问题
  -> ExpertResearchAgent 调度搜索与阅读工具
      -> MemoryPaperTool 先检查历史已读论文
      -> PaperSearchAgent 多源搜索候选论文
      -> PaperTriageAgent 筛选值得读的 PDF
      -> PaperQA Reader 读取 PDF 并抽取 evidence
  -> SynthesisAgent 综合证据生成最终回答
```

`PaperSearchAgent` 和 `PaperTriageAgent` 之间有反馈闭环：如果 TriageAgent 判断当前候选论文不够好，会把拒绝原因和缺口反馈给 SearchAgent，SearchAgent 再进行下一轮补充搜索。

## 3. 目录结构

```text
embodiedai_kb/
  langgraph_workflow/       # LangGraph 多智能体工作流
  paper_search/             # arXiv / OpenAlex / OpenReview / Crossref connector
  search/                   # 本地 metadata 检索
  storage/                  # SQLite 数据库读写

scripts/
  ask_literature_langgraph.py  # 终端主入口
  chat_literature.py           # 终端连续对话入口
  web_app.py                   # 本地网页界面
  ask_literature.py            # 早期 PaperQA 单轮入口，保留作兼容

third_party/paper-qa/        # PaperQA2 源码

data/
  db/                        # 本地 metadata SQLite 数据库，若提交包中包含则可直接使用
  metadata/                  # 运行记录、调试日志
  pdf_cache/literature/      # PDF 缓存
  memory/                    # thread 记忆
  paperqa_agent_runs/last/   # 每轮 PaperQA reader 使用的 PDF 工作目录

mini_arxiv_qa/               # 单独拆出的 arXiv + PaperQA 终端小系统
```

## 4. 环境准备

进入项目目录：

```bash
cd /path/to/python-final-project
```

激活 conda 环境：

```bash
conda activate base
```

安装 PaperQA：

```bash
python -m pip install -e third_party/paper-qa
```

如环境中缺少依赖，可补装：

```bash
python -m pip install langgraph litellm flask markdown
```

当前开发环境中已验证：

```text
Python 3.13.13
```

## 5. API 配置

### 5.1 OpenAI 官方 API

```bash
export OPENAI_API_KEY="你的 OpenAI API Key"
```

运行时可指定模型：

```bash
--llm gpt-4o-mini
```

### 5.2 OpenAI-compatible 中转站

如果使用兼容 OpenAI 接口的中转站：

```bash
export OPENAI_API_KEY="你的中转站 API Key"
export OPENAI_BASE_URL="https://你的中转站地址/v1"
```

脚本检测到 `OPENAI_BASE_URL` 后，会自动把裸模型名转成 LiteLLM 需要的 `openai/<model>` 格式。

### 5.3 Tavily Web Search，可选

如果配置 Tavily，网页搜索会更稳定：

```bash
export TAVILY_API_KEY="你的 Tavily API Key"
```

如果不配置 Tavily，系统会尝试使用 DuckDuckGo HTML 搜索作为 fallback。

### 5.4 `.env` 配置，可选

也可以在项目根目录新建 `.env`：

```bash
OPENAI_API_KEY=你的 API Key
OPENAI_BASE_URL=https://你的中转站地址/v1
TAVILY_API_KEY=你的 Tavily API Key
```

程序启动时会自动读取 `.env` 和 `.env.local`。

## 6. 终端运行

### 6.1 推荐测试命令

```bash
python scripts/ask_literature_langgraph.py "帮我调研 2026 年 VLA 机器人操作方向的最新代表性论文，总结主要技术趋势和代表性工作。" \
  --include-frontier \
  --year-from 2024 \
  --year-to 2026 \
  --paperqa-k 6 \
  --agent-search-count 6 \
  --synthesis-max-tokens 6000 \
  --llm gpt-4o-mini
```

### 6.2 精读某篇论文

```bash
python scripts/ask_literature_langgraph.py "帮我精读 OpenVLA 这篇论文，并调研它的相关方向。" \
  --include-frontier \
  --year-from 2024 \
  --year-to 2026 \
  --paperqa-k 6 \
  --agent-search-count 6 \
  --synthesis-max-tokens 7000 \
  --llm gpt-4o-mini
```

### 6.3 只测试规划、搜索和筛选，不进入 PDF 精读

```bash
python scripts/ask_literature_langgraph.py "帮我调研 VLA 机器人操作方向" \
  --include-frontier \
  --dry-run
```

### 6.4 终端连续对话

```bash
python scripts/chat_literature.py \
  --thread-id demo_chat \
  --compact \
  --include-frontier \
  --year-from 2024 \
  --year-to 2026 \
  --paperqa-k 6 \
  --agent-search-count 6 \
  --llm gpt-4o-mini
```

REPL 中：

```text
:q      退出
:reset  下一轮清空当前 thread memory
```

## 7. 网页界面运行

启动 Flask 网页端：

```bash
python scripts/web_app.py --host 127.0.0.1 --port 7860
```

浏览器打开：

```text
http://127.0.0.1:7860
```

网页端功能：

- 左侧切换或新建 thread；
- 主区域以聊天窗口形式显示问题和回答；
- 回答下方展示选中论文、PDF、PaperQA evidence 和运行指标；
- 默认启用 frontier/latest 论文库；
- 高级参数可在侧边栏调整。

网页端是同步请求，复杂论文调研可能需要数分钟。如果超时，可优先使用终端命令调试。

## 8. 常用参数

- `--include-frontier`：启用最新/热点论文 metadata 库。
- `--include-arxiv`：启用本地 arXiv metadata 库。
- `--year-from 2024 --year-to 2026`：限制论文年份。
- `--paperqa-k 6`：最多选择多少篇 PDF 给 PaperQA 阅读。
- `--agent-search-count 6`：PaperQA 本地检索每次返回多少篇。
- `--paper-search-triage-rounds 2`：PaperSearchAgent 与 PaperTriageAgent 最多反馈循环几轮。
- `--paper-search-triage-min-selected 1`：TriageAgent 至少选中多少篇后可以停止补搜。
- `--paper-search-web-queries 5`：PaperSearchAgent 每轮最多发送多少条网页搜索 query。
- `--academic-paper-sources arxiv,openalex,openreview`：在线学术搜索源。
- `--academic-paper-request-delay 1.5`：学术平台请求间隔。若遇到限流，可调到 `3` 或 `5`。
- `--academic-paper-max-workers 2`：学术平台并发请求数。若遇到 429/503/403，建议设为 `1`。
- `--paperqa-reader-mode explicit-tools`：默认 PaperQA reader 模式，显式调用检索和 evidence 工具。
- `--paperqa-reader-mode paperqa-agent`：先尝试 PaperQA 原生 agent，再 fallback。
- `--paperqa-answer-mode evidence-only`：默认只让 PaperQA 抽 evidence，最终回答交给 SynthesisAgent。
- `--paperqa-per-paper-evidence-count 4`：对前 N 篇论文逐篇补充 evidence。
- `--synthesis-max-tokens 6000`：最终回答长度上限。
- `--llm-timeout 180`：单次 LLM 请求超时。
- `--dry-run`：只跑规划、搜索、筛选，不下载和阅读 PDF。
- `--download-only`：下载 PDF 后停止，不生成回答。
- `--thread-id demo`：指定 thread 记忆名称。
- `--reset-memory`：运行前清空当前 thread 记忆。
- `--no-memory`：关闭记忆读写。

查看全部参数：

```bash
python scripts/ask_literature_langgraph.py --help
```

## 9. 输出与调试

### 9.1 主运行记录

每次终端运行会写入：

```bash
data/metadata/ask_literature_langgraph_last.json
```

网页端每次请求会写入：

```bash
data/metadata/web_runs/{run_id}.json
```

### 9.2 Paper Search 调试日志

系统会额外保存一份专门用于检查搜索和筛选的日志：

```bash
data/metadata/paper_search_logs/last.json
```

网页端每次请求会生成独立日志：

```bash
data/metadata/paper_search_logs/{run_id}.json
```

快速查看 Search/Triage 反馈闭环：

```bash
python - <<'PY'
import json
p = "data/metadata/paper_search_logs/last.json"
s = json.load(open(p, encoding="utf-8"))
print(json.dumps(s.get("paper_search_trace", {}).get("search_triage_loop", {}), ensure_ascii=False, indent=2))
PY
```

### 9.3 PDF 与记忆目录

PDF 缓存：

```bash
data/pdf_cache/literature/
```

当前 PaperQA 工作目录：

```bash
data/paperqa_agent_runs/last/
```

Thread memory：

```bash
data/memory/{thread_id}.jsonl
data/memory/global/
data/memory/threads/{thread_id}/store/
```

## 10. Metadata 数据说明

系统同时支持在线搜索和本地 metadata 检索。若本地数据库存在，默认会使用：

| 本地库 | 默认使用 | 用途 |
|---|---:|---|
| `data/db/topconf_papers.sqlite` | 是 | 顶会/会议论文主库 |
| `data/db/frontier_papers.sqlite` | 命令行需 `--include-frontier`，网页默认启用 | 最新/热点论文补充库 |
| `data/db/papers_recent_3y.sqlite` | 需 `--include-arxiv` | 近三年 arXiv 相关论文补充库 |

如果缺少本地数据库，系统仍可通过在线 arXiv/OpenAlex/OpenReview/Web Search 获取候选论文，但检索覆盖会受网络和 API 限流影响。

### 10.1 Metadata 数据索引

本项目的 metadata 分为两类：

| 类型 | 本地路径 | 是否运行必需 | 说明 |
|---|---|---:|---|
| SQLite 运行库 | `data/db/topconf_papers.sqlite` | 是 | 默认主库，包含顶会/会议论文 metadata |
| SQLite 运行库 | `data/db/frontier_papers.sqlite` | 推荐 | 最新/热点论文库，`--include-frontier` 或网页端默认使用 |
| SQLite 运行库 | `data/db/papers_recent_3y.sqlite` | 可选 | 近三年 arXiv 补充库，需 `--include-arxiv` |
| SQLite 辅助库 | `data/db/papers.sqlite` | 可选 | 早期采集库，当前主流程一般不直接依赖 |
| HF 数据集导出版 | `data/hf_dataset/data/all_curated.jsonl` | 可选 | 去重后的 JSONL 合并集，便于上传 Hugging Face 和人工检查 |
| HF 数据集导出版 | `data/hf_dataset/data/topconf_all.jsonl` | 可选 | 顶会论文 JSONL 导出版 |
| HF 数据集导出版 | `data/hf_dataset/data/frontier_2026_quality.jsonl` | 可选 | 2026 frontier 论文 JSONL 导出版 |
| HF 数据集导出版 | `data/hf_dataset/data/arxiv_recent_3y_score_gte_4.jsonl` | 可选 | 近三年 arXiv JSONL 导出版 |
| HF 数据集说明 | `data/hf_dataset/README.md` | 可选 | Hugging Face Dataset Card |
| HF 数据集 manifest | `data/hf_dataset/metadata_manifest.json` | 可选 | 数据规模、字段、来源和覆盖统计 |

实际运行时，`embodiedai_kb/search/metadata_search.py` 读取的是 `data/db/*.sqlite`。`data/hf_dataset/` 主要用于 Hugging Face 上传、数据预览和备份；如果只恢复 JSONL 而不恢复 SQLite，系统仍然不能直接使用本地 metadata 检索。

### 10.2 上传 metadata 到 Hugging Face

Hugging Face 官方推荐使用 `hf` CLI 管理 Hub 文件。参考官方文档：Upload files to the Hub 和 CLI guide。

安装 CLI：

```bash
python -m pip install -U huggingface_hub
```

登录：

```bash
hf auth login
```

创建一个 dataset 仓库，例如：

```bash
hf repo create YOUR_NAME/embodied-ai-literature-metadata \
  --type dataset \
  --private
```

建议上传两部分：

1. `sqlite/`：系统运行需要的 SQLite metadata 数据库；
2. `hf_dataset/`：便于老师检查的数据集导出版、Dataset Card 和 manifest。

上传命令：

```bash
# 上传运行必需的 SQLite 数据库
hf upload YOUR_NAME/embodied-ai-literature-metadata \
  data/db sqlite \
  --repo-type dataset

# 上传 Hugging Face JSONL 导出版
hf upload YOUR_NAME/embodied-ai-literature-metadata \
  data/hf_dataset . \
  --repo-type dataset
```

上传完成后，Hugging Face 仓库中建议保持如下结构：

```text
README.md
metadata_manifest.json
data/
  all_curated.jsonl
  topconf_all.jsonl
  frontier_2026_quality.jsonl
  arxiv_recent_3y_score_gte_4.jsonl
sqlite/
  topconf_papers.sqlite
  frontier_papers.sqlite
  papers_recent_3y.sqlite
  papers.sqlite
```

其中 `sqlite/*.sqlite` 用于恢复系统运行，`data/*.jsonl` 用于数据预览和人工检查。

### 10.3 从 Hugging Face 恢复 metadata

如果老师拿到的代码包中没有 metadata 数据库，可以按下面步骤恢复。

安装 CLI 并登录：

```bash
python -m pip install -U huggingface_hub
hf auth login
```

在项目根目录创建数据目录：

```bash
mkdir -p data/db data/hf_dataset
```

下载 SQLite 运行库：

```bash
hf download YOUR_NAME/embodied-ai-literature-metadata \
  --repo-type dataset \
  --include "sqlite/*.sqlite" \
  --local-dir data/_hf_metadata_download

cp data/_hf_metadata_download/sqlite/*.sqlite data/db/
```

可选：下载 JSONL 导出版，方便检查数据内容：

```bash
hf download YOUR_NAME/embodied-ai-literature-metadata \
  --repo-type dataset \
  --include "README.md" \
  --include "metadata_manifest.json" \
  --include "data/*.jsonl" \
  --local-dir data/hf_dataset
```

恢复后检查文件是否齐全：

```bash
ls -lh data/db
```

至少应看到：

```text
topconf_papers.sqlite
frontier_papers.sqlite
papers_recent_3y.sqlite
```

也可以用 SQLite 检查记录数：

```bash
python - <<'PY'
import sqlite3
from pathlib import Path

for path in [
    Path("data/db/topconf_papers.sqlite"),
    Path("data/db/frontier_papers.sqlite"),
    Path("data/db/papers_recent_3y.sqlite"),
]:
    conn = sqlite3.connect(path)
    count = conn.execute("select count(*) from papers").fetchone()[0]
    conn.close()
    print(path, count)
PY
```

当前版本参考记录数：

```text
data/db/topconf_papers.sqlite       79068
data/db/frontier_papers.sqlite        351
data/db/papers_recent_3y.sqlite      1625
```

恢复完成后，可以运行 dry-run 验证 metadata 检索：

```bash
python scripts/ask_literature_langgraph.py "帮我调研 2026 年 VLA 机器人操作方向的最新代表性论文" \
  --include-frontier \
  --year-from 2024 \
  --year-to 2026 \
  --paperqa-k 3 \
  --agent-search-count 4 \
  --dry-run
```

若输出中出现 `Metadata candidates` 或 `PaperSearchAgent` 的 metadata candidate 数量，说明本地 metadata 已恢复成功。

## 11. 常见问题

### 1. 程序卡在搜索阶段

常见原因是 arXiv / OpenAlex / OpenReview 限流或网络代理不稳定。可以降低并发并增加 delay：

```bash
--academic-paper-max-workers 1 \
--academic-paper-request-delay 5 \
--academic-paper-request-timeout 12 \
--academic-paper-search-timeout 75
```

### 2. PaperTriageAgent 选中论文为 0

说明候选论文与问题不够匹配，或者作者/机构消歧不足。可以查看：

```bash
data/metadata/paper_search_logs/last.json
```

重点看：

```text
paper_search_trace.query_analysis.tool_queries
paper_triage_trace.rejected_reasons
paper_triage_trace.coverage_notes
```

### 3. PaperQA evidence 很少

可以提高：

```bash
--paperqa-title-query-count 8 \
--paperqa-per-paper-evidence-count 6 \
--evidence-k 16
```

### 4. 网页端超时

网页端为了避免长时间卡住，默认单轮有超时限制。复杂问题建议用终端运行，或者在网页高级参数中提高 timeout。

### 5. 回答被截断

提高最终回答 token 上限：

```bash
--synthesis-max-tokens 9000
```

## 12. 与 `mini_arxiv_qa/` 的关系

`mini_arxiv_qa/` 是一个更小的终端版系统，只包含：

```text
arXiv Search + PDF Download + PaperQA
```

根目录系统是完整多智能体版本，包含：

```text
Planner / Router
ResearchPlanningAgent
PaperSearchAgent
PaperTriageAgent
PaperQA Reader
SynthesisAgent
Thread Memory
Web UI
```
