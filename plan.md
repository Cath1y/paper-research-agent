# EmbodiedAI-KB 实现草案

## 1. 项目定位

构建一个面向 **具身智能 / VLA / Embodied Agent / Agentic Robotics** 方向的多智能体科研辅助系统。

核心功能：

- 自动整理近年顶会具身智能论文；
- 构建轻量论文知识库；
- 支持文献调研、论文精读、论文对比；
- 支持 research idea 生成；
- 支持 GitHub 代码仓库结构分析；
- 支持个性化学习路线规划。

---

## 2. 总体架构

```text
User Query
    ↓
Task Planner / Router Agent
    ↓
Workflow Plan
    ↓
Specialized Agents
    ├── Literature Survey Agent
    ├── Paper Reading Agent
    ├── Gap Analysis Agent
    ├── Idea Generation Agent
    ├── Feasibility Critic Agent
    ├── Code Analysis Agent
    ├── Learning Planner Agent
    └── Verifier Agent
    ↓
Final Writer Agent
    ↓
Final Answer
````

---

## 3. 技术栈

```text
Python
LangGraph：多 Agent 工作流
LangChain：LLM 调用与工具封装
Chroma / FAISS：向量数据库
SQLite：结构化论文信息
PyMuPDF：PDF 解析
GitHub API / GitPython：代码仓库分析
Streamlit：前端展示
GPT-5.2 / DeepSeek V4 Pro：主力 LLM
```

---

## 4. 数据存储设计

不保存所有论文全文，只保存必要信息。

### 4.1 三层数据

```text
metadata_only:
    title, authors, year, venue, abstract, url

summary_ready:
    motivation, method, limitation, related work, category

fulltext_indexed:
    重点论文的 PDF chunks + embedding
```

### 4.2 存储原则

```text
全量存 metadata + abstract
部分存 structured summary
重点论文才存全文 chunks
代码仓库只存 README + repo tree + code summary
PDF 和代码原文只做本地 cache
```

---

## 5. 数据目录结构

```text
data/
├── db/
│   └── papers.sqlite
│
├── metadata/
│   └── papers.jsonl
│
├── summaries/
│   ├── openvla_2024.json
│   ├── roboclaw_2026.json
│   └── rt2_2023.json
│
├── pdf_cache/
│   ├── openvla_2024.pdf
│   └── roboclaw_2026.pdf
│
├── parsed_text/
│   ├── openvla_2024.md
│   └── roboclaw_2026.md
│
├── vector_db/
│   ├── abstract_chroma/
│   └── fulltext_chroma/
│
└── repo_cache/
    ├── roboclaw/
    │   ├── repo_tree.json
    │   ├── readme.md
    │   └── code_summary.json
```

---

## 6. SQLite 表设计

### 6.1 papers 表

```sql
CREATE TABLE papers (
    paper_id TEXT PRIMARY KEY,
    title TEXT,
    authors TEXT,
    year INTEGER,
    venue TEXT,
    abstract TEXT,
    paper_url TEXT,
    pdf_url TEXT,
    code_url TEXT,
    project_url TEXT,
    keywords TEXT,
    category TEXT,
    source TEXT,
    has_fulltext INTEGER DEFAULT 0,
    has_summary INTEGER DEFAULT 0,
    has_code_analysis INTEGER DEFAULT 0
);
```

### 6.2 paper_summaries 表

```sql
CREATE TABLE paper_summaries (
    paper_id TEXT PRIMARY KEY,
    motivation TEXT,
    problem TEXT,
    method TEXT,
    key_modules TEXT,
    training_data TEXT,
    experiments TEXT,
    limitations TEXT,
    related_papers TEXT,
    ideas TEXT,
    FOREIGN KEY (paper_id) REFERENCES papers(paper_id)
);
```

### 6.3 repos 表

```sql
CREATE TABLE repos (
    repo_id TEXT PRIMARY KEY,
    paper_id TEXT,
    github_url TEXT,
    repo_tree TEXT,
    readme_summary TEXT,
    core_modules TEXT,
    setup_steps TEXT,
    train_entry TEXT,
    inference_entry TEXT,
    difficulty TEXT,
    FOREIGN KEY (paper_id) REFERENCES papers(paper_id)
);
```

---

## 7. 论文筛选范围

### 7.1 会议

```text
CVPR
ICCV
ECCV
NeurIPS
ICML
ICLR
AAAI
IJCAI
CoRL
ICRA
RSS
```

### 7.2 关键词

```text
embodied AI
embodied agent
vision-language-action
VLA
robot learning
robotic manipulation
language-guided robotics
vision-and-language navigation
world model
spatial intelligence
long-horizon task
agentic robotics
GUI agent
computer-use agent
robot foundation model
data collection
```

---

## 8. Agent 设计

### 8.1 Agent 定义

```text
Agent = LLM + Prompt + Tools + State + Output Schema
```

不同 Agent 的差异主要来自：

```text
1. system prompt 不同
2. 可调用工具不同
3. 输出格式不同
4. 读取和写入的 state 字段不同
5. 在 workflow 中的位置不同
```

主要 Agent 可以全部使用 GPT-5.2 或 DeepSeek V4 Pro。

---

## 9. 全局 State 设计

```python
from typing import TypedDict, List, Dict, Optional

class AgentState(TypedDict):
    user_query: str
    workflow_plan: Dict

    retrieved_papers: List[Dict]
    paper_summaries: List[Dict]
    research_gaps: List[Dict]
    candidate_ideas: List[Dict]
    feasibility_reviews: List[Dict]
    code_analysis_results: List[Dict]

    verification_result: Optional[Dict]
    final_answer: str
```

---

## 10. Router / Task Planner Agent

Router 不做单选分类，而是生成 workflow plan。

### 输入

```text
用户问题
```

### 输出

```json
{
  "intent": "research_idea_generation",
  "complexity": "high",
  "execution_mode": "sequential",
  "required_agents": [
    "LiteratureSurveyAgent",
    "PaperReadingAgent",
    "GapAnalysisAgent",
    "IdeaGenerationAgent",
    "FeasibilityCriticAgent",
    "CodeAnalysisAgent",
    "FinalWriterAgent"
  ],
  "subtasks": [
    {
      "agent": "LiteratureSurveyAgent",
      "task": "Retrieve papers related to RoboClaw, VLA, embodied agent, and long-horizon robotic tasks."
    },
    {
      "agent": "PaperReadingAgent",
      "task": "Summarize RoboClaw's motivation, method, EAP mechanism, and limitations."
    },
    {
      "agent": "GapAnalysisAgent",
      "task": "Extract research gaps from RoboClaw and related papers."
    },
    {
      "agent": "IdeaGenerationAgent",
      "task": "Generate undergraduate-level project ideas."
    },
    {
      "agent": "FeasibilityCriticAgent",
      "task": "Evaluate feasibility and implementation risks."
    },
    {
      "agent": "CodeAnalysisAgent",
      "task": "Analyze related GitHub repositories."
    },
    {
      "agent": "FinalWriterAgent",
      "task": "Produce a structured final answer."
    }
  ]
}
```

---

## 11. 主要 Agent 分工

### 11.1 Literature Survey Agent

用途：

```text
检索相关论文
整理方向脉络
按年份、会议、子方向分类
推荐精读论文
```

工具：

```text
paper_metadata_search
abstract_vector_search
filter_by_year
filter_by_venue
taxonomy_classifier
```

输出：

```json
{
  "topic": "...",
  "relevant_papers": [],
  "timeline": [],
  "categories": [],
  "recommended_reading_order": []
}
```

---

### 11.2 Paper Reading Agent

用途：

```text
精读单篇论文
重点解释 motivation 和 method
简单概括 experiment
总结 limitation
```

工具：

```text
get_paper_summary
retrieve_fulltext_chunks
get_paper_sections
```

输出结构：

```markdown
# 论文精读：{title}

## 1. Motivation
## 2. Problem Definition
## 3. Method Overview
## 4. Key Modules
## 5. Training Data
## 6. Experiments
## 7. Limitations
## 8. Related Papers
## 9. Reading Suggestions
```

---

### 11.3 Gap Analysis Agent

用途：

```text
从论文 limitation、future work、method difference 中找研究空白
```

输入：

```text
paper_summaries
retrieved_papers
```

输出：

```json
{
  "gaps": [
    {
      "gap": "...",
      "evidence_papers": [],
      "difficulty": "low / medium / high",
      "possible_direction": "..."
    }
  ]
}
```

---

### 11.4 Idea Generation Agent

用途：

```text
生成本科生可完成的 research idea
```

输入：

```text
research_gaps
paper_summaries
taxonomy
```

输出：

```json
{
  "ideas": [
    {
      "title": "...",
      "motivation": "...",
      "method": "...",
      "baseline": [],
      "evaluation": [],
      "related_papers": [],
      "difficulty": "medium",
      "risks": []
    }
  ]
}
```

---

### 11.5 Feasibility Critic Agent

用途：

```text
评估 idea 是否适合课程项目 / 本科科研
```

检查点：

```text
是否需要真实机器人
是否需要大规模训练
是否可以用仿真或 mock environment
是否有 baseline
是否有可量化指标
是否能在课程周期完成
```

输出：

```json
{
  "idea_title": "...",
  "feasible": true,
  "difficulty": "medium",
  "main_risks": [],
  "suggested_simplification": "..."
}
```

---

### 11.6 Code Analysis Agent

用途：

```text
分析论文 GitHub 仓库结构
给出复现路径
```

工具：

```text
get_repo_tree
read_readme
find_requirements
find_train_entry
find_inference_entry
```

输出结构：

```markdown
# 代码仓库分析：{repo_name}

## 1. Repository Purpose
## 2. Directory Structure
## 3. Core Modules
## 4. Training Entry
## 5. Inference Entry
## 6. Data Format
## 7. Setup Steps
## 8. Minimal Reproduction Path
## 9. Reproduction Risks
## 10. Suggested Reading Order
```

---

### 11.7 Learning Planner Agent

用途：

```text
根据用户背景生成学习路径
```

输入：

```text
user_profile
taxonomy
paper_difficulty
paper_dependencies
```

输出：

```markdown
# 个性化学习路线

## Stage 1: VLM 基础
CLIP → BLIP-2 → LLaVA → InstructBLIP

## Stage 2: VLA 基础
PaLM-E → RT-1 → RT-2 → OpenVLA

## Stage 3: Embodied Agent
RoboClaw → Octo → VLA-Pruner

## Stage 4: Practice
1. 跑通 OpenVLA inference
2. 做 RoboClaw-style mock demo
3. 尝试 ManiSkill / AI2-THOR 小实验
```

---

### 11.8 Verifier Agent

用途：

```text
检查回答是否有证据支持
降低幻觉
```

检查内容：

```text
是否引用真实论文
是否混淆论文贡献
是否夸大实验结论
是否基于检索内容回答
idea 是否过大
代码分析是否基于 README / repo tree
```

输出：

```json
{
  "is_supported": true,
  "unsupported_claims": [],
  "needs_revision": false,
  "revision_suggestions": []
}
```

---

## 12. Agent 上下文传递

前一个 Agent 的输出作为后一个 Agent 的上下文。

但不直接传完整自然语言，而是传结构化结果。

```text
Literature Survey Agent
→ 写入 retrieved_papers / paper_summaries

Gap Analysis Agent
→ 读取 paper_summaries
→ 写入 research_gaps

Idea Generation Agent
→ 读取 research_gaps
→ 写入 candidate_ideas

Feasibility Critic Agent
→ 读取 candidate_ideas
→ 写入 feasibility_reviews

Code Analysis Agent
→ 读取 related repos
→ 写入 code_analysis_results

Final Writer Agent
→ 读取所有中间结果
→ 生成 final_answer
```

---

## 13. RAG 检索策略

### 13.1 文献调研

使用：

```text
metadata database
abstract vector index
```

不查全文。

### 13.2 论文精读

优先使用：

```text
structured summary
```

不够时再查：

```text
fulltext vector index
```

### 13.3 idea 生成

主要使用：

```text
limitations
future_work
method gaps
taxonomy
```

### 13.4 代码分析

主要使用：

```text
README
repo tree
requirements
train / inference entry
```

---

## 14. 项目目录结构

```text
embodiedai-kb/
├── README.md
├── requirements.txt
├── .env.example
│
├── data/
│   ├── db/
│   ├── metadata/
│   ├── summaries/
│   ├── pdf_cache/
│   ├── parsed_text/
│   ├── vector_db/
│   ├── repo_cache/
│   └── taxonomy/
│
├── embodiedai_kb/
│   ├── config/
│   │   ├── settings.py
│   │   └── prompts.py
│   │
│   ├── data_collection/
│   │   ├── arxiv_client.py
│   │   ├── openreview_client.py
│   │   ├── semantic_scholar_client.py
│   │   ├── cvf_scraper.py
│   │   └── github_client.py
│   │
│   ├── processing/
│   │   ├── pdf_parser.py
│   │   ├── section_splitter.py
│   │   ├── metadata_extractor.py
│   │   ├── taxonomy_classifier.py
│   │   └── repo_analyzer.py
│   │
│   ├── storage/
│   │   ├── database.py
│   │   ├── vector_store.py
│   │   └── schemas.py
│   │
│   ├── agents/
│   │   ├── task_planner_agent.py
│   │   ├── literature_survey_agent.py
│   │   ├── paper_reading_agent.py
│   │   ├── gap_analysis_agent.py
│   │   ├── idea_generation_agent.py
│   │   ├── feasibility_critic_agent.py
│   │   ├── code_analysis_agent.py
│   │   ├── learning_planner_agent.py
│   │   ├── verifier_agent.py
│   │   └── final_writer_agent.py
│   │
│   ├── graph/
│   │   └── workflow.py
│   │
│   └── tools/
│       ├── retrieval_tool.py
│       ├── paper_search_tool.py
│       ├── code_search_tool.py
│       └── visualization_tool.py
│
├── app/
│   └── streamlit_app.py
│
└── scripts/
    ├── collect_metadata.py
    ├── download_papers.py
    ├── parse_papers.py
    ├── build_index.py
    ├── analyze_repos.py
    └── run_demo.py
```

---

## 15. 核心实现流程

### Step 1: 采集论文 metadata

```text
输入关键词
→ 从 arXiv / OpenReview / Semantic Scholar / CVF 获取论文信息
→ 存入 SQLite
```

### Step 2: 建立 abstract index

```text
title + abstract + keywords
→ embedding
→ Chroma / FAISS
```

### Step 3: 按需解析重点论文

```text
用户请求精读某篇论文
→ 检查 has_fulltext
→ 如果没有，下载 PDF
→ PyMuPDF 解析
→ chunk
→ 写入 fulltext index
```

### Step 4: 生成 structured summary

```text
fulltext / abstract
→ LLM extraction
→ motivation / method / limitation / related work
→ 写入 paper_summaries 表
```

### Step 5: 分析代码仓库

```text
GitHub URL
→ 获取 README
→ 获取 repo tree
→ 查找 requirements / train / inference 文件
→ 生成 code_summary
→ 写入 repos 表
```

### Step 6: 执行多 Agent workflow

```text
用户输入问题
→ Task Planner 生成 workflow plan
→ 各 Agent 按计划执行
→ 中间结果写入 state
→ Verifier 检查
→ Final Writer 输出最终结果
```

---

## 16. LangGraph 伪代码

```python
def task_planner_node(state):
    user_query = state["user_query"]

    plan = llm.invoke(f"""
    你是一个多智能体科研系统的任务规划器。
    请根据用户问题生成 workflow plan。

    可用 Agent:
    - LiteratureSurveyAgent
    - PaperReadingAgent
    - GapAnalysisAgent
    - IdeaGenerationAgent
    - FeasibilityCriticAgent
    - CodeAnalysisAgent
    - LearningPlannerAgent
    - VerifierAgent
    - FinalWriterAgent

    请返回 JSON:
    {{
      "intent": "...",
      "complexity": "simple / medium / high",
      "execution_mode": "single / sequential / parallel",
      "required_agents": [...],
      "subtasks": [...]
    }}

    用户问题:
    {user_query}
    """)

    state["workflow_plan"] = parse_json(plan)
    return state
```

```python
def execute_plan_node(state):
    plan = state["workflow_plan"]

    for subtask in plan["subtasks"]:
        agent_name = subtask["agent"]
        task = subtask["task"]

        agent = get_agent(agent_name)
        result = agent.run(task, state)

        state = update_state_with_result(
            state=state,
            agent_name=agent_name,
            result=result
        )

    return state
```

```python
def verifier_node(state):
    result = verifier_agent.run(
        task="请检查所有中间结果和最终草稿是否有证据支持。",
        state=state
    )
    state["verification_result"] = result
    return state
```

```python
def final_writer_node(state):
    final_answer = final_writer_agent.run(
        task="请基于所有中间结果生成结构化最终回答。",
        state=state
    )
    state["final_answer"] = final_answer
    return state
```

---

## 17. Streamlit 页面设计

页面包含：

```text
1. 用户问题输入框
2. 模式选择：
   - 自动规划
   - 文献调研
   - 论文精读
   - 论文对比
   - idea 生成
   - 代码分析
   - 学习路线规划

3. Agent 执行路径展示
4. 检索到的论文展示
5. 中间结果展示
6. 最终回答展示
```

---

## 18. MVP 功能清单

```text
[ ] 采集 50 篇具身智能相关论文 metadata
[ ] 构建 SQLite 数据库
[ ] 构建 abstract 向量索引
[ ] 解析 20 篇重点论文全文
[ ] 生成 20 篇 structured summary
[ ] 分析 5 个 GitHub repo
[ ] 实现 Task Planner Agent
[ ] 实现 Literature Survey Agent
[ ] 实现 Paper Reading Agent
[ ] 实现 Idea Generation Agent
[ ] 实现 Code Analysis Agent
[ ] 实现 Learning Planner Agent
[ ] 实现 Verifier Agent
[ ] 实现 Streamlit Demo
```

---

## 19. Baseline 对比

### Baseline 1: Single-Agent RAG

```text
用户问题
→ 检索 top-k chunks
→ 单个 LLM 回答
```

### Ours: Multi-Agent Workflow

```text
用户问题
→ Task Planner 生成执行计划
→ 多 Agent 分工处理
→ 中间结果结构化传递
→ Verifier 检查
→ Final Writer 汇总
```

### 对比指标

```text
Answer Correctness
Evidence Faithfulness
Coverage
Structure Quality
Hallucination Rate
Idea Feasibility
Routing Quality
User Satisfaction
```

---

## 20. 风险控制

```text
1. PDF 解析不稳定：
   - 优先使用 abstract / introduction / method / conclusion
   - 重点论文允许人工修正 summary

2. 论文筛选不准确：
   - 关键词初筛 + LLM 分类 + 人工抽样检查

3. Agent 幻觉：
   - 强制引用知识库证据
   - 使用 Verifier Agent 检查

4. 代码分析过难：
   - 只分析 README、目录树和入口文件
   - 不尝试自动运行代码

5. 项目范围过大：
   - MVP 限定 50 篇 metadata、20 篇全文、5 个 repo
```

---

## 21. 最终交付物

```text
1. GitHub 项目代码
2. Streamlit Demo
3. SQLite 论文数据库
4. Chroma / FAISS 向量索引
5. 多 Agent workflow
6. 测试问题集
7. Single-Agent vs Multi-Agent 对比结果
8. 项目报告
9. 演示视频
```

```
```
