# 基于 arXiv 的智能论文问答系统

本系统面向专业论文学习场景，提供一个终端版 arXiv 知识库问答工具：

> 根据用户问题从 arXiv 检索相关论文，下载 PDF，按需构建临时论文知识库，并基于论文证据生成回答。

系统不需要预先维护固定数据库。每次提问时，它会围绕当前问题动态检索论文、缓存 PDF，并用 PaperQA 对 PDF 内容建立检索索引，相当于为当前问题临时构建一个小型专业领域知识库。

## 功能流程

```text
用户输入专业论文问题
  -> 生成或接收 arXiv 检索 query
  -> 调用 arXiv API 检索论文
  -> 选择 top-k 篇带 PDF 的论文
  -> 下载/缓存 PDF
  -> PaperQA 为 PDF 建立临时检索索引
  -> 输出带论文证据的回答
```

## 目录结构

```text
./
  ask_arxiv.py      # 主程序：arXiv 检索、PDF 下载、PaperQA 问答
  README.md         # 运行说明
  cache/pdfs/       # 运行后生成，缓存下载的 PDF，不需要提交
  runs/last.json    # 运行后生成，保存最近一次运行记录，不需要提交
```

## 1. 环境准备

请先进入本代码目录：

```bash
cd /path/to/this/code
```

激活 conda 环境：

```bash
conda activate base
```

安装 PaperQA 依赖。若老师拿到的是完整项目压缩包，`third_party/paper-qa` 位于本目录上一级，可执行：

```bash
python -m pip install -e ../third_party/paper-qa
```

如果目录结构不同，也可以直接安装 PyPI 版本：

```bash
python -m pip install paper-qa
```

如果环境中缺少 `litellm` 等依赖，可补装：

```bash
python -m pip install litellm
```

## 2. API 配置

### 方式 A：使用 OpenAI 官方 API

```bash
export OPENAI_API_KEY="你的 OpenAI API Key"
```

运行时指定模型，例如：

```bash
--llm gpt-4o-mini
```

### 方式 B：使用 OpenAI-compatible 中转站

如果使用兼容 OpenAI 接口的中转站，需要配置：

```bash
export OPENAI_API_KEY="你的中转站 API Key"
export OPENAI_BASE_URL="https://你的中转站地址/v1"
```

然后正常指定模型名即可：

```bash
--llm gpt-4o-mini
```

脚本检测到 `OPENAI_BASE_URL` 后，会自动把模型名转换成 LiteLLM 需要的 `openai/<model>` 格式。

### 可选：写入 `.env`

也可以在当前目录或项目根目录新建 `.env`：

```bash
OPENAI_API_KEY=你的 API Key
OPENAI_BASE_URL=https://你的中转站地址/v1
```

程序启动时会自动读取当前目录所在项目中的 `.env` 和 `.env.local`。

## 3. 快速运行

推荐先运行一个 dry-run，确认 arXiv 检索正常：

```bash
python ask_arxiv.py "请精读 OpenVLA 论文" \
  --query 'ti:"OpenVLA"' \
  --max-results 3 \
  --paper-k 2 \
  --dry-run
```

如果能看到候选论文列表，说明 arXiv 检索链路正常。

`--dry-run` 只测试检索和论文选择，不会下载 PDF，也不会消耗 LLM 问答 token。

## 4. 完整问答示例

下面命令会搜索 arXiv、下载 PDF，并调用 PaperQA 生成回答：

```bash
python ask_arxiv.py "请调研 OpenVLA 这篇论文，并总结它和 RT-2、Octo 的关系" \
  --query 'ti:"OpenVLA"' \
  --query 'all:"OpenVLA" AND all:"RT-2"' \
  --max-results 8 \
  --paper-k 4 \
  --llm gpt-4o-mini
```

也可以不手动写 `--query`，让模型根据问题生成 arXiv query：

```bash
python ask_arxiv.py "2026 年 VLA 机器人操作有哪些最新趋势？" \
  --max-results 10 \
  --paper-k 5 \
  --year-from 2024 \
  --year-to 2026 \
  --llm gpt-4o-mini
```

## 5. 常用参数

- `--query`：手动添加 arXiv query，可重复传入。支持 `au:`、`ti:`、`abs:`、`all:`、`AND`、`OR` 等 arXiv API 语法。
- `--max-results`：每个 query 最多返回多少篇论文。
- `--paper-k`：最多选择多少篇 PDF 交给 PaperQA 阅读。
- `--year-from / --year-to`：按照 arXiv 发布时间过滤论文。
- `--dry-run`：只检索和选择论文，不下载 PDF，也不调用 PaperQA。
- `--download-only`：只检索和下载 PDF，不调用 PaperQA 回答。
- `--llm`：指定问答模型，例如 `gpt-4o-mini`、`gpt-4o`。
- `--embedding sparse`：默认值，使用 PaperQA 的 sparse 检索，减少 embedding API 依赖。
- `--run-json`：指定运行记录保存路径。
- `--request-delay`：arXiv 请求间隔。若遇到限流，可调大到 `3` 或 `5`。

查看完整参数：

```bash
python ask_arxiv.py --help
```

## 6. 输出文件

最近一次运行记录：

```bash
runs/last.json
```

PDF 缓存目录：

```bash
cache/pdfs/
```

运行记录 JSON 中包含：

```text
question       用户问题
queries        实际使用的 arXiv 检索 query
candidates     arXiv 返回的候选论文
selected       被选中并交给 PaperQA 的论文
answer         PaperQA 最终回答
paperqa_trace  PaperQA 配置与读取信息
```

## 7. 常见问题

### 1. arXiv 检索没有结果

可以手动传入更精确的 query，例如：

```bash
--query 'ti:"OpenVLA"'
--query 'au:"Sergey Levine" AND all:"robot"'
--query 'all:"vision-language-action" AND all:"robot manipulation"'
```

### 2. 下载 PDF 失败

可能是网络、arXiv 限流或 PDF 链接临时不可用。可以：

```bash
--request-delay 5
--download-retries 3
```

### 3. PaperQA 报 API 错误

检查：

```bash
echo $OPENAI_API_KEY
echo $OPENAI_BASE_URL
```

如果使用中转站，确认 `OPENAI_BASE_URL` 是 API base URL，而不是 API key。

### 4. 回答质量不够好

可以提高阅读论文数量：

```bash
--paper-k 6
```

也可以手动提供更精确的 query，让系统下载更相关的 PDF。


