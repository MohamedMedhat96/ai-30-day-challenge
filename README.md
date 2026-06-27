# 30-Day AI Engineering Deep Dive

A build-first, 30-day curriculum for self learning to understand and ship real AI systems — from raw LLM API calls through tool use, agent loops, MCP servers, and production-grade agentic pipelines.

**Daily commitment:** ~45 min (12 min concept · 25 min hands-on · 8 min self-test)  - Takes a lot less in the beginning
**Language:** Python 3.12+  
**Models:** Claude (Haiku for iteration, Sonnet/Opus for quality testing)

## Structure

| Week | Focus |
|------|-------|
| 1 | LLM fundamentals — API calls, prompt engineering, structured output, streaming, caching |
| 2 | Tool use & the agent loop — function calling, multi-tool agents, memory, reasoning patterns |
| 3 | MCP (Model Context Protocol) — servers, clients, transports, security |
| 4 | Production systems — RAG, evals, observability, guardrails, multi-agent patterns |
| Day 29–30 | Capstone project + final exam |

Weekly capstones on Days 7, 14, 21, and 28 are integration days that wire the week's concepts into a shipped artifact.

## Setup

Requires Python 3.12+ and an Anthropic API key.

```bash
# Install dependencies
uv add anthropic "mcp>=1.27,<2"

# Set your API key (add to ~/.zshrc to persist)
export ANTHROPIC_API_KEY="sk-ant-..."
```

Run `main.py` to verify setup:

```bash
python main.py
# expected output: setup works
```

## How to use this repo

- Work through one day at a time in order — each day builds on the last
- Keep `notes.md` updated: end each day with "Today I learned that…"
- Cover the answer key before checking it; self-test first
- If a day runs over 45 min, stop and finish the hands-on next morning as a warm-up

## Dependencies

- [`anthropic`](https://github.com/anthropics/anthropic-sdk-python) — Claude API SDK
- [`mcp`](https://github.com/modelcontextprotocol/python-sdk) — Model Context Protocol SDK (FastMCP bundled)
