# AgentDesk

Enterprise multi-agent support-resolution copilot on Google Cloud (Vertex AI / Gemini).
Hybrid RAG over structured + unstructured data, LangGraph agents (supervisor, ReAct tools,
self-reflection), LLM-native tracing (tokens, cost, latency) and an eval harness used as a release gate.

Status: work in progress.

## Quickstart (offline, no GCP needed)
    python -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    python -m app.tools.seed data/orders.db
    python -m evals.run_eval
