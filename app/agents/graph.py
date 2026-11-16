"""
LangGraph workflow (hierarchical delegation + self-reflection):

  triage(supervisor) --spam/high-risk--> finalize
        |
     retrieve --needs_order_data--> data_agent (ReAct) --> draft --> reflect --grounded--> finalize
        |______________no order_____________________________^   ^          |
                                                                |__revise__| (bounded by max_revisions)

State is checkpointed per thread_id (MemorySaver locally; use a Postgres/Firestore saver in prod) so a
run can be resumed, audited or replayed -- the "state management" half of production agent systems.
"""
from __future__ import annotations
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from app.agents.nodes import Agents
from app.agents.state import AgentState
from app.config import get_settings


def build_graph(agents: Agents):
    s = get_settings()
    g = StateGraph(AgentState)
    for name in ("triage", "retrieve", "data_agent", "draft", "reflect", "finalize"):
        g.add_node(name, getattr(agents, name))
    g.set_entry_point("triage")

    g.add_conditional_edges("triage",
        lambda st: "finalize" if st["intent"] == "spam" or st["risk"] == "high" else "retrieve",
        {"finalize": "finalize", "retrieve": "retrieve"})
    g.add_conditional_edges("retrieve",
        lambda st: "data_agent" if st.get("needs_order_data") else "draft",
        {"data_agent": "data_agent", "draft": "draft"})
    g.add_edge("data_agent", "draft")
    g.add_edge("draft", "reflect")
    g.add_conditional_edges("reflect",
        lambda st: "finalize" if st["grounded"] or st["revisions"] > s.max_revisions else "draft",
        {"finalize": "finalize", "draft": "draft"})
    g.add_edge("finalize", END)
    return g.compile(checkpointer=MemorySaver())
