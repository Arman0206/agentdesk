from __future__ import annotations
from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    # input
    ticket: str
    customer_id: str | None
    tenant: str
    # triage
    intent: str
    risk: str                 # low | high
    needs_order_data: bool
    # retrieval / tools
    passages: list[dict]
    has_support: bool
    observations: list[str]   # ReAct observations
    facts: list[str]
    react_steps: int
    # generation + reflection
    draft: str
    feedback: str
    grounded: bool
    revisions: int
    # outcome
    outcome: str              # auto_send | needs_review | escalated | dropped
    reply: str
    reasons: list[str]
