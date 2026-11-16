"""Agent nodes. Each node = one span, so latency/tokens/cost are attributable per agent."""
from __future__ import annotations
import json, re
from app.agents import prompts as P
from app.config import get_settings
from app.llm.base import LLMClient
from app.observability.tracing import record_llm_call, span
from app.rag.retriever import HybridRetriever
from app.tools.orders import TOOLS

PII = [(re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b"), "[email]"),
       (re.compile(r"\b(?:\d[ -]?){13,16}\b"), "[card]"),
       (re.compile(r"(?<![\d-])(?:\+\d{1,3}[\s.-]?)?\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}(?![\d-])"), "[phone]")]
INJECTION = re.compile(r"ignore (all |any )?(previous|prior) instructions|system prompt|you are now", re.I)
AUTO_SEND_INTENTS = {"how_to", "technical_issue", "order_status", "account_security"}


def _json(text: str, default: dict) -> dict:
    try:
        return json.loads(re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip())
    except Exception:
        return default


class Agents:
    def __init__(self, llm: LLMClient, retriever: HybridRetriever):
        self.llm, self.retriever, self.s = llm, retriever, get_settings()

    def _call(self, sp, task, prompt, system, json_mode=False):
        res = self.llm.generate(prompt, system=system, task=task, json_mode=json_mode)
        record_llm_call(sp, task, res)
        return res.text

    # 1) Supervisor / triage --------------------------------------------------------------
    def triage(self, st):
        with span("triage") as sp:
            raw = self._call(sp, "triage", f"TICKET:\n{st['ticket']}\n", P.TRIAGE_SYSTEM, True)
            out = _json(raw, {"intent": "other", "risk": "high", "needs_order_data": False})  # fail closed
            reasons = []
            if INJECTION.search(st["ticket"]):
                out["risk"] = "high"; reasons.append("prompt_injection_suspected")
            sp.attrs.update(out)
            return {"intent": out.get("intent", "other"), "risk": out.get("risk", "high"),
                    "needs_order_data": bool(out.get("needs_order_data")), "reasons": reasons,
                    "revisions": 0, "observations": [], "facts": [], "react_steps": 0}

    # 2) Retrieval specialist -------------------------------------------------------------
    def retrieve(self, st):
        with span("retrieve") as sp:
            hits = self.retriever.search(st["ticket"], tenant=st.get("tenant", "default"))
            support = any(h["sparse"] for h in hits) or any(h["dense_score"] >= 0.55 for h in hits)
            sp.attrs.update(n=len(hits), has_support=support, top=[h["id"] for h in hits[:3]])
            return {"passages": hits, "has_support": support}

    # 3) Data specialist: bounded ReAct loop over tools -----------------------------------------
    def data_agent(self, st):
        with span("data_agent") as sp:
            obs, facts = list(st.get("observations", [])), list(st.get("facts", []))
            steps = 0
            while steps < self.s.max_react_steps:
                steps += 1
                prompt = (f"TICKET:\n{st['ticket']}\n\nCUSTOMER_ID:\n{st.get('customer_id')}\n\n"
                          f"OBSERVATIONS:\n" + ("\n".join(obs) or "none") + "\n")
                step = _json(self._call(sp, "react", prompt, P.REACT_SYSTEM, True), {"action": "finish"})
                if step.get("action") != "lookup_order":
                    break
                result = TOOLS["lookup_order"](step.get("args", {}).get("order_id", ""), st.get("customer_id"))
                obs.append(f"lookup_order({step['args'].get('order_id')}) -> {json.dumps(result)}")
                if "error" not in result:
                    tr = f" Tracking number: {result['tracking']}." if result.get("tracking") else ""
                    facts.append(f"Your order {result['order_id']} ({result['item']}) placed {result['placed_date']} "
                                 f"is currently {result['status']}.{tr}")
                else:
                    facts.append("We could not find that order on your account.")
            sp.attrs["steps"] = steps
            return {"observations": obs, "facts": facts, "react_steps": steps}

    # 4) Drafting agent --------------------------------------------------------------------------
    def draft(self, st):
        with span("draft", revision=st.get("revisions", 0)) as sp:
            ctx = "\n".join(f"[{i}] {p['text']}" for i, p in enumerate(st.get("passages", [])[:4], 1))
            prompt = (f"TICKET:\n{st['ticket']}\n\nCONTEXT:\n{ctx or 'none'}\n\n"
                      f"FACTS:\n" + ("\n".join(f"- {f}" for f in st.get("facts", [])) or "none") + "\n")
            if st.get("feedback"):
                prompt += f"\nFEEDBACK:\n{st['feedback']}\n"
            return {"draft": self._call(sp, "draft", prompt, P.DRAFT_SYSTEM)}

    # 5) Self-reflection critic -------------------------------------------------------------------
    def reflect(self, st):
        with span("reflect") as sp:
            ctx = "\n".join(p["text"] for p in st.get("passages", [])[:4])
            prompt = (f"DRAFT:\n{st['draft']}\n\nCONTEXT:\n{ctx or 'none'}\n\n"
                      f"FACTS:\n" + "\n".join(st.get("facts", [])) + "\n")
            out = _json(self._call(sp, "reflect", prompt, P.REFLECT_SYSTEM, True),
                        {"grounded": False, "feedback": "critic output unparseable"})  # fail closed
            sp.attrs.update(grounded=out.get("grounded"), unsupported=out.get("unsupported", []))
            return {"grounded": bool(out.get("grounded")), "feedback": out.get("feedback", ""),
                    "revisions": st.get("revisions", 0) + 1}

    # 6) Policy guard + final decision ---------------------------------------------------------------
    def finalize(self, st):
        with span("finalize") as sp:
            reasons = list(st.get("reasons", []))
            reply = st.get("draft", "")
            for rx, tag in PII:
                reply = rx.sub(tag, reply)
            if st.get("intent") == "spam":
                outcome = "dropped"
            elif st.get("risk") == "high":
                outcome = "escalated"; reasons.append("high_risk")
            elif not st.get("has_support") and not st.get("facts"):
                outcome = "escalated"; reasons.append("no_supporting_context")
            elif not st.get("grounded", False):
                outcome = "escalated"; reasons.append("ungrounded_after_revisions")
            elif st.get("intent") in AUTO_SEND_INTENTS:
                outcome = "auto_send"
            else:
                outcome = "needs_review"; reasons.append("human_approval_policy")
            sp.attrs.update(outcome=outcome, reasons=reasons)
            return {"outcome": outcome, "reply": "" if outcome in ("dropped", "escalated") else reply, "reasons": reasons}
