"""
Deterministic offline LLM used for CI, unit tests and the eval harness (no network, no cost).

It implements the *same contract* as Gemini for each task (triage JSON, ReAct JSON, grounded drafting,
critic JSON) using simple heuristics, so the full multi-agent graph -- routing, tool use, the
self-reflection loop, guardrails, tracing -- can be exercised end-to-end offline.
IMPORTANT: it is a test double. Quality numbers must be measured with provider=vertex.
"""
from __future__ import annotations
import json, re, time
from app.llm.base import LLMResult, approx_tokens
from app.llm.tenacity_lite import retry  # noqa: F401  (kept for symmetry)

STOP = set("a an the and or of to in on for is are was be can i my me you your it this that with at by from how do does what when why not no we our if as so have has had will would could should please there their they them been being any all".split())
_W = re.compile(r"[a-z0-9\-]+")
ORDER_RE = re.compile(r"\bORD-\d{3,8}\b", re.I)


def content_words(t: str) -> set[str]:
    return {w for w in _W.findall(t.lower()) if w not in STOP and len(w) > 2}


def section(prompt: str, name: str) -> str:
    m = re.search(rf"^{name}:\n(.*?)(?=^\w[\w_ ]*:\n|\Z)", prompt, re.S | re.M)
    return m.group(1).strip() if m else ""


class MockLLM:
    model = "mock"

    def __init__(self, hallucinate_first_draft: bool = False):
        self.hallucinate_first_draft = hallucinate_first_draft

    # ---- tasks -----------------------------------------------------------------------------
    def _triage(self, p: str) -> str:
        t = section(p, "TICKET").lower()
        high = bool(re.search(r"lawyer|lawsuit|sue |chargeback|legal action|hacked|harass|threat", t))
        if re.search(r"free money|crypto giveaway|click here|winner|viagra|http[s]?://", t):
            intent = "spam"
        elif re.search(r"refund|charge|invoice|billing|cancel|money back", t):
            intent = "billing_refund"
        elif ORDER_RE.search(t) or re.search(r"\border\b|shipping|delivery|tracking|package", t):
            intent = "order_status"
        elif re.search(r"2fa|two-factor|password|sso|saml|security|residency|login", t):
            intent = "account_security"
        elif re.search(r"error|e-\d{4}|crash|slow|not syncing|sync|bug|fail", t):
            intent = "technical_issue"
        elif re.search(r"how (do|can|to)|where|plan|price|pricing|storage|limit", t):
            intent = "how_to"
        else:
            intent = "other"
        needs = bool(ORDER_RE.search(t))
        return json.dumps({"intent": intent, "risk": "high" if high else "low", "needs_order_data": needs})

    def _react(self, p: str) -> str:
        t, obs = section(p, "TICKET"), section(p, "OBSERVATIONS")
        m = ORDER_RE.search(t)
        if m and m.group(0).upper() not in obs.upper():
            return json.dumps({"thought": "Ticket cites an order id; fetch it.", "action": "lookup_order",
                               "args": {"order_id": m.group(0).upper()}})
        return json.dumps({"thought": "Enough information.", "action": "finish", "args": {}})

    def _draft(self, p: str) -> str:
        ticket, ctx, facts = section(p, "TICKET"), section(p, "CONTEXT"), section(p, "FACTS")
        feedback = section(p, "FEEDBACK")
        qwords = content_words(ticket)
        cands: list[tuple[float, str, str]] = []
        for line in ctx.splitlines():
            m = re.match(r"\[(\d+)\]\s*(.*)", line)
            if not m:
                continue
            n, text = m.groups()
            text = text.split(": ", 1)[-1] if ": " in text[:80] else text
            for sent in re.split(r"(?<=[.!?])\s+", text):
                if len(sent) < 15:
                    continue
                w = content_words(sent)
                cands.append((len(w & qwords) / (len(w) ** 0.5 or 1), sent.strip(), n))
        cands.sort(key=lambda x: x[0], reverse=True)
        top = cands[0][0] if cands else 0
        picked = [(s, n) for sc, s, n in cands[:2] if sc > 0 and sc >= 0.6 * top]
        body: list[str] = []
        for f in facts.splitlines():
            f = f.lstrip("- ").strip()
            if f and f.lower() != "none":
                body.append(f)
        for s, n in picked:
            body.append(f"{s.rstrip('.')} [{n}].")
        if not body:
            body.append("I could not find a documented answer, so I am passing this to a specialist who will follow up.")
        if self.hallucinate_first_draft and not feedback:
            body.append("We also offer lifetime free upgrades and a complimentary titanium Nimbus Hub to every customer.")
        return "Hi there,\n\n" + " ".join(body) + "\n\nBest regards,\nSupport Team"

    def _reflect(self, p: str) -> str:
        draft, ctx, facts = section(p, "DRAFT"), section(p, "CONTEXT"), section(p, "FACTS")
        support = content_words(re.sub(r"\[\d+\]", "", ctx + " " + facts))
        body = [l for l in draft.splitlines() if l.strip() and not l.startswith(("Hi", "Best", "Support Team"))]
        unsupported = []
        for sent in re.split(r"(?<=[.!?])\s+", " ".join(body)):
            w = content_words(re.sub(r"\[\d+\]", "", sent))
            if len(w) >= 4 and len(w & support) / len(w) < 0.6:
                unsupported.append(sent.strip())
        return json.dumps({"grounded": not unsupported, "unsupported": unsupported,
                           "feedback": "Remove or rewrite unsupported claims; use only CONTEXT/FACTS." if unsupported else ""})

    # ---- interface ---------------------------------------------------------------------------
    def generate(self, prompt: str, *, system: str = "", task: str = "generic", json_mode: bool = False) -> LLMResult:
        t0 = time.perf_counter()
        fn = {"triage": self._triage, "react": self._react, "draft": self._draft, "reflect": self._reflect}.get(task)
        text = fn(prompt) if fn else ""
        return LLMResult(text=text, model=self.model, prompt_tokens=approx_tokens(system + prompt),
                         completion_tokens=approx_tokens(text), latency_s=time.perf_counter() - t0)
