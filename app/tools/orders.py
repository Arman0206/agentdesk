"""
Agent tool: order lookup against a structured system of record.

Security notes (what a customer security review will ask about):
  * parameterised SQL only -- the LLM never writes SQL
  * authorisation is enforced IN THE TOOL: an order is returned only if it belongs to the
    authenticated customer_id, so a prompt-injected ticket cannot read someone else's order (IDOR)
  * the tool returns a minimal projection of columns (no payment details)
"""
from __future__ import annotations
import re, sqlite3
from app.config import get_settings

ORDER_ID = re.compile(r"\bORD-\d{3,8}\b", re.I)


def lookup_order(order_id: str, customer_id: str | None) -> dict:
    if not ORDER_ID.fullmatch(order_id or ""):
        return {"error": "invalid_order_id"}
    if not customer_id:
        return {"error": "unauthenticated"}
    con = sqlite3.connect(get_settings().orders_db)
    con.row_factory = sqlite3.Row
    try:
        row = con.execute(
            "SELECT order_id, item, status, placed_date, tracking, amount_usd "
            "FROM orders WHERE order_id = ? AND customer_id = ?",
            (order_id.upper(), customer_id)).fetchone()
    finally:
        con.close()
    return dict(row) if row else {"error": "not_found"}


TOOLS = {"lookup_order": lookup_order}
