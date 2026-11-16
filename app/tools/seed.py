"""Seed the demo orders DB (stands in for a customer's ERP / order-management system)."""
import sqlite3, sys
from pathlib import Path

ROWS = [
    ("ORD-1001", "CUST-1", "Nimbus Hub", "shipped", "2026-11-02", "TRK-88213", 199.00),
    ("ORD-1002", "CUST-2", "Nimbus Hub", "processing", "2026-11-12", None, 199.00),
    ("ORD-1003", "CUST-1", "Business annual plan", "paid", "2026-10-20", None, 1800.00),
    ("ORD-1004", "CUST-3", "Starter monthly plan", "paid", "2026-11-01", None, 30.00),
    ("ORD-1005", "CUST-2", "Nimbus Hub", "delivered", "2026-10-15", "TRK-77102", 199.00),
]


def seed(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.execute("DROP TABLE IF EXISTS orders")
    con.execute("""CREATE TABLE orders(order_id TEXT PRIMARY KEY, customer_id TEXT, item TEXT,
                   status TEXT, placed_date TEXT, tracking TEXT, amount_usd REAL)""")
    con.executemany("INSERT INTO orders VALUES (?,?,?,?,?,?,?)", ROWS)
    con.commit(); con.close()


if __name__ == "__main__":
    seed(sys.argv[1] if len(sys.argv) > 1 else "data/orders.db")
    print("seeded")
