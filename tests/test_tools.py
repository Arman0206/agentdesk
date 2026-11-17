from app.tools.orders import lookup_order


def test_owner_can_read_own_order(settings):
    r = lookup_order("ORD-1001", "CUST-1")
    assert r["status"] == "shipped" and r["tracking"] == "TRK-88213"


def test_other_customer_cannot_read_order(settings):          # IDOR protection
    assert lookup_order("ORD-1001", "CUST-2") == {"error": "not_found"}


def test_unauthenticated_gets_nothing(settings):
    assert lookup_order("ORD-1001", None) == {"error": "unauthenticated"}


def test_sql_injection_is_rejected(settings):
    assert lookup_order("ORD-1001' OR '1'='1", "CUST-1") == {"error": "invalid_order_id"}
