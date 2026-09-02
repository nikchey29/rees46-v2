"""Canonical REES46 event schema."""

EXPECTED_COLUMNS = (
    "event_time",
    "event_type",
    "product_id",
    "category_id",
    "category_code",
    "brand",
    "price",
    "user_id",
    "user_session",
)

VALID_EVENT_TYPES = frozenset(
    {
        "view",
        "cart",
        "remove_from_cart",
        "purchase",
    }
)
