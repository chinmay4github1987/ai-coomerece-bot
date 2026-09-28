"""Return policy: the single source of truth for eligibility. The LLM never decides this."""

from dataclasses import dataclass
from datetime import date, timedelta

from app.orders import Order

RETURN_WINDOW_DAYS = 30
NON_RETURNABLE_CATEGORIES = frozenset({"gift_card", "perishable", "personal_care", "final_sale"})

POLICY_TEXT = f"""Return policy:
- Returns are accepted within {RETURN_WINDOW_DAYS} days of delivery.
- Items must be unused and in their original packaging.
- Orders that have not been delivered yet cannot be returned.
- These categories are final sale and cannot be returned: {", ".join(sorted(NON_RETURNABLE_CATEGORIES))}.
- Returns can be started from the order details page or by this assistant.
- Refunds are issued to the original payment method after the returned item passes inspection."""


@dataclass(frozen=True)
class Eligibility:
    eligible: bool
    reasons: tuple[str, ...]
    return_deadline: date | None

    def as_dict(self) -> dict:
        return {
            "eligible": self.eligible,
            "reasons": list(self.reasons),
            "return_deadline": self.return_deadline.isoformat() if self.return_deadline else None,
        }


def check_eligibility(
    order: Order,
    sku: str,
    *,
    item_unused: bool,
    original_packaging: bool,
    today: date | None = None,
) -> Eligibility:
    item = order.item(sku)
    if item is None:
        raise KeyError(f"SKU {sku} is not part of order {order.order_id}")
    today = today or date.today()

    reasons = []
    deadline = None
    if order.status != "delivered" or order.delivered_on is None:
        reasons.append("Order has not been delivered yet.")
    else:
        deadline = order.delivered_on + timedelta(days=RETURN_WINDOW_DAYS)
        if today > deadline:
            reasons.append(f"The {RETURN_WINDOW_DAYS}-day return window ended on {deadline.isoformat()}.")
    if item.category in NON_RETURNABLE_CATEGORIES:
        reasons.append(f"Items in category '{item.category}' are final sale.")
    if not item_unused:
        reasons.append("Item has been used.")
    if not original_packaging:
        reasons.append("Item is not in its original packaging.")

    return Eligibility(eligible=not reasons, reasons=tuple(reasons), return_deadline=deadline)
