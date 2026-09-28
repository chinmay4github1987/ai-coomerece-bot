"""Order data access. Seed data stores delivery as "days ago" so the demo never ages."""

import json
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

ORDERS_PATH = Path(__file__).resolve().parent.parent / "data" / "orders.json"


@dataclass(frozen=True)
class Item:
    sku: str
    name: str
    category: str
    price: float


@dataclass(frozen=True)
class Order:
    order_id: str
    email: str
    status: str
    delivered_on: date | None
    items: tuple[Item, ...]

    def item(self, sku: str) -> Item | None:
        return next((i for i in self.items if i.sku == sku), None)

    def public_view(self) -> dict:
        """What the agent may see and repeat: no email or payment data."""
        return {
            "order_id": self.order_id,
            "status": self.status,
            "delivered_on": self.delivered_on.isoformat() if self.delivered_on else None,
            "items": [
                {"sku": i.sku, "name": i.name, "category": i.category, "price": i.price}
                for i in self.items
            ],
        }


def load_orders(path: Path = ORDERS_PATH, today: date | None = None) -> dict[str, Order]:
    today = today or date.today()
    orders = {}
    for raw in json.loads(path.read_text(encoding="utf-8")):
        days = raw["delivered_days_ago"]
        orders[raw["order_id"]] = Order(
            order_id=raw["order_id"],
            email=raw["email"].lower(),
            status=raw["status"],
            delivered_on=today - timedelta(days=days) if days is not None else None,
            items=tuple(Item(**i) for i in raw["items"]),
        )
    return orders
