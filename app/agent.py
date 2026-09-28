"""Autonomous returns agent: a Claude tool-use loop over order lookup, policy check and return creation."""

import json
import logging
import uuid
from dataclasses import dataclass, field

import anthropic

from app.orders import Order
from app.policy import POLICY_TEXT, check_eligibility
from app.store import Conversation

log = logging.getLogger(__name__)

MODEL = "claude-opus-5"
MAX_STEPS = 8
DECISIONS = ("eligible", "not_eligible", "return_started", "info_only", "needs_human")

SYSTEM_PROMPT = f"""You are the returns assistant for an online store. You help customers find out whether an item can be returned and start returns for eligible items.

{POLICY_TEXT}

How to work:
- Before discussing a specific order, verify it with lookup_order using the order ID and the email on the order. Ask the customer for whichever is missing.
- Decide eligibility only with check_return_eligibility; never judge it yourself. If the customer hasn't said whether the item is unused and still in its original packaging, ask before checking.
- Call start_return only when the customer asks to return the item and it was found eligible.
- You cannot make exceptions to the policy, promise refund dates, or offer compensation. If the customer wants any of that, or you can't resolve the request, set decision to needs_human.
- End every turn by calling final_answer with your reply to the customer. Set order_id and sku whenever the decision is about a specific item.
- Never repeat the customer's email address or payment details back to them."""

TOOLS = [
    {
        "name": "lookup_order",
        "description": "Verify an order belongs to the customer and return its status, delivery date and items. Both the order ID and the email on the order are required.",
        "input_schema": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "email": {"type": "string"},
            },
            "required": ["order_id", "email"],
            "additionalProperties": False,
        },
    },
    {
        "name": "check_return_eligibility",
        "description": "Apply the return policy to one item of a verified order. Returns eligible true/false, the reasons, and the return deadline.",
        "input_schema": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "sku": {"type": "string"},
                "item_unused": {"type": "boolean", "description": "Customer confirmed the item is unused."},
                "original_packaging": {"type": "boolean", "description": "Customer confirmed the item is in its original packaging."},
            },
            "required": ["order_id", "sku", "item_unused", "original_packaging"],
            "additionalProperties": False,
        },
    },
    {
        "name": "start_return",
        "description": "Create a return authorization (RMA) for an item that check_return_eligibility found eligible.",
        "input_schema": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "sku": {"type": "string"},
                "reason": {"type": "string", "description": "Customer's reason for the return."},
            },
            "required": ["order_id", "sku", "reason"],
            "additionalProperties": False,
        },
    },
    {
        "name": "final_answer",
        "description": "Send your reply to the customer and end your turn. Call exactly once per turn, last.",
        "input_schema": {
            "type": "object",
            "properties": {
                "message": {"type": "string", "description": "Reply shown to the customer."},
                "decision": {
                    "type": "string",
                    "enum": list(DECISIONS),
                    "description": "eligible / not_eligible: result of check_return_eligibility for order_id+sku. return_started: start_return succeeded. info_only: general answer or asking for details. needs_human: a person must handle this.",
                },
                "order_id": {"type": "string"},
                "sku": {"type": "string"},
            },
            "required": ["message", "decision"],
            "additionalProperties": False,
        },
    },
]


class ToolError(Exception):
    pass


@dataclass
class TurnResult:
    final: dict | None = None
    tool_calls: list[dict] = field(default_factory=list)
    error: str | None = None
    draft: str | None = None  # plain text the model wrote, if any


def _str(inp: dict, key: str) -> str:
    value = inp.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ToolError(f"'{key}' must be a non-empty string")
    return value.strip()


def _bool(inp: dict, key: str) -> bool:
    value = inp.get(key)
    if not isinstance(value, bool):
        raise ToolError(f"'{key}' must be true or false")
    return value


class ReturnsAgent:
    def __init__(self, client, orders: dict[str, Order]):
        self.client = client
        self.orders = orders

    # --- tools -------------------------------------------------------------

    def _verified_order(self, conv: Conversation, order_id: str) -> Order:
        if order_id not in conv.verified_orders:
            raise ToolError(f"Order {order_id} is not verified in this conversation; call lookup_order first")
        return self.orders[order_id]

    def _execute(self, conv: Conversation, name: str, inp: dict) -> dict:
        if name == "lookup_order":
            order_id, email = _str(inp, "order_id").upper(), _str(inp, "email").lower()
            order = self.orders.get(order_id)
            if order is None or order.email != email:
                # Same message for both cases so the tool can't be used to probe order IDs.
                raise ToolError("No order matches that order ID and email")
            conv.verified_orders.add(order_id)
            return order.public_view()

        if name == "check_return_eligibility":
            order = self._verified_order(conv, _str(inp, "order_id").upper())
            sku = _str(inp, "sku").upper()
            try:
                result = check_eligibility(order, sku, item_unused=_bool(inp, "item_unused"),
                                           original_packaging=_bool(inp, "original_packaging"))
            except KeyError as e:
                raise ToolError(str(e)) from e
            conv.eligibility[f"{order.order_id}:{sku}"] = result.as_dict()
            return result.as_dict()

        if name == "start_return":
            order = self._verified_order(conv, _str(inp, "order_id").upper())
            sku = _str(inp, "sku").upper()
            reason = _str(inp, "reason")
            key = f"{order.order_id}:{sku}"
            if key in conv.returns:
                return {"rma_id": conv.returns[key], "already_started": True}
            check = conv.eligibility.get(key)
            if not check or not check["eligible"]:
                raise ToolError("Item has not been found eligible; call check_return_eligibility first")
            rma = "RMA-" + uuid.uuid4().hex[:8].upper()
            conv.returns[key] = rma
            log.info("Return %s created for %s (%s)", rma, key, reason)
            return {"rma_id": rma, "instructions": "A prepaid label will be emailed to the address on the order."}

        raise ToolError(f"Unknown tool {name}")

    @staticmethod
    def _validate_final(inp: dict) -> dict:
        message = _str(inp, "message")
        decision = inp.get("decision")
        if decision not in DECISIONS:
            raise ToolError(f"'decision' must be one of {', '.join(DECISIONS)}")
        final = {"message": message, "decision": decision}
        for key in ("order_id", "sku"):
            if isinstance(inp.get(key), str) and inp[key].strip():
                final[key] = inp[key].strip().upper()
        return final

    # --- loop --------------------------------------------------------------

    def run_turn(self, conv: Conversation, user_text: str) -> TurnResult:
        conv.messages.append({"role": "user", "content": user_text})
        turn = TurnResult()

        for _ in range(MAX_STEPS):
            try:
                response = self.client.beta.messages.create(
                    model=MODEL,
                    max_tokens=16000,
                    system=SYSTEM_PROMPT,
                    tools=TOOLS,
                    messages=conv.messages,
                    betas=["server-side-fallback-2026-07-01"],
                    extra_body={"fallbacks": "default"},
                )
            except anthropic.APIError as e:
                log.exception("Claude API call failed")
                turn.error = f"api_error: {type(e).__name__}"
                return turn

            conv.messages.append({"role": "assistant", "content": response.content})
            text = "".join(b.text for b in response.content if b.type == "text").strip()
            if text:
                turn.draft = text

            if response.stop_reason == "refusal":
                turn.error = "refusal"
                return turn
            if response.stop_reason != "tool_use":
                turn.error = f"no_final_answer (stop_reason={response.stop_reason})"
                return turn

            results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                inp = block.input if isinstance(block.input, dict) else {}
                try:
                    if block.name == "final_answer":
                        turn.final = self._validate_final(inp)
                        output = {"sent": True}
                    else:
                        output = self._execute(conv, block.name, inp)
                    is_error = False
                except ToolError as e:
                    output, is_error = {"error": str(e)}, True
                if block.name != "final_answer":
                    turn.tool_calls.append({"name": block.name, "input": inp, "output": output, "is_error": is_error})
                results.append({"type": "tool_result", "tool_use_id": block.id,
                                "content": json.dumps(output), "is_error": is_error})
            conv.messages.append({"role": "user", "content": results})

            if turn.final:
                turn.draft = turn.final["message"]
                return turn

        turn.error = f"max_steps ({MAX_STEPS}) reached without final_answer"
        return turn
