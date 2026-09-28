"""Offline stand-in for Claude, used when DEMO_MODE=1 so the bot can be tried without API credits.

It speaks the same `client.beta.messages.create` interface and drives the real tools with simple keyword rules.
Tools, the policy engine, evals and escalations all run exactly as they do with the live model.
"""

import json
import re
import uuid
from types import SimpleNamespace

ORDER_RE = re.compile(r"\b([A-Za-z]\d{4})\b")
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
USED_NO = re.compile(r"\b(unused|unopened|brand new|never (been )?(used|worn|opened)|not (been )?(used|worn|opened)|still new)\b", re.I)
USED_YES = re.compile(r"\b(used|worn|opened|washed)\b", re.I)
PACK_NO = re.compile(r"\b((no|without|lost|threw away|thrown away|missing)( the| its| my| original)? (box|packaging|package)|"
                     r"not in (the |its )?original)\b", re.I)
PACK_YES = re.compile(r"\b(original (box|packaging|package)|in the box|sealed|with (the )?(box|packaging))\b", re.I)
WANTS_RETURN = re.compile(r"\b(start|process|begin|go ahead|yes,? (please )?return|return it|want to return|like to return)\b", re.I)
WANTS_EXCEPTION = re.compile(r"\b(exception|waive|compensat\w*|refund (today|now|immediately))\b", re.I)


def _block(**kw):
    return SimpleNamespace(**kw)


def _tool_step(name: str, **inp):
    return SimpleNamespace(stop_reason="tool_use", content=[
        _block(type="tool_use", id=f"toolu_demo_{uuid.uuid4().hex[:12]}", name=name, input=inp)])


def _final(message: str, decision: str, order_id: str | None = None, sku: str | None = None):
    inp = {"message": message, "decision": decision}
    if order_id and sku:
        inp.update(order_id=order_id, sku=sku)
    return _tool_step("final_answer", **inp)


def _customer_texts(messages: list[dict]) -> list[str]:
    return [m["content"] for m in messages
            if m["role"] == "user" and isinstance(m["content"], str) and not m["content"].startswith("[Support agent")]


def _this_turn_results(messages: list[dict]) -> dict[str, tuple[dict, dict, bool]]:
    """Tool name -> (input, output, is_error) for tools already called in the current turn."""
    start = max(i for i, m in enumerate(messages) if m["role"] == "user" and isinstance(m["content"], str))
    calls, results = {}, {}
    for m in messages[start + 1:]:
        for b in m["content"]:
            if m["role"] == "assistant" and b.type == "tool_use":
                calls[b.id] = (b.name, b.input)
            elif m["role"] == "user":
                results[b["tool_use_id"]] = (json.loads(b["content"]), b["is_error"])
    return {name: (inp, *results[tid]) for tid, (name, inp) in calls.items() if tid in results}


def _latest(texts: list[str], rules: list[tuple[re.Pattern, bool]]) -> bool | None:
    """Most recent customer statement wins; within a message, earlier rules win ("unused" contains "used")."""
    for text in reversed(texts):
        for pattern, value in rules:
            if pattern.search(text):
                return value
    return None


def _pick_item(items: list[dict], texts: list[str]) -> dict | None:
    if len(items) == 1:
        return items[0]
    for text in reversed(texts):
        lowered = text.lower()
        for item in items:
            words = [w for w in item["name"].lower().split() if len(w) > 3]
            if item["sku"].lower() in lowered or any(w in lowered for w in words):
                return item
    return None


class DemoClient:
    def __init__(self):
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, *, messages: list[dict], **_):
        texts = _customer_texts(messages)
        done = _this_turn_results(messages)
        all_text = " ".join(texts)

        if WANTS_EXCEPTION.search(texts[-1]):
            return _final("I can't make exceptions to the return policy myself, so I'm bringing in a colleague.",
                          "needs_human")

        order_ids = ORDER_RE.findall(all_text)
        emails = EMAIL_RE.findall(all_text)
        if not order_ids or not emails:
            missing = " and ".join(x for x, have in (("order ID", order_ids), ("the email on the order", emails))
                                   if not have)
            return _final(f"Happy to help with a return. Could you share your {missing}?", "info_only")
        order_id, email = order_ids[-1].upper(), emails[-1]

        if "lookup_order" not in done:
            return _tool_step("lookup_order", order_id=order_id, email=email)
        _, order, failed = done["lookup_order"]
        if failed:
            return _final("I couldn't find an order with that order ID and email. Could you double-check both?",
                          "info_only")

        item = _pick_item(order["items"], texts)
        if item is None:
            names = ", ".join(f"{i['name']} ({i['sku']})" for i in order["items"])
            return _final(f"I found order {order_id}. Which item would you like to return: {names}?", "info_only")
        sku = item["sku"]

        unused = _latest(texts, [(USED_NO, True), (USED_YES, False)])
        packaging = _latest(texts, [(PACK_NO, False), (PACK_YES, True)])
        if unused is None or packaging is None:
            return _final(f"Got it, the {item['name']}. Is it unused, and is it still in its original packaging?",
                          "info_only")

        if "check_return_eligibility" not in done:
            return _tool_step("check_return_eligibility", order_id=order_id, sku=sku,
                              item_unused=unused, original_packaging=packaging)
        _, check, _ = done["check_return_eligibility"]
        if not check["eligible"]:
            return _final(f"Sorry, the {item['name']} can't be returned: " + " ".join(check["reasons"]),
                          "not_eligible", order_id, sku)

        if not WANTS_RETURN.search(texts[-1]):
            return _final(f"Good news: the {item['name']} can be returned until {check['return_deadline']}. "
                          "Say \"start my return\" and I'll set it up.", "eligible", order_id, sku)

        if "start_return" not in done:
            return _tool_step("start_return", order_id=order_id, sku=sku, reason=texts[-1][:200])
        _, rma, _ = done["start_return"]
        return _final(f"Your return for the {item['name']} is started (reference {rma['rma_id']}). "
                      "A prepaid label will be emailed to the address on the order.", "return_started", order_id, sku)
