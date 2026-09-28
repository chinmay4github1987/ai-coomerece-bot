from datetime import date, timedelta
from types import SimpleNamespace

import anthropic
import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.orders import load_orders
from app.policy import check_eligibility

TOKEN = "test-token"


# --- fake Claude client ------------------------------------------------------

def tool(name, **inp):
    return SimpleNamespace(type="tool_use", id=f"tu_{name}_{id(inp)}", name=name, input=inp)


def step(*blocks, stop_reason="tool_use"):
    return SimpleNamespace(stop_reason=stop_reason, content=list(blocks))


class ScriptedClient:
    """Returns the scripted responses in order; raises the item instead if it is an exception."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        nxt = self.responses.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt


def make_client(responses):
    fake = ScriptedClient(responses)
    return fake, TestClient(create_app(client=fake, support_token=TOKEN))


def chat(http, message, conversation_id=None):
    res = http.post("/api/chat", json={"conversation_id": conversation_id, "message": message})
    assert res.status_code == 200
    return res.json()


AUTH = {"X-Support-Token": TOKEN}


# --- policy ------------------------------------------------------------------

@pytest.fixture
def orders():
    return load_orders()


def test_policy_eligible_within_window(orders):
    r = check_eligibility(orders["A1001"], "HP-200", item_unused=True, original_packaging=True)
    assert r.eligible and r.reasons == ()


def test_policy_rejects_each_rule(orders):
    assert not check_eligibility(orders["A1002"], "JK-310", item_unused=True, original_packaging=True).eligible
    assert not check_eligibility(orders["A1003"], "LP-020", item_unused=True, original_packaging=True).eligible
    assert not check_eligibility(orders["A1001"], "GC-50", item_unused=True, original_packaging=True).eligible
    assert not check_eligibility(orders["A1001"], "HP-200", item_unused=False, original_packaging=True).eligible
    assert not check_eligibility(orders["A1001"], "HP-200", item_unused=True, original_packaging=False).eligible


def test_policy_window_boundary(orders):
    order = orders["A1001"]
    last_day = order.delivered_on + timedelta(days=30)
    assert check_eligibility(order, "HP-200", item_unused=True, original_packaging=True, today=last_day).eligible
    assert not check_eligibility(order, "HP-200", item_unused=True, original_packaging=True,
                                 today=last_day + timedelta(days=1)).eligible


# --- agent + evals: pass path ------------------------------------------------

def test_grounded_answer_passes_and_is_sent():
    fake, http = make_client([
        step(tool("lookup_order", order_id="A1001", email="alice@example.com")),
        step(tool("check_return_eligibility", order_id="A1001", sku="HP-200", item_unused=True, original_packaging=True)),
        step(tool("start_return", order_id="A1001", sku="HP-200", reason="Too small")),
        step(tool("final_answer", message="Your return is started. A prepaid label is on its way.",
                  decision="return_started", order_id="A1001", sku="HP-200")),
    ])
    view = chat(http, "Return my headphones, order A1001, alice@example.com. Unused, in the box.")
    assert view["status"] == "bot"
    assert view["transcript"][-1] == {**view["transcript"][-1], "role": "bot",
                                      "text": "Your return is started. A prepaid label is on its way."}
    assert http.get("/api/escalations", headers=AUTH).json() == []
    assert fake.calls[0]["model"] == "claude-opus-5"


def test_info_only_turn_passes():
    _, http = make_client([step(tool("final_answer", message="Sure, what's your order ID and email?", decision="info_only"))])
    assert chat(http, "Can I return something?")["status"] == "bot"


# --- evals: fail paths escalate ----------------------------------------------

def escalation_flags(http):
    items = http.get("/api/escalations", headers=AUTH).json()
    assert len(items) == 1
    return items[0]


def test_policy_mismatch_escalates_and_withholds_draft():
    _, http = make_client([
        step(tool("lookup_order", order_id="A1002", email="bob@example.com")),
        step(tool("check_return_eligibility", order_id="A1002", sku="JK-310", item_unused=True, original_packaging=True)),
        step(tool("final_answer", message="Good news, your jacket can be returned!", decision="eligible",
                  order_id="A1002", sku="JK-310")),
    ])
    view = chat(http, "Can I return my jacket? A1002 bob@example.com, never worn")
    assert view["status"] == "escalated"
    assert "Good news" not in view["transcript"][-1]["text"]
    esc = escalation_flags(http)
    assert esc["flags"] == ["POLICY_MISMATCH"]
    assert esc["agent_draft"] == "Good news, your jacket can be returned!"


def test_ungrounded_decision_escalates():
    _, http = make_client([step(tool("final_answer", message="Yes, that's eligible.", decision="eligible",
                                     order_id="A1001", sku="HP-200"))])
    chat(http, "is A1001 headphones returnable")
    assert escalation_flags(http)["flags"] == ["POLICY_MISMATCH"]


@pytest.mark.parametrize("message,flag", [
    ("I'll make an exception and waive the window for you.", "UNSUPPORTED_PROMISE"),
    ("I've noted alice@example.com on file.", "PII_LEAK"),
])
def test_reply_content_evals(message, flag):
    _, http = make_client([step(tool("final_answer", message=message, decision="info_only"))])
    chat(http, "hello")
    assert escalation_flags(http)["flags"] == [flag]


def test_customer_asking_for_human_escalates():
    _, http = make_client([step(tool("final_answer", message="Let me help.", decision="info_only"))])
    chat(http, "I want to talk to a manager")
    assert escalation_flags(http)["flags"] == ["HIGH_RISK_OR_HUMAN_REQUESTED"]


def test_agent_needs_human_escalates():
    _, http = make_client([step(tool("final_answer", message="Let me get a colleague.", decision="needs_human"))])
    chat(http, "My parcel arrived damaged and I want compensation for the delay")
    assert escalation_flags(http)["flags"] == ["AGENT_REQUESTED_HUMAN"]


def test_api_error_escalates():
    request = SimpleNamespace(method="POST", url="https://api.anthropic.com/v1/messages")
    err = anthropic.APIConnectionError(request=request)
    _, http = make_client([err])
    view = chat(http, "hello")
    assert view["status"] == "escalated"
    assert escalation_flags(http)["flags"] == ["AGENT_FAILURE"]


def test_refusal_and_plain_text_escalate():
    for resp in (step(stop_reason="refusal"), step(SimpleNamespace(type="text", text="hi"), stop_reason="end_turn")):
        _, http = make_client([resp])
        chat(http, "hello")
        assert escalation_flags(http)["flags"] == ["AGENT_FAILURE"]


# --- tool guards -------------------------------------------------------------

def test_tools_require_verified_order_and_eligibility():
    fake, http = make_client([
        step(tool("lookup_order", order_id="A1001", email="wrong@example.com")),
        step(tool("check_return_eligibility", order_id="A1001", sku="HP-200", item_unused=True, original_packaging=True)),
        step(tool("start_return", order_id="A1001", sku="HP-200", reason="x")),
        step(tool("final_answer", message="I couldn't verify that order. Please check the ID and email.", decision="info_only")),
    ])
    view = chat(http, "return A1001")
    assert view["status"] == "bot"
    tool_results = [m for m in fake.calls[-1]["messages"] if m["role"] == "user" and isinstance(m["content"], list)]
    assert all(r["is_error"] for msg in tool_results[:3] for r in msg["content"])


# --- human handoff -----------------------------------------------------------

def test_human_reply_and_resolve_flow():
    _, http = make_client([
        step(tool("final_answer", message="Let me get a colleague.", decision="needs_human")),
        step(tool("final_answer", message="Anything else?", decision="info_only")),
    ])
    view = chat(http, "complicated problem")
    conv_id = view["conversation_id"]
    esc_id = escalation_flags(http)["id"]

    # While escalated, customer messages wait for the human; the agent is not called.
    assert chat(http, "hello?", conv_id)["status"] == "escalated"

    http.post(f"/api/escalations/{esc_id}/reply", headers=AUTH, json={"message": "Hi, I'm Sam from support."})
    transcript = http.get(f"/api/conversations/{conv_id}").json()["transcript"]
    assert transcript[-1]["role"] == "human"

    assert http.post(f"/api/escalations/{esc_id}/resolve", headers=AUTH).json()["status"] == "bot"
    assert chat(http, "thanks", conv_id)["transcript"][-1]["text"] == "Anything else?"
    assert http.post(f"/api/escalations/{esc_id}/reply", headers=AUTH, json={"message": "x"}).status_code == 409


def test_support_api_requires_token():
    _, http = make_client([])
    assert http.get("/api/escalations").status_code == 401
    assert http.get("/api/escalations", headers={"X-Support-Token": "nope"}).status_code == 401
    unconfigured = TestClient(create_app(client=ScriptedClient([]), support_token=""))
    assert unconfigured.get("/api/escalations", headers=AUTH).status_code == 503


def test_pages_served():
    _, http = make_client([])
    assert "Returns Help" in http.get("/").text
    assert "Support Escalations" in http.get("/support").text
