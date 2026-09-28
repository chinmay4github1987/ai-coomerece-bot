"""Chat flow: input guardrails -> agent turn -> output evals -> reply to customer, or hand off to the human support queue."""

import logging

from app.agent import ReturnsAgent, TurnResult
from app.evals import as_dicts, redact_input, run_evals, run_input_evals
from app.store import Conversation, Escalation, Store, now_iso

log = logging.getLogger(__name__)

HANDOFF_MESSAGE = ("Thanks for your patience. I've passed your conversation to our support team, "
                   "and a team member will reply here shortly.")


class ChatService:
    def __init__(self, agent: ReturnsAgent, store: Store):
        self.agent = agent
        self.store = store

    def handle_customer_message(self, conversation_id: str | None, text: str) -> Conversation:
        conv = self.store.get_or_create(conversation_id)
        text = redact_input(text)
        with conv.lock:
            conv.say("customer", text)
            if conv.status == "escalated":
                # A human owns this conversation now; keep the agent's history complete for when it returns.
                conv.messages.append({"role": "user", "content": text})
                return conv

            results = run_input_evals(text)
            if all(r.passed for r in results):
                turn = self.agent.run_turn(conv, text)
                results += run_evals(conv, text, turn)
            else:
                # Blocked before the model: keep the agent's history complete, but don't call it.
                conv.messages.append({"role": "user", "content": text})
                turn = TurnResult()
            flags = [r.flag for r in results if not r.passed]
            conv.eval_log.append({"at": now_iso(), "customer_message": text, "passed": not flags,
                                  "flags": flags, "evals": as_dicts(results), "tool_calls": turn.tool_calls,
                                  "agent_draft": turn.draft})

            if not flags:
                conv.say("bot", turn.final["message"])
                return conv

            # Failed evals: withhold the agent's draft and hand off.
            esc = self.store.open_escalation(conv, flags, as_dicts(results), turn.draft)
            log.warning("Escalated %s as %s: %s", conv.id, esc.id, ", ".join(flags))
            conv.status = "escalated"
            conv.say("bot", HANDOFF_MESSAGE)
            return conv

    def human_reply(self, esc: Escalation, text: str) -> Conversation:
        conv = self.store.get(esc.conversation_id)
        with conv.lock:
            conv.say("human", text)
            # Keep the agent's context complete if the conversation returns to it.
            conv.messages.append({"role": "user", "content": f"[Support agent replied to the customer]: {text}"})
        return conv

    def resolve(self, esc: Escalation) -> Conversation:
        conv = self.store.get(esc.conversation_id)
        with conv.lock:
            esc.status, esc.resolved_at = "resolved", now_iso()
            if not any(e.status == "open" for e in self.store.list_escalations() if e.conversation_id == conv.id):
                conv.status = "bot"
        return conv
