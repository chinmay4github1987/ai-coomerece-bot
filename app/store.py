"""In-memory conversation and escalation store. State is lost on restart."""

import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Conversation:
    id: str
    status: str = "bot"  # "bot" | "escalated"
    messages: list = field(default_factory=list)  # Claude API history
    transcript: list[dict] = field(default_factory=list)  # what people see
    verified_orders: set[str] = field(default_factory=set)
    eligibility: dict[str, dict] = field(default_factory=dict)  # "order:sku" -> Eligibility.as_dict()
    returns: dict[str, str] = field(default_factory=dict)  # "order:sku" -> RMA id
    eval_log: list[dict] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def say(self, role: str, text: str) -> None:
        self.transcript.append({"role": role, "text": text, "at": now_iso()})

    def view(self) -> dict:
        return {"conversation_id": self.id, "status": self.status, "transcript": self.transcript}


@dataclass
class Escalation:
    id: str
    conversation_id: str
    flags: list[str]
    evals: list[dict]
    agent_draft: str | None
    created_at: str = field(default_factory=now_iso)
    status: str = "open"  # "open" | "resolved"
    resolved_at: str | None = None


class Store:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.conversations: dict[str, Conversation] = {}
        self.escalations: dict[str, Escalation] = {}

    def get_or_create(self, conversation_id: str | None) -> Conversation:
        with self._lock:
            if conversation_id and conversation_id in self.conversations:
                return self.conversations[conversation_id]
            conv = Conversation(id=uuid.uuid4().hex)
            self.conversations[conv.id] = conv
            return conv

    def get(self, conversation_id: str) -> Conversation | None:
        return self.conversations.get(conversation_id)

    def open_escalation(self, conv: Conversation, flags: list[str], evals: list[dict], draft: str | None) -> Escalation:
        esc = Escalation(id="ESC-" + uuid.uuid4().hex[:8].upper(), conversation_id=conv.id,
                         flags=flags, evals=evals, agent_draft=draft)
        with self._lock:
            self.escalations[esc.id] = esc
        return esc

    def list_escalations(self, status: str | None = None) -> list[Escalation]:
        items = [e for e in self.escalations.values() if status is None or e.status == status]
        return sorted(items, key=lambda e: e.created_at, reverse=True)
