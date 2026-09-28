"""FastAPI app: customer chat API, support-team escalation API, and the two web pages."""

import logging
import os
import secrets
from dataclasses import asdict
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.agent import ReturnsAgent
from app.orders import load_orders
from app.service import ChatService
from app.store import Store

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


class ChatIn(BaseModel):
    conversation_id: str | None = Field(default=None, max_length=64)
    message: str = Field(min_length=1, max_length=2000)


class ReplyIn(BaseModel):
    message: str = Field(min_length=1, max_length=4000)


def create_app(client=None, support_token: str | None = None) -> FastAPI:
    if client is None:
        import anthropic
        client = anthropic.Anthropic()
    support_token = support_token if support_token is not None else os.environ.get("SUPPORT_TOKEN", "")

    store = Store()
    service = ChatService(ReturnsAgent(client, load_orders()), store)
    app = FastAPI(title="AI Commerce Returns Bot")

    def require_support(x_support_token: str = Header(default="")) -> None:
        if not support_token:
            raise HTTPException(503, "SUPPORT_TOKEN is not configured on the server")
        if not secrets.compare_digest(x_support_token.encode(), support_token.encode()):
            raise HTTPException(401, "Invalid support token")

    def get_escalation(escalation_id: str):
        esc = store.escalations.get(escalation_id)
        if esc is None:
            raise HTTPException(404, "Escalation not found")
        return esc

    # --- customer ---------------------------------------------------------

    @app.post("/api/chat")
    def chat(body: ChatIn) -> dict:
        return service.handle_customer_message(body.conversation_id, body.message.strip()).view()

    @app.get("/api/conversations/{conversation_id}")
    def conversation(conversation_id: str) -> dict:
        conv = store.get(conversation_id)
        if conv is None:
            raise HTTPException(404, "Conversation not found")
        return conv.view()

    # --- support team -----------------------------------------------------

    @app.get("/api/escalations", dependencies=[Depends(require_support)])
    def escalations(status: str | None = "open") -> list[dict]:
        return [asdict(e) for e in store.list_escalations(status or None)]

    @app.get("/api/escalations/{escalation_id}", dependencies=[Depends(require_support)])
    def escalation_detail(escalation_id: str) -> dict:
        esc = get_escalation(escalation_id)
        conv = store.get(esc.conversation_id)
        return {**asdict(esc), "conversation": conv.view(), "eval_log": conv.eval_log,
                "eligibility": conv.eligibility, "returns": conv.returns,
                "verified_orders": sorted(conv.verified_orders)}

    @app.post("/api/escalations/{escalation_id}/reply", dependencies=[Depends(require_support)])
    def reply(escalation_id: str, body: ReplyIn) -> dict:
        esc = get_escalation(escalation_id)
        if esc.status != "open":
            raise HTTPException(409, "Escalation is already resolved")
        return service.human_reply(esc, body.message.strip()).view()

    @app.post("/api/escalations/{escalation_id}/resolve", dependencies=[Depends(require_support)])
    def resolve(escalation_id: str) -> dict:
        return service.resolve(get_escalation(escalation_id)).view()

    # --- pages ------------------------------------------------------------

    @app.get("/", include_in_schema=False)
    def customer_page():
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/support", include_in_schema=False)
    def support_page():
        return FileResponse(STATIC_DIR / "support.html")

    return app
