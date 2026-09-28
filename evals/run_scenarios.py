"""Run the returns agent (real Claude API calls) against scripted scenarios and report PASS/FAIL.

Usage:  python -m evals.run_scenarios [scenario_name ...]
Exit code is 1 if any scenario fails.
"""

import json
import sys
from pathlib import Path

import anthropic

from app.agent import ReturnsAgent
from app.orders import load_orders
from app.service import ChatService
from app.store import Store

SCENARIOS = Path(__file__).with_name("scenarios.json")


def run_scenario(service: ChatService, scenario: dict) -> tuple[bool, list[str]]:
    conv = None
    for message in scenario["messages"]:
        conv = service.handle_customer_message(conv.id if conv else None, message)

    problems = []
    if conv.status != scenario["expect_status"]:
        problems.append(f"status {conv.status} != {scenario['expect_status']}")

    last = conv.eval_log[-1] if conv.eval_log else {}
    flags = [f for entry in conv.eval_log for f in entry["flags"]]
    missing = set(scenario.get("expect_flags", [])) - set(flags)
    if missing:
        problems.append(f"missing flags {sorted(missing)} (got {flags})")

    if "expect_decision" in scenario:
        final_decision = _last_decision(conv)
        if final_decision != scenario["expect_decision"]:
            problems.append(f"decision {final_decision} != {scenario['expect_decision']}")
    if problems and last.get("agent_draft"):
        problems.append(f"last draft: {last['agent_draft'][:160]!r}")
    return not problems, problems


def _last_decision(conv) -> str | None:
    """Read the decision from the last final_answer tool call in the Claude history."""
    for message in reversed(conv.messages):
        if message["role"] != "assistant":
            continue
        for block in message["content"]:
            if getattr(block, "type", None) == "tool_use" and block.name == "final_answer":
                return block.input.get("decision")
    return None


def main(names: list[str]) -> int:
    scenarios = json.loads(SCENARIOS.read_text(encoding="utf-8"))
    if names:
        scenarios = [s for s in scenarios if s["name"] in names]
    agent = ReturnsAgent(anthropic.Anthropic(), load_orders())

    failures = 0
    for scenario in scenarios:
        ok, problems = run_scenario(ChatService(agent, Store()), scenario)
        failures += not ok
        print(f"{'PASS' if ok else 'FAIL'}  {scenario['name']}")
        for p in problems:
            print(f"      - {p}")
    print(f"\n{len(scenarios) - failures}/{len(scenarios)} scenarios passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
