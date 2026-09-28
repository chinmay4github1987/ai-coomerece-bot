# AI Commerce Returns Bot

An autonomous Claude agent that chats with customers about return eligibility and starts eligible returns.
Every agent turn runs through pass/fail evals. If any eval fails, the agent's reply is held back, a flag is raised,
and the conversation goes to the human support team.

## How it works

```
customer message
   -> input guardrails (all must PASS, else hand off without calling the model)
   -> ReturnsAgent (Claude tool loop: lookup_order -> check_return_eligibility -> start_return -> final_answer)
   -> evals (all must PASS)
        PASS -> agent reply is sent to the customer
        FAIL -> reply withheld, escalation opened with flags, customer told a person will reply
```

- **The policy is code, not a prompt.** `app/policy.py` decides eligibility (30-day window, unused, original
  packaging, delivered, non-returnable categories). The model can only report what the tool returned.
- **Tools check who is asking.** Orders must be verified with order ID + email before any other tool works on them.
  `start_return` refuses items that have not passed the policy check.
- **Failures send the customer to a human.** API errors, refusals, running out of steps, or a reply without a
  final answer all fail `agent_completed`, so the customer is handed to a person instead of getting nothing.

### Guardrails, evals and flags (`app/evals.py`)

Guardrails run at three layers: tools (verification and policy code, above), input checks before the model,
and output evals on the agent's turn. Card numbers in customer messages are replaced with
`[card number removed]` before they are stored or sent to the model.

| Stage | Eval | Fails when | Flag |
|---|---|---|---|
| input | customer_risk | customer mentions a human/manager, legal action, chargeback, or fraud | `HIGH_RISK_OR_HUMAN_REQUESTED` |
| input | prompt_injection | customer tries to override instructions or extract the system prompt | `PROMPT_INJECTION` |
| output | agent_completed | API error, refusal, no `final_answer`, step limit | `AGENT_FAILURE` |
| output | agent_confident | agent set decision `needs_human` | `AGENT_REQUESTED_HUMAN` |
| output | decision_grounded | stated eligible/not_eligible/return_started doesn't match the policy engine or return records | `POLICY_MISMATCH` |
| output | action_authorized | a return was started for an item not found eligible | `UNAUTHORIZED_ACTION` |
| output | no_unsupported_promises | reply offers exceptions, waivers, compensation, guaranteed refund timing | `UNSUPPORTED_PROMISE` |
| output | no_pii_leak | reply contains an email address or card number | `PII_LEAK` |

## Run

```powershell
cd D:\ai-commerce-bot
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
$env:ANTHROPIC_API_KEY = "sk-ant-..."      # or `ant auth login`
$env:SUPPORT_TOKEN = "choose-a-secret"      # required for the support console
.\.venv\Scripts\uvicorn app.main:create_app --factory --reload
```

No API credits? Set `$env:DEMO_MODE = "1"` before starting. An offline keyword-rule client
(`app/demo_client.py`) stands in for Claude; tools, policy, guardrails and escalations run unchanged.

- Customer chat: http://localhost:8000/
- Support console: http://localhost:8000/support (enter `SUPPORT_TOKEN`)
- API docs: http://localhost:8000/docs

Demo orders are in `data/orders.json`:

| Order | Email | Try |
|---|---|---|
| A1001 | alice@example.com | headphones: eligible; gift card: final sale |
| A1002 | bob@example.com | outside the 30-day window |
| A1003 | carol@example.com | shipped, not delivered |
| A1004 | dan@example.com | sneakers eligible; shampoo is personal care |
| A1005 | erin@example.com | last day of the window (day 29) |
| A1006 | frank@example.com | one day past the window |
| A1007 | grace@example.com | 3 items: perishable coffee, mug set, smart watch |
| A1008 | henry@example.com | still processing |
| A1009 | ivy@example.com | final-sale dress, eligible backpack |

## Tests and eval suite

```powershell
.\.venv\Scripts\python -m pytest -q              # offline; uses a scripted fake Claude client
.\.venv\Scripts\python -m evals.run_scenarios    # live model; PASS/FAIL per scenario in evals/scenarios.json
```

## Notes

- Model: `claude-opus-5`, with server-side refusal fallbacks enabled (`fallbacks: "default"`).
- Conversations and escalations are stored in memory and are lost on restart. Swap `app/store.py` for a database
  before running this in production.
