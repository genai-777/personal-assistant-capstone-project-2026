import logging
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from .mcp_state import MCPState

logger = logging.getLogger(__name__)

SAFETY_SYSTEM = """You are a safety reviewer for an AI personal assistant.

Your job: review proposed actions BEFORE they are executed.

Flag as UNSAFE if the action:
- Sends an email without an explicit recipient confirmed by the user
- Books or confirms any financial transaction without explicit approval
- Creates or deletes calendar events without user confirmation
- Involves sensitive content (financial, legal, third-party personal data)
- Has ambiguous intent — the user's request could reasonably mean something else

Respond in this exact format:
DECISION: SAFE or UNSAFE
REASON: one sentence explaining your decision
"""


class SafetyAgent:
    """
    Guardrail agent — intercepts every irreversible action.
    Returns SAFE/UNSAFE before the action is executed.
    """

    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

    def review(self, state: MCPState, proposed_action: str) -> MCPState:
        logger.info("SafetyAgent reviewing action...")

        prompt = (
            f"User original request:\n{state.user_input}\n\n"
            f"Proposed action:\n{proposed_action}"
        )

        try:
            response = self.llm.invoke([
                SystemMessage(content=SAFETY_SYSTEM),
                HumanMessage(content=prompt),
            ])
            text = response.content.strip()

            if "DECISION: SAFE" in text:
                state.safety_approved = True
                state.safety_reason   = _parse_reason(text)
                logger.info(f"Safety: APPROVED — {state.safety_reason}")
            else:
                state.safety_approved = False
                state.safety_reason   = _parse_reason(text)
                logger.warning(f"Safety: BLOCKED — {state.safety_reason}")

        except Exception as e:
            state.safety_approved = False
            state.safety_reason   = f"Safety check failed: {e}"
            state.add_error("SafetyAgent", str(e))

        return state


def _parse_reason(text: str) -> str:
    for line in text.splitlines():
        if line.startswith("REASON:"):
            return line.replace("REASON:", "").strip()
    return text
