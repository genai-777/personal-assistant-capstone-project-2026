import logging
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from .mcp_state import MCPState

logger = logging.getLogger(__name__)

# ── ToT system prompt ─────────────────────────────────────────────────────────

TOT_SYSTEM = """You are a meeting preparation agent using Tree of Thought reasoning.

Given a meeting topic and research findings, explore THREE candidate architectural options.
For each option, reason through trade-offs across these criteria:
  - Alignment with meeting constraints (weight: 30%)
  - Industry support from research (weight: 25%)
  - Implementation feasibility (weight: 25%)
  - Risk level — security, cost, vendor lock-in (weight: 20%)

Score each option 0–100 based on the criteria above.
Prune any option scoring below 50.
Select the highest scoring option as the recommendation.

Output this exact structure:

## Meeting Brief: [Meeting Title]

### Best Practices (with citations)
[numbered list from research, each with source URL]

### Option Analysis (ToT)
**Option A — [name]**: [trade-off summary] | Score: X/100
**Option B — [name]**: [trade-off summary] | Score: X/100
**Option C — [name]**: [trade-off summary] | Score: X/100

### Recommended Approach
[top-scoring option with justification]

### Key Clarifying Questions
1. [question]
2. [question]
3. [question]

### Architectural Decision Matrix
| Factor | Option A | Option B | Option C |
|---|---|---|---|
| Performance | | | |
| Cost | | | |
| Security | | | |
| Complexity | | | |
| Vendor lock-in | | | |

If research quality is low, flag it clearly before the brief.
"""


class MeetingBriefAgent:
    """
    Generates structured meeting briefs using ToT reasoning.
    Hosts the 3-option beam search (K=2) decision matrix.
    """

    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

    def run(self, state: MCPState) -> MCPState:
        logger.info("MeetingBriefAgent generating brief with ToT...")

        quality_warning = ""
        if not state.sources_valid:
            quality_warning = (
                "⚠️ Research sources could not be fully verified. "
                "The brief below is based on available information — "
                "please validate key claims before the meeting.\n\n"
            )

        prompt = (
            f"Meeting context:\n{state.calendar_data}\n\n"
            f"Research findings:\n{state.research_output}\n\n"
            f"User request:\n{state.user_input}"
        )

        try:
            response          = self.llm.invoke([
                SystemMessage(content=TOT_SYSTEM),
                HumanMessage(content=prompt),
            ])
            state.meeting_brief = quality_warning + response.content
        except Exception as e:
            state.add_error("MeetingBriefAgent", str(e))
            state.meeting_brief = f"Brief generation error: {e}"

        return state
