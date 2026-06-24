import logging
import os
from pathlib import Path
from dotenv import load_dotenv

# Explicitly load .env from project root
env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from memory.memory_store  import MemoryStore
from tools.gmail_tool     import send_email
from tools.calendar_tool  import get_upcoming_meetings

from .mcp_state           import MCPState
from .email_agent         import EmailAgent
from .research_agent      import ResearchAgent
from .meeting_brief_agent import MeetingBriefAgent
from .safety_agent        import SafetyAgent

logger = logging.getLogger(__name__)

# ── router ────────────────────────────────────────────────────────────────────

ROUTER_SYSTEM = """You are a task router for a personal productivity assistant.

Classify the user request into exactly one of these task types:
  email    — reading, drafting, replying to, or managing email / inbox
  calendar — checking schedule, listing meetings, asking what is on today/tomorrow/this week
  meeting  — preparing a meeting BRIEF, researching a meeting topic, decision matrix, prep notes
  summary  — daily summary, daily recap, what did I do today
  travel   — flights, hotels, trip planning, booking
  research — user wants to search or research a specific topic, product, technology, or feature
  general  — anything else

Rules:
- "what meetings do I have" or "what is on my calendar" -> calendar
- "prepare a brief" or "research my next meeting" or "prep for my meeting" -> meeting
- "search for", "look up", "what is", "tell me about [product/technology]" -> research
- When in doubt between calendar and meeting, choose calendar

Respond with ONLY the single task type word. No explanation."""

# ── system prompts ────────────────────────────────────────────────────────────

CALENDAR_SYSTEM = """You are a helpful assistant presenting calendar information clearly.

Given the user's upcoming meetings from Google Calendar, present them in a clean, 
easy-to-read format. Include:
- Meeting title
- Date and time
- Attendees (if any)
- Location or video link (if any)
- Description (if any)

After listing meetings, ask if they would like a full meeting brief prepared for any of them."""

SUMMARY_SYSTEM = """You are a productivity assistant generating a daily summary.

Given today's activity log and current calendar data, produce:
1. Meetings today — title and time
2. Completed tasks — what was done
3. Open action items — unresolved threads or pending tasks
4. Next steps — prioritised list for tomorrow

Be concise. Flag anything urgent."""

GENERAL_SYSTEM = """You are a helpful personal productivity assistant.
Answer clearly and concisely based on the user preferences below.

User preferences:
{preferences}"""


class Orchestrator:
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4-turbo", temperature=0,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        self.memory         = MemoryStore()
        self.chat_history   = []
        self.email_agent    = EmailAgent(self.llm)
        self.research_agent = ResearchAgent(self.llm)
        self.brief_agent    = MeetingBriefAgent(self.llm)
        self.safety_agent   = SafetyAgent(self.llm)

    # ── public ────────────────────────────────────────────────────────────────

    def chat(self, message: str) -> str:
        state           = MCPState(user_input=message)
        state.task_type = self._classify(message)
        prefs           = self.memory.get_preferences()

        logger.info(f"Task type: {state.task_type} | Input: {message[:60]}")

        if state.task_type == "email":
            state = self._handle_email(state, prefs)
        elif state.task_type == "calendar":
            state = self._handle_calendar(state)
        elif state.task_type == "meeting":
            state = self._handle_meeting(state)
        elif state.task_type == "summary":
            state = self._handle_summary(state)
        elif state.task_type == "research":
            state = self._handle_research(state)
        else:
            state = self._handle_general(state, prefs)

        self.chat_history.extend([
            HumanMessage(content=message),
            AIMessage(content=state.final_output),
        ])
        if len(self.chat_history) > 20:
            self.chat_history = self.chat_history[-20:]

        self.memory.log_task(state.task_type, message[:150], state.final_output[:300])
        return state.final_output

    def reset(self):
        self.chat_history = []

    def _handle_research(self, state: MCPState) -> MCPState:
        """Direct web research — search and summarise for any topic."""
        logger.info(f"Research handler: {state.user_input[:80]}")
        state = self.research_agent.run(state, state.user_input)

        try:
            response = self.llm.invoke([
                SystemMessage(content=(
                    "You are a helpful assistant. Summarise the web research findings "
                    "below clearly and concisely. Cite every source with its URL. "
                    "Do not use your training knowledge — use only what is in the findings."
                )),
                HumanMessage(content=(
                    f"Research findings:\n{state.research_output}\n\n"
                    f"User asked: {state.user_input}"
                )),
            ])
            state.final_output = response.content
        except Exception as e:
            state.final_output = state.research_output  # fallback to raw
        return state

    # ── handlers ──────────────────────────────────────────────────────────────

    def _handle_calendar(self, state: MCPState) -> MCPState:
        """Simply fetch and display calendar — no research, no brief."""
        try:
            # Determine days ahead from user input
            days = 7 if any(w in state.user_input.lower() for w in ["week", "7 day"]) else 1
            state.calendar_data = get_upcoming_meetings.run(str(days))
        except Exception as e:
            state.add_error("Orchestrator", f"Calendar fetch failed: {e}")
            state.calendar_data = "Calendar unavailable."

        try:
            response = self.llm.invoke([
                SystemMessage(content=CALENDAR_SYSTEM),
                HumanMessage(content=(
                    f"User asked: {state.user_input}\n\n"
                    f"Calendar data:\n{state.calendar_data}"
                )),
            ])
            state.final_output = response.content
        except Exception as e:
            state.final_output = state.calendar_data  # fallback to raw data
        return state

    def _handle_email(self, state: MCPState, prefs: dict) -> MCPState:
        state = self.email_agent.run(
            state, self.chat_history, prefs.get("email_tone", "professional")
        )
        if state.email_draft and "draft" in state.email_draft.lower():
            state = self.safety_agent.review(state, state.email_draft)
            if state.safety_approved:
                state.final_output = (
                    state.email_draft +
                    "\n\n✅ Safety check passed. Reply 'send it' to send."
                )
            else:
                state.final_output = (
                    f"⚠️ Safety Agent blocked this action:\n{state.safety_reason}\n\n"
                    "Please clarify your request."
                )
        else:
            state.final_output = state.email_draft
        return state

    def _handle_meeting(self, state: MCPState) -> MCPState:
        """Full pipeline: calendar + research + ToT brief."""
        # Step 1: fetch calendar
        try:
            state.calendar_data = get_upcoming_meetings.run("2")
        except Exception as e:
            state.add_error("Orchestrator", f"Calendar fetch failed: {e}")
            state.calendar_data = "Calendar unavailable."

        # Step 2: build a rich research query from meeting title + agenda
        topic = _build_research_query(state.calendar_data, state.user_input)
        logger.info(f"Research query: {topic}")
        state = self.research_agent.run(state, topic)

        # Step 3: generate brief with ToT
        state = self.brief_agent.run(state)
        state.final_output = state.meeting_brief
        return state

    def _handle_summary(self, state: MCPState) -> MCPState:
        daily_context = self.memory.get_daily_context()
        try:
            calendar_today = get_upcoming_meetings.run("1")
        except Exception:
            calendar_today = "Calendar unavailable."

        try:
            response = self.llm.invoke([
                SystemMessage(content=SUMMARY_SYSTEM),
                HumanMessage(content=(
                    f"Today's activity log:\n{daily_context}\n\n"
                    f"Calendar:\n{calendar_today}\n\n"
                    f"User request:\n{state.user_input}"
                )),
            ])
            state.final_output = response.content
        except Exception as e:
            state.add_error("Orchestrator", str(e))
            state.final_output = f"Summary error: {e}"
        return state

    def _handle_general(self, state: MCPState, prefs: dict) -> MCPState:
        # If the question looks like it needs current/specific knowledge,
        # run web research first before answering
        if _needs_web_search(state.user_input):
            logger.info("General handler: routing through Research Agent for current info")
            state = self.research_agent.run(state, state.user_input)
            context = f"Web research findings:\n{state.research_output}\n\n"
        else:
            context = ""

        try:
            response = self.llm.invoke([
                SystemMessage(content=GENERAL_SYSTEM.format(
                    preferences=self.memory.format_preferences()
                )),
                *self.chat_history,
                HumanMessage(content=(
                    f"{context}"
                    f"User question: {state.user_input}\n\n"
                    f"IMPORTANT: If web research findings are provided above, "
                    f"base your answer on those findings and cite sources. "
                    f"Do not rely on your training data for product-specific questions."
                )),
            ])
            state.final_output = response.content
        except Exception as e:
            state.add_error("Orchestrator", str(e))
            state.final_output = f"Error: {e}"
        return state

    # ── classifier ────────────────────────────────────────────────────────────

    def _classify(self, message: str) -> str:
        # Keyword pre-check before hitting the LLM — faster and more reliable
        msg = message.lower()

        # Calendar keywords — simple schedule check
        if any(w in msg for w in [
            "what meetings", "what's on", "whats on", "my schedule",
            "do i have today", "do i have tomorrow", "do i have this week",
            "show my calendar", "check my calendar", "upcoming meetings"
        ]):
            return "calendar"

        # Meeting brief keywords
        if any(w in msg for w in [
            "prepare a brief", "meeting brief", "prep for", "research my meeting",
            "brief for", "prepare for my", "meeting prep", "decision matrix"
        ]):
            return "meeting"

        # Email keywords
        if any(w in msg for w in [
            "email", "inbox", "reply", "draft", "unread", "message", "send"
        ]):
            return "email"

        # Summary keywords
        if any(w in msg for w in [
            "daily summary", "summary", "recap", "what did i do", "today's tasks"
        ]):
            return "summary"

        # Research keywords
        if any(w in msg for w in [
            "search for", "look up", "what is", "tell me about",
            "how does", "explain", "find info", "research"
        ]):
            return "research"

        # Travel keywords
        if any(w in msg for w in ["flight", "hotel", "travel", "trip", "book"]):
            return "travel"

        # Fall back to LLM classification for ambiguous inputs
        try:
            resp = self.llm.invoke([
                SystemMessage(content=ROUTER_SYSTEM),
                HumanMessage(content=message),
            ])
            task = resp.content.strip().lower()
            valid = {"email", "calendar", "meeting", "summary", "travel", "general"}
            return task if task in valid else "general"
        except Exception:
            return "general"


# ── helpers ───────────────────────────────────────────────────────────────────

def _build_research_query(calendar_data: str, user_input: str) -> str:
    """
    Build a detailed research query from the meeting title, agenda,
    and user input so the Research Agent runs targeted searches.
    """
    lines   = calendar_data.splitlines() if calendar_data else []
    title   = ""
    agenda  = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("📅"):
            title = stripped.replace("📅", "").strip()
        # Capture agenda items — lines starting with - or bullet or agenda keywords
        if any(stripped.startswith(p) for p in ["-", "•", "*", "Agenda", "agenda"]):
            agenda.append(stripped.lstrip("-•* "))

    # Build the research prompt
    parts = []
    if title:
        parts.append(f"Meeting topic: {title}")
    if agenda:
        parts.append("Agenda items:\n" + "\n".join(f"- {a}" for a in agenda[:6]))
    if user_input and user_input.lower() not in ["prepare a brief", "meeting brief"]:
        parts.append(f"Additional context from user: {user_input}")

    parts.append(
        "Please research: best practices, implementation strategies, "
        "known limitations, and any relevant industry frameworks or tools."
    )

    return "\n\n".join(parts) if parts else user_input


def _needs_web_search(message: str) -> bool:
    """
    Returns True if the question likely needs current web information
    rather than GPT-4 training data.
    """
    msg = message.lower()
    signals = [
        "salesforce", "agentforce", "servicenow", "zendesk",
        "what is", "how does", "tell me about", "explain",
        "latest", "new feature", "release", "version",
        "omni channel", "omnichannel", "routing", "implementation",
    ]
    return any(s in msg for s in signals)