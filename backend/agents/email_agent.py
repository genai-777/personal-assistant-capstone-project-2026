import logging
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from tools.gmail_tool import get_gmail_tools
from .mcp_state import MCPState

logger = logging.getLogger(__name__)

SYSTEM = """You are an email management agent.

Tools available: read_emails, get_email_thread, draft_email.
You do NOT have access to send_email — drafting is your boundary.

Workflow:
1. Call read_emails to see unread messages
2. Call get_email_thread for full context before drafting any reply
3. Call draft_email to save a draft — never attempt to send directly

Tone preference: {email_tone}

Be concise. Summarise threads before drafting. Always confirm the recipient is correct.
"""


class EmailAgent:
    """
    Handles inbox reading and draft creation.
    Does NOT send — send_email is gated by the Safety Agent.
    """

    def __init__(self, llm: ChatOpenAI):
        self.tools    = get_gmail_tools()
        # Remove send_email from this agent's toolset (tool access limit guardrail)
        self.tools    = [t for t in self.tools if t.name != "send_email"]
        prompt        = ChatPromptTemplate.from_messages([
            ("system", SYSTEM),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ])
        agent         = create_openai_tools_agent(llm, self.tools, prompt)
        self.executor = AgentExecutor(
            agent=agent, tools=self.tools,
            verbose=True, max_iterations=6, handle_parsing_errors=True
        )

    def run(self, state: MCPState, chat_history: list, email_tone: str) -> MCPState:
        logger.info("EmailAgent running...")
        try:
            result           = self.executor.invoke({
                "input":        state.user_input,
                "chat_history": chat_history,
                "email_tone":   email_tone,
            })
            state.email_draft = result["output"]
        except Exception as e:
            state.add_error("EmailAgent", str(e))
            state.email_draft = f"Email agent error: {e}"
        return state
