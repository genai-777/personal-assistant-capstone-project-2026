import logging
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from tools.search_tool import web_search
from .mcp_state import MCPState

logger = logging.getLogger(__name__)

SYSTEM = """You are a research agent. Your only tool is web_search.

For every search:
1. Run a targeted query on the topic
2. Evaluate each result — prefer: official docs, reputable blogs, recent publications
3. Discard: forums, undated pages, low-authority domains
4. Return findings as a numbered list with Title and URL for every item

If results are weak (no authoritative sources found), say:
"LOW QUALITY: [reason]. Suggest refining query: [better query]"

Always cite sources. Never fabricate facts.
"""


class ResearchAgent:
    MAX_RETRIES = 2

    def __init__(self, llm: ChatOpenAI):
        tools         = [web_search]
        prompt        = ChatPromptTemplate.from_messages([
            ("system", SYSTEM),
            ("human",  "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ])
        agent         = create_openai_tools_agent(llm, tools, prompt)
        self.executor = AgentExecutor(
            agent=agent, tools=tools,
            verbose=True, max_iterations=5, handle_parsing_errors=True
        )

    def run(self, state: MCPState, topic: str) -> MCPState:
        logger.info(f"ResearchAgent searching: {topic}")
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                result = self.executor.invoke({"input": topic})
                output = result["output"]
                if output.startswith("LOW QUALITY") and attempt < self.MAX_RETRIES:
                    logger.warning(f"Attempt {attempt}: low quality, retrying...")
                    topic = _extract_refined_query(output) or topic
                    continue
                state.research_output = output
                state.sources_valid   = not output.startswith("LOW QUALITY")
                break
            except Exception as e:
                state.add_error("ResearchAgent", str(e))
                state.research_output = f"Research error: {e}"
                state.sources_valid   = False
                break
        return state


def _extract_refined_query(text: str) -> str:
    for line in text.splitlines():
        if "Suggest refining query:" in line:
            return line.split("Suggest refining query:")[-1].strip()
    return ""
