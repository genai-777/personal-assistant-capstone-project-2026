from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MCPState:
    """
    Shared context passed between agents.
    Acts as the MCP state manager — every agent reads and writes here
    so no agent needs to re-fetch data another already retrieved.
    """
    # Routing
    task_type:       str = ""           # email | meeting | summary | travel | general
    user_input:      str = ""

    # Email agent output
    email_data:      str = ""           # raw email content retrieved
    email_draft:     str = ""           # draft ready for safety check

    # Calendar agent output
    calendar_data:   str = ""           # upcoming meetings

    # Research agent output
    research_output: str = ""           # cited findings
    sources_valid:   bool = False       # passed quality check

    # Meeting brief agent output
    meeting_brief:   str = ""           # final structured brief

    # Safety agent
    safety_approved: bool = False
    safety_reason:   str = ""

    # Final
    final_output:    str = ""
    errors:          list = field(default_factory=list)

    def add_error(self, agent: str, msg: str):
        self.errors.append(f"[{agent}] {msg}")

    def has_errors(self) -> bool:
        return len(self.errors) > 0
