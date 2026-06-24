# Personal Productivity Assistant

A 5-agent AI system for email management, meeting preparation, daily summaries,
and travel coordination. Built with LangChain, OpenAI GPT-4, and FastAPI.

---

## Architecture

```
User (browser)
     │
     ▼
FastAPI  /chat
     │
     ▼
Orchestrator  ──► EmailAgent        ──► Gmail API
              ──► ResearchAgent     ──► Tavily (web search)
              ──► MeetingBriefAgent ──► ToT reasoning
              ──► SafetyAgent       ──► approves irreversible actions
              │
              └── MCPState          ──► shared context between agents
                  MemoryStore       ──► preferences.json + task_history.json
```

---

## Project Structure

```
personal_assistant/
├── backend/
│   ├── main.py                    # FastAPI server
│   ├── agents/
│   │   ├── orchestrator.py        # routes tasks to agents
│   │   ├── email_agent.py         # inbox + drafts (no send access)
│   │   ├── research_agent.py      # web search + source quality check
│   │   ├── meeting_brief_agent.py # ToT decision matrix
│   │   ├── safety_agent.py        # reviews irreversible actions
│   │   └── mcp_state.py           # shared state across agents
│   ├── tools/
│   │   ├── gmail_tool.py
│   │   ├── calendar_tool.py
│   │   └── search_tool.py
│   ├── memory/
│   │   └── memory_store.py
│   └── requirements.txt
├── frontend/
│   └── index.html
├── credentials/                   # ← git-ignored, your Google OAuth files go here
├── memory/                        # ← git-ignored, auto-created on first run
├── .env.example
└── .gitignore
```

---

## Setup — Step by Step

### Step 1 — Clone and create virtual environment

```bash
git clone <your-repo>
cd personal_assistant

python3 -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

### Step 2 — Install dependencies

```bash
cd backend
pip install -r requirements.txt
```

### Step 3 — Set up environment variables

```bash
cd ..                           # back to project root
cp .env.example .env
```

Open `.env` and fill in:

```
OPENAI_API_KEY=sk-...           # https://platform.openai.com/api-keys
TAVILY_API_KEY=tvly-...         # https://app.tavily.com  (free tier)
CALENDAR_TIMEZONE=America/Los_Angeles
```

### Step 4 — Set up Google credentials (Gmail + Calendar)

1. Go to [Google Cloud Console](https://console.cloud.google.com)
2. Create a new project
3. Go to **APIs & Services → Library** and enable:
   - **Gmail API**
   - **Google Calendar API**
4. Go to **APIs & Services → Credentials**
5. Click **Create Credentials → OAuth 2.0 Client ID**
   - Application type: **Desktop app**
   - Name: anything (e.g. "Personal Assistant")
6. Click **Download JSON**
7. Rename the file to `credentials.json` and place it at:

```
personal_assistant/credentials/credentials.json
```

> **First run only:** A browser window will open asking you to sign in with Google.
> After approval, a token file is saved locally — you won't be prompted again.

### Step 5 — Run the app

```bash
cd backend
source ../venv/bin/activate
python main.py
```

Open your browser at: **http://localhost:8000**

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET`   | `/`                   | Chat UI |
| `POST`  | `/chat`               | Send message to agent |
| `POST`  | `/reset`              | Clear session history |
| `GET`   | `/preferences`        | View stored preferences |
| `PATCH` | `/preferences/{key}`  | Update a preference |
| `GET`   | `/history`            | Recent task log |
| `GET`   | `/health`             | Server status + loaded agents |

### Update a preference (example)

```bash
curl -X PATCH http://localhost:8000/preferences/email_tone \
  -H "Content-Type: application/json" \
  -d '{"value": "casual and direct"}'
```

---

## Example Prompts

```
Check my unread emails
Draft a reply to the email from John about the Q3 report
What meetings do I have today?
Prepare a brief for my API gateway design review
Give me a daily summary
```

---

## Safety Guardrails

| Guardrail | How it works |
|---|---|
| Draft-before-send | Email Agent has no access to `send_email`; sending requires explicit user confirmation routed through Safety Agent |
| Tool access limits | Each agent only holds the tools its role needs |
| Source verification | Research Agent rejects low-quality results and retries before passing findings to Meeting Brief Agent |
| Safety Agent | Reviews all irreversible actions (send, create event, confirm booking) before execution |
| Audit trail | Every tool call and output is logged to `memory/task_history.json` |

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `credentials.json not found` | Download from Google Cloud Console and place in `credentials/` |
| `OPENAI_API_KEY not set` | Check your `.env` file is in the project root |
| `TavilySearchResults` error | Confirm `TAVILY_API_KEY` is set in `.env` |
| Port 8000 already in use | Change `APP_PORT=8001` in `.env` |
| Google auth browser doesn't open | Run `python main.py` from a machine with a browser, not a headless server |
