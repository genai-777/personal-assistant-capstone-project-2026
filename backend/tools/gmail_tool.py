import os
import base64
import pickle
from pathlib import Path
from email.mime.text import MIMEText
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from langchain.tools import tool

SCOPES     = ["https://www.googleapis.com/auth/gmail.readonly",
               "https://www.googleapis.com/auth/gmail.send",
               "https://www.googleapis.com/auth/gmail.compose"]

# Absolute paths — works regardless of which folder you run from
_BASE      = Path(__file__).parent.parent.parent
CREDS_PATH = str(_BASE / "credentials" / "credentials.json")
TOKEN_PATH = str(_BASE / "credentials" / "gmail_token.pickle")


def get_gmail_service():
    creds = None
    if os.path.exists(TOKEN_PATH):
        with open(TOKEN_PATH, "rb") as f:
            creds = pickle.load(f)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists(CREDS_PATH):
                raise FileNotFoundError(
                    f"credentials.json not found at {CREDS_PATH}. "
                    "Download from Google Cloud Console and place in the credentials/ folder."
                )
            flow  = InstalledAppFlow.from_client_secrets_file(CREDS_PATH, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(TOKEN_PATH, "wb") as f:
            pickle.dump(creds, f)
    return build("gmail", "v1", credentials=creds)


@tool
def read_emails(max_results: int = 5) -> str:
    """Read latest unread emails. Returns sender, date, subject, preview."""
    try:
        service  = get_gmail_service()
        results  = service.users().messages().list(
            userId="me", labelIds=["UNREAD"], maxResults=max_results
        ).execute()
        messages = results.get("messages", [])
        if not messages:
            return "No unread emails."
        out = []
        for msg in messages:
            d       = service.users().messages().get(userId="me", id=msg["id"]).execute()
            headers = d["payload"]["headers"]
            subject = next((h["value"] for h in headers if h["name"] == "Subject"), "No Subject")
            sender  = next((h["value"] for h in headers if h["name"] == "From"),    "Unknown")
            date    = next((h["value"] for h in headers if h["name"] == "Date"),    "Unknown")
            snippet = d.get("snippet", "")
            out.append(f"ID: {msg['id']}\nFrom: {sender}\nDate: {date}\nSubject: {subject}\nPreview: {snippet}")
        return "\n\n---\n\n".join(out)
    except Exception as e:
        return f"Error reading emails: {e}"


@tool
def get_email_thread(message_id: str) -> str:
    """Get full body of a specific email by ID. Call before drafting a reply."""
    try:
        service = get_gmail_service()
        msg     = service.users().messages().get(userId="me", id=message_id, format="full").execute()
        headers = msg["payload"]["headers"]
        subject = next((h["value"] for h in headers if h["name"] == "Subject"), "No Subject")
        sender  = next((h["value"] for h in headers if h["name"] == "From"),    "Unknown")
        date    = next((h["value"] for h in headers if h["name"] == "Date"),    "Unknown")
        body    = _extract_body(msg["payload"])
        return f"From: {sender}\nDate: {date}\nSubject: {subject}\n\nBody:\n{body}"
    except Exception as e:
        return f"Error retrieving email: {e}"


@tool
def draft_email(to: str, subject: str, body: str) -> str:
    """
    Save email as Gmail draft. Does NOT send — always draft first.
    Returns draft ID and content for user review.
    """
    try:
        service = get_gmail_service()
        message = MIMEText(body)
        message["to"]      = to
        message["subject"] = subject
        raw   = base64.urlsafe_b64encode(message.as_bytes()).decode()
        draft = service.users().drafts().create(
            userId="me", body={"message": {"raw": raw}}
        ).execute()
        return (
            f"✅ Draft saved (ID: {draft['id']})\n\n"
            f"To: {to}\nSubject: {subject}\n\n{body}\n\n"
            f"Say 'send it' to send, or ask me to revise."
        )
    except Exception as e:
        return f"Error creating draft: {e}"


@tool
def send_email(to: str, subject: str, body: str) -> str:
    """
    Send email immediately. SAFETY: only call after explicit user confirmation.
    Always use draft_email first.
    """
    try:
        service = get_gmail_service()
        message = MIMEText(body)
        message["to"]      = to
        message["subject"] = subject
        raw  = base64.urlsafe_b64encode(message.as_bytes()).decode()
        sent = service.users().messages().send(userId="me", body={"raw": raw}).execute()
        return f"✅ Email sent. Message ID: {sent['id']}"
    except Exception as e:
        return f"Error sending email: {e}"


def _extract_body(payload: dict) -> str:
    if "parts" in payload:
        for part in payload["parts"]:
            if part["mimeType"] == "text/plain":
                data = part["body"].get("data", "")
                if data:
                    return base64.urlsafe_b64decode(data).decode("utf-8")
        return _extract_body(payload["parts"][0])
    data = payload.get("body", {}).get("data", "")
    return base64.urlsafe_b64decode(data).decode("utf-8") if data else "(No body)"


def get_gmail_tools():
    return [read_emails, get_email_thread, draft_email, send_email]
