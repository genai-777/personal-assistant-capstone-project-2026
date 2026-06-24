import os
import pickle
import logging
from pathlib import Path
from datetime import datetime, timedelta
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from langchain.tools import tool

logger = logging.getLogger(__name__)

SCOPES     = ["https://www.googleapis.com/auth/calendar"]

_BASE      = Path(__file__).parent.parent.parent
CREDS_PATH = str(_BASE / "credentials" / "credentials.json")
TOKEN_PATH = str(_BASE / "credentials" / "calendar_token.pickle")
TIMEZONE   = os.getenv("CALENDAR_TIMEZONE", "America/Los_Angeles")


def get_calendar_service():
    creds = None

    logger.info(f"Looking for calendar token at: {TOKEN_PATH}")
    logger.info(f"Looking for credentials at: {CREDS_PATH}")

    if os.path.exists(TOKEN_PATH):
        with open(TOKEN_PATH, "rb") as f:
            creds = pickle.load(f)
        logger.info("Calendar token loaded")
    else:
        logger.warning("No calendar token found — will trigger OAuth")

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            logger.info("Refreshing expired calendar token")
            creds.refresh(Request())
        else:
            if not os.path.exists(CREDS_PATH):
                raise FileNotFoundError(
                    f"credentials.json not found at {CREDS_PATH}. "
                    "Download from Google Cloud Console."
                )
            logger.info("Starting OAuth flow for Calendar...")
            flow  = InstalledAppFlow.from_client_secrets_file(CREDS_PATH, SCOPES)
            creds = flow.run_local_server(port=0)

        os.makedirs(os.path.dirname(TOKEN_PATH), exist_ok=True)
        with open(TOKEN_PATH, "wb") as f:
            pickle.dump(creds, f)
        logger.info("Calendar token saved")

    return build("calendar", "v3", credentials=creds)


@tool
def get_upcoming_meetings(days_ahead: int = 1) -> str:
    """
    Retrieve upcoming calendar meetings for the next N days.
    Returns title, time, location, attendees, description.
    Use days_ahead=1 for today, days_ahead=7 for the week.
    """
    try:
        logger.info(f"Fetching calendar for next {days_ahead} day(s)")
        service = get_calendar_service()
        now     = datetime.utcnow()
        end     = now + timedelta(days=int(days_ahead))

        logger.info(f"Querying events from {now.isoformat()} to {end.isoformat()}")

        result  = service.events().list(
            calendarId="primary",
            timeMin=now.isoformat() + "Z",
            timeMax=end.isoformat() + "Z",
            singleEvents=True,
            orderBy="startTime",
        ).execute()

        events = result.get("items", [])
        logger.info(f"Found {len(events)} calendar event(s)")

        if not events:
            return f"No meetings scheduled in the next {days_ahead} day(s)."

        out = []
        for e in events:
            start     = e["start"].get("dateTime", e["start"].get("date"))
            end_time  = e["end"].get("dateTime",   e["end"].get("date"))
            attendees = [a.get("email", "") for a in e.get("attendees", [])]
            out.append(
                f"📅 {e.get('summary', 'Untitled')}\n"
                f"   Start: {_fmt(start)}  End: {_fmt(end_time)}\n"
                f"   Location: {e.get('location', 'N/A')}\n"
                f"   Attendees: {', '.join(attendees) or 'None'}\n"
                f"   Description: {e.get('description', '')[:300]}"
            )
        return "\n\n---\n\n".join(out)

    except Exception as e:
        logger.error(f"Calendar error: {e}", exc_info=True)
        return f"Calendar error: {e}"


@tool
def create_calendar_event(
    title: str, start_datetime: str, end_datetime: str,
    description: str = "", attendees: str = ""
) -> str:
    """
    Create a Google Calendar event.
    SAFETY: always confirm with user before calling.
    Datetimes in ISO 8601: '2024-06-10T14:00:00'
    """
    try:
        service       = get_calendar_service()
        attendee_list = [{"email": e.strip()} for e in attendees.split(",") if e.strip()]
        event = {
            "summary":     title,
            "description": description,
            "start":  {"dateTime": start_datetime, "timeZone": TIMEZONE},
            "end":    {"dateTime": end_datetime,   "timeZone": TIMEZONE},
            "attendees": attendee_list,
        }
        created = service.events().insert(calendarId="primary", body=event).execute()
        return f"✅ Event created: {title}\n   Link: {created.get('htmlLink', 'N/A')}"
    except Exception as e:
        logger.error(f"Create event error: {e}", exc_info=True)
        return f"Error creating event: {e}"


def _fmt(iso: str) -> str:
    try:
        return datetime.fromisoformat(iso.replace("Z", "+00:00")).strftime("%a %b %d %I:%M %p")
    except Exception:
        return iso


def get_calendar_tools():
    return [get_upcoming_meetings, create_calendar_event]