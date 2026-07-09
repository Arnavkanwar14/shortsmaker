"""YouTube upload via the free YouTube Data API v3.

One-time setup (owner does this once):
  1. console.cloud.google.com -> new project -> enable "YouTube Data API v3"
  2. OAuth consent screen: External, add yourself as a test user
  3. Credentials -> Create -> OAuth client ID -> type "Desktop app"
  4. Download the JSON, save it in the project root as
     `youtube_client_secret.json` (gitignored)

Then "Connect YouTube" in the UI runs the consent flow once and saves a
refresh token to `youtube_token.json` (gitignored); uploads reuse it.

Quota: uploading costs ~1600 units of the default 10,000/day -> ~6
uploads/day, which matches a Shorts posting cadence.
"""
from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger("shortsmaker")

SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]
ROOT = Path(__file__).resolve().parent.parent
CLIENT_SECRET = ROOT / "youtube_client_secret.json"
TOKEN_FILE = ROOT / "youtube_token.json"

TITLE_MAX = 100          # YouTube hard limits
DESC_MAX = 4900
TAGS_CHARS_MAX = 460


def is_configured() -> bool:
    """True once the owner has dropped in the OAuth client secret."""
    return CLIENT_SECRET.is_file()


def _load_creds():
    """Return valid Credentials, refreshing if needed, or None."""
    if not TOKEN_FILE.is_file():
        return None
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    creds = Credentials.from_authorized_user_file(str(TOKEN_FILE), SCOPES)
    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
            TOKEN_FILE.write_text(creds.to_json(), encoding="utf-8")
        except Exception as e:
            log.warning("YouTube token refresh failed: %s", e)
            return None
    return creds if creds and creds.valid else None


def is_authorized() -> bool:
    try:
        return _load_creds() is not None
    except Exception:
        return False


def connect() -> None:
    """Run the one-time consent flow: opens a browser, saves the token.
    Blocks until the owner finishes consent in the browser."""
    if not is_configured():
        raise RuntimeError(
            "youtube_client_secret.json not found in the project root -- "
            "create an OAuth 'Desktop app' client (see youtube.py docstring).")
    from google_auth_oauthlib.flow import InstalledAppFlow
    flow = InstalledAppFlow.from_client_secrets_file(str(CLIENT_SECRET), SCOPES)
    # Desktop-app clients allow loopback redirects on any port automatically
    creds = flow.run_local_server(port=0, open_browser=True,
                                  prompt="consent",
                                  authorization_prompt_message="")
    TOKEN_FILE.write_text(creds.to_json(), encoding="utf-8")
    log.info("YouTube connected -- token saved to %s", TOKEN_FILE.name)


def build_description(metadata: dict | None, fallback_title: str) -> tuple[str, str, list[str]]:
    """(title, description, tags) ready for the YouTube API.

    The metadata's `description` is already the complete text (hook, story,
    related line, CTA, hashtag block from highlights.finalize_metadata), so
    this mostly enforces YouTube's limits and guarantees a shorts tag."""
    md = metadata or {}
    title = (md.get("title") or fallback_title or "Short clip")[:TITLE_MAX]

    # prefer the dedicated keyword tags; fall back to the hashtags
    tags = [t.lstrip("#") for t in (md.get("tags") or md.get("hashtags") or []) if t]
    if not any(t.lower() == "shorts" for t in tags):
        tags.insert(0, "shorts")
    trimmed, total = [], 0
    for t in tags:                       # YouTube's ~500-char tag budget
        if len(trimmed) >= 15 or total + len(t) + 2 > TAGS_CHARS_MAX:
            break
        trimmed.append(t)
        total += len(t) + 2

    description = (md.get("description") or title)
    if "#shorts" not in description.lower():
        description = f"{description}\n\n#Shorts".strip()
    return title, description[:DESC_MAX], trimmed


def upload(video_path: Path, title: str, description: str, tags: list[str],
           privacy: str = "private", category_id: str = "22") -> str:
    """Upload a video, return its youtube.com/shorts/<id> URL.
    privacy: private | unlisted | public. category 22 = People & Blogs."""
    creds = _load_creds()
    if creds is None:
        raise RuntimeError("not connected to YouTube -- click Connect first")
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload

    youtube = build("youtube", "v3", credentials=creds, cache_discovery=False)
    body = {
        "snippet": {"title": title, "description": description,
                    "tags": tags, "categoryId": category_id},
        "status": {"privacyStatus": privacy, "selfDeclaredMadeForKids": False},
    }
    media = MediaFileUpload(str(video_path), mimetype="video/mp4", resumable=True)
    request = youtube.videos().insert(part="snippet,status", body=body,
                                      media_body=media)
    response = None
    while response is None:
        _, response = request.next_chunk()
    vid = response["id"]
    log.info("uploaded to YouTube: %s (%s)", vid, privacy)
    return f"https://www.youtube.com/shorts/{vid}"
