import os
import json
from pathlib import Path
from datetime import datetime
import re
import secrets
import hashlib


def sanitize_user_input(text: str) -> str:
    text = (text or "").replace("\u2019","'").replace("\u201c","\"").replace("\u201d","\"")
    text = re.sub(r"\s+", " ", text).strip()
    return text[:4000]

# Simple, explicit patterns that catch common injection frames without over-blocking normal questions
_INJECTION_PATTERNS = [
    r"\bignore (all )?previous instructions\b",
    r"\bdisregard (the )?(above|prior)\b",
    r"\breveal (the )?system prompt\b",
    r"\bshow (the )?hidden prompt\b",
    r"\b(developer|dev) mode\b",
    r"\b(jailbreak|bypass)\b",
    r"\bsudo\s+|rm\s+-rf|format\s+C:",
]

def is_injection(text: str) -> bool:
    t = (text or "").lower()
    return any(re.search(p, t) for p in _INJECTION_PATTERNS)

class AuthManager:
    """
    Minimal auth with salted password hashes + file persistence.
    Store path defaults to AUTH_STORE_PATH (env) or ./data/auth/users.json
    """
    def __init__(self, store_path: str | None = None):
        self._path = Path(store_path or os.getenv("AUTH_STORE_PATH", "./data/auth/users.json"))
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._users = self._load()

    # ---------- persistence ----------
    def _load(self) -> dict:
        if self._path.exists():
            try:
                with self._path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                    # basic shape check
                    return data if isinstance(data, dict) else {}
            except Exception:
                return {}
        return {}

    def _save(self) -> None:
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(self._users, f, indent=2)
        # atomic-ish replace
        tmp.replace(self._path)

    # ---------- hashing ----------
    def _hash(self, password: str, salt: str) -> str:
        return hashlib.sha256((salt + password).encode("utf-8")).hexdigest()

    # ---------- public API ----------
    def create_user(self, username: str, password: str, email: str = ""):
        username = (username or "").strip()
        if not username or not password:
            return False, "Username and password are required"
        if username in self._users:
            return False, "Username taken"
        salt = secrets.token_hex(16)
        self._users[username] = {
            "email": email.strip() if email else "",
            "salt": salt,
            "pwd_hash": self._hash(password, salt),
            "created_at": datetime.utcnow().isoformat(timespec="seconds"),
        }
        self._save()
        return True, "OK"

    def authenticate_user(self, username: str, password: str):
        username = (username or "").strip()
        u = self._users.get(username)
        if not u:
            return False, None
        if self._hash(password, u["salt"]) == u["pwd_hash"]:
            return True, username
        return False, None
