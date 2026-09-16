"""Local console credentials and expiring, server-owned browser sessions."""
import secrets
import threading
import time
from http.cookies import CookieError, SimpleCookie


COOKIE_NAME = "alpha_garden_session"
SESSION_SECONDS = 12 * 60 * 60


class ConsoleAuth:
    def __init__(self):
        self._sessions = {}
        self._lock = threading.Lock()

    def login(self, username, password):
        if not isinstance(username, str) or not isinstance(password, str):
            return None
        if not (secrets.compare_digest(username.encode(), b"admin")
                and secrets.compare_digest(password.encode(), b"123123")):
            return None
        token = secrets.token_urlsafe(32)
        now = time.monotonic()
        with self._lock:
            self._sessions = {key: expires for key, expires in self._sessions.items() if expires > now}
            self._sessions[token] = now + SESSION_SECONDS
        return token

    def authenticated(self, cookie):
        token = self._token(cookie)
        with self._lock:
            expires = self._sessions.get(token, 0)
            if expires <= time.monotonic():
                self._sessions.pop(token, None)
                return False
            return True

    def logout(self, cookie):
        with self._lock:
            self._sessions.pop(self._token(cookie), None)

    @staticmethod
    def _token(header):
        cookie = SimpleCookie()
        try:
            cookie.load(header or "")
        except CookieError:
            return ""
        return cookie[COOKIE_NAME].value if COOKIE_NAME in cookie else ""

    @staticmethod
    def cookie(token=""):
        return f"{COOKIE_NAME}={token}; Path=/api; HttpOnly; SameSite=Strict; Max-Age={SESSION_SECONDS if token else 0}"
