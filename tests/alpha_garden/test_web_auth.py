import unittest
from unittest.mock import patch

from alpha_garden.web_auth import ConsoleAuth, SESSION_SECONDS


class ConsoleAuthTests(unittest.TestCase):
    def test_sessions_expire_are_revoked_and_do_not_survive_service_restart(self):
        auth = ConsoleAuth()
        with patch("alpha_garden.web_auth.time.monotonic", return_value=100):
            token = auth.login("admin", "123123")
            cookie = auth.cookie(token)
            self.assertTrue(auth.authenticated(cookie))
            self.assertIn("HttpOnly", cookie)
            self.assertIn("SameSite=Strict", cookie)
            self.assertNotIn("123123", cookie)
            self.assertFalse(ConsoleAuth().authenticated(cookie))
            self.assertFalse(auth.authenticated("alpha_garden_session=forged"))
        with patch("alpha_garden.web_auth.time.monotonic", return_value=100 + SESSION_SECONDS):
            self.assertFalse(auth.authenticated(cookie))
        token = auth.login("admin", "123123")
        auth.logout(auth.cookie(token))
        self.assertFalse(auth.authenticated(auth.cookie(token)))
        self.assertIn("Max-Age=0", auth.cookie())
