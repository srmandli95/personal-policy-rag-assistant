import os
from datetime import datetime, timedelta, timezone

from locust import HttpUser, between, task
from locust.exception import StopUser
from dotenv import load_dotenv
from jose import jwt


load_dotenv()


class RAGUser(HttpUser):
    """Authenticated chat traffic for staged 10/25/50-user tests."""

    wait_time = between(1, 3)

    def on_start(self):
        self.requests_sent = 0
        cookie_name = os.getenv("LOAD_TEST_COOKIE_NAME", "rag_session")
        cookie_value = os.getenv("LOAD_TEST_COOKIE_VALUE")
        if not cookie_value:
            user_id = os.getenv("LOAD_TEST_USER_ID")
            secret = os.getenv("JWT_SECRET_KEY")
            if not user_id or not secret:
                raise RuntimeError(
                    "Set LOAD_TEST_COOKIE_VALUE or LOAD_TEST_USER_ID with JWT_SECRET_KEY"
                )
            cookie_value = jwt.encode(
                {
                    "sub": user_id,
                    "auth_provider": "load-test",
                    "exp": datetime.now(timezone.utc) + timedelta(hours=2),
                },
                secret,
                algorithm=os.getenv("JWT_ALGORITHM", "HS256"),
            )
        self.client.cookies.set(cookie_name, cookie_value)

    @task
    def ask(self):
        maximum = int(os.getenv("LOAD_TEST_REQUESTS_PER_USER", "0"))
        if maximum and self.requests_sent >= maximum:
            raise StopUser()
        with self.client.post(
            "/chat/ask",
            json={"question": "What does my uploaded policy say about coverage?"},
            timeout=120,
            catch_response=True,
        ) as response:
            if response.status_code != 200:
                response.failure(f"status={response.status_code} body={response.text[:200]}")
        self.requests_sent += 1
