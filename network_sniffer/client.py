"""Resilient HTTP delivery for normalised packet events."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import requests


LOGGER = logging.getLogger(__name__)


class BackendClient:
    """Small, failure-isolated client for the FastAPI analysis endpoint.

    A temporary API outage must not terminate packet capture. Failed event
    delivery is logged and the capture loop continues; the backend owns
    persistent audit logging once an event arrives.
    """

    def __init__(self, base_url: str, timeout_seconds: float, verify_tls: bool = True) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.verify_tls = verify_tls
        self._session = requests.Session()

    def submit(self, event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Submit one event, returning structured API output if delivery succeeds."""

        try:
            response = self._session.post(
                self.base_url,
                json=event,
                timeout=self.timeout_seconds,
                verify=self.verify_tls,
            )
            response.raise_for_status()
            payload = response.json()
            return payload if isinstance(payload, dict) else {"response": payload}
        except requests.RequestException as exc:
            LOGGER.warning("Unable to deliver packet event to %s: %s", self.base_url, exc)
        except ValueError:
            LOGGER.warning("Backend at %s returned a non-JSON response", self.base_url)
        return None

    def close(self) -> None:
        self._session.close()
