import requests
import json
from typing import Optional

class CortexClient:
    def __init__(self, cortex_url: str = "http://localhost:8000"):
        self.cortex_url = cortex_url
        self.session = requests.Session()

    def send_transcript(self, text: str) -> bool:
        """Send transcription to Mellanrum Cortex"""
        url = f"{self.cortex_url}/api/transcripts"
        payload = {"text": text}

        try:
            response = self.session.post(
                url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=5
            )
            response.raise_for_status()
            return True
        except requests.exceptions.RequestException as e:
            print(f"Failed to send transcript: {str(e)}")
            return False

    def get_status(self) -> Optional[dict]:
        """Check connection to Cortex"""
        url = f"{self.cortex_url}/api/status"

        try:
            response = self.session.get(url, timeout=5)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Failed to get status: {str(e)}")
            return None
