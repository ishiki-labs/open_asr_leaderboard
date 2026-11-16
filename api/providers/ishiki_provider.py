import os
from typing import Optional

import requests

from . import APIProvider, PermanentError, register

DEFAULT_API_URL = "https://api.ishikilabs.ai"


@register("ishiki-labs")
class IshikiLabsProvider(APIProvider):
    """Ishiki Labs ASR API. Model variants: "ishiki-asr-0.5". Env: ISHIKI_API_KEY, ISHIKI_API_URL."""

    def transcribe(
        self,
        model_variant: str,
        audio_file_path: Optional[str],
        sample: dict,
        use_url: bool = False,
        language: str = "en",
        prompt: Optional[str] = None,
    ) -> str:
        api_key = os.environ.get("ISHIKI_API_KEY")
        if not api_key:
            raise PermanentError("ISHIKI_API_KEY environment variable not set")

        base = (os.environ.get("ISHIKI_API_URL") or DEFAULT_API_URL).rstrip("/")
        headers = {"Authorization": f"Bearer {api_key}"}
        data = {
            "model": model_variant,
            "language": language or "en",
            "store": "false",
        }
        if use_url:
            audio_bytes = requests.get(sample["row"]["audio"][0]["src"], timeout=60).content
            files = {"file": ("audio.wav", audio_bytes, "audio/wav")}
        else:
            if not audio_file_path:
                raise PermanentError("ishiki-labs provider requires an audio file")
            with open(audio_file_path, "rb") as f:
                files = {"file": (os.path.basename(audio_file_path), f.read(), "audio/wav")}

        r = requests.post(
            f"{base}/v1/transcribe",
            headers=headers,
            files=files,
            data=data,
            timeout=300,
        )
        if r.status_code in (400, 401, 403, 413, 422):
            raise PermanentError(f"ishiki-labs API returned {r.status_code}: {r.text}")
        r.raise_for_status()
        return r.json()["text"]
