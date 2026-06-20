import json
import os
import sys

import requests


def main():
    path = os.path.join("static", "temp_audio", "voice_32fe89a0-4a19-46cd-9702-54be9da51ccd.webm")
    if not os.path.exists(path):
        path = os.path.join("static", "temp_audio", "voice_32fe89a0-4a19-46cd-9702-54be9da51ccd.wav")
    if not os.path.exists(path):
        return 3

    with open(path, "rb") as f:
        files = {"audio": (os.path.basename(path), f, "application/octet-stream")}
        r = requests.post("http://127.0.0.1:5001/api/voice_to_text", files=files, timeout=120)
    if r.status_code != 200:
        return 4

    try:
        obj = json.loads(r.text)
    except Exception:
        return 5

    if obj.get("ok") is True and (obj.get("text") or "").strip():
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())

