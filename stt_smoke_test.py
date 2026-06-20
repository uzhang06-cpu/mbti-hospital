import json
import os

import requests


def post_file(path: str):
    with open(path, "rb") as f:
        files = {"audio": (os.path.basename(path), f, "application/octet-stream")}
        r = requests.post("http://127.0.0.1:5001/api/voice_to_text", files=files, timeout=60)
        return r.status_code, r.text


def main():
    base = os.path.join("static", "temp_audio")
    candidates = [
        "voice_32fe89a0-4a19-46cd-9702-54be9da51ccd.wav",
        "voice_dd6c21fe-96b1-4261-8217-cd3360783237.wav",
        "voice_32fe89a0-4a19-46cd-9702-54be9da51ccd.webm",
    ]
    paths = [os.path.join(base, c) for c in candidates if os.path.exists(os.path.join(base, c))]
    if not paths:
        raise SystemExit("no test audio files found")

    out = []
    for p in paths:
        code, text = post_file(p)
        out.append({"file": p, "http": code, "body": text})
        print("FILE", p, "HTTP", code, "LEN", len(text), flush=True)
        try:
            obj = json.loads(text)
            preview = json.dumps(obj, ensure_ascii=False)[:400]
            print("PREVIEW", preview, flush=True)
        except Exception:
            print("PREVIEW", text[:400], flush=True)
        print("-" * 60, flush=True)

    with open("stt_smoke_out.jsonl", "w", encoding="utf-8") as f:
        for row in out:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
