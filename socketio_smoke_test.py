import time

import socketio


def main():
    received = set()
    s = socketio.Client(logger=False, engineio_logger=False)

    @s.event
    def connect():
        print("connected", flush=True)

    @s.event
    def disconnect():
        print("disconnected", flush=True)

    @s.on("ai_message")
    def on_ai(d):
        role = (d or {}).get("role")
        mode = (d or {}).get("mode")
        content = ((d or {}).get("content") or "").replace("\n", " ")
        if mode == "full" and role:
            received.add(role)
            print("AI_FULL", role, content[:80], flush=True)

    s.connect("http://127.0.0.1:5001", transports=["polling"])
    s.emit("user_message", {"content": "@ENFP @INFP @ENTP @ENFJ 测试一下都回一句"})

    deadline = time.time() + 20
    while time.time() < deadline:
        if received.issuperset({"ENFP", "INFP", "ENTP", "ENFJ"}):
            break
        time.sleep(0.25)

    print("RECEIVED", sorted(received), flush=True)
    s.disconnect()
    return 0 if received.issuperset({"ENFP", "INFP", "ENTP", "ENFJ"}) else 2


if __name__ == "__main__":
    raise SystemExit(main())

