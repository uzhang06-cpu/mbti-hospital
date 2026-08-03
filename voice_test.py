# -*- coding: utf-8 -*-
"""严格评测：语音消息(TTS)行为。名字含 _test → 被 .dockerignore 排除，不进镜像。
用 sys.modules 桩掉重依赖，import app 后 mock 掉 LLM/TTS/DB/socket，真实跑 trigger_dm_reply。"""
import sys, types, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

def stub(name, **attrs):
    m = types.ModuleType(name)
    for k, v in attrs.items(): setattr(m, k, v)
    sys.modules[name] = m
    return m

# ── 桩掉重依赖 ──
ev = stub("eventlet", monkey_patch=lambda *a, **k: None)
tp = stub("eventlet.tpool", execute=lambda fn, *a, **k: fn(*a, **k)); ev.tpool = tp
stub("dotenv", load_dotenv=lambda *a, **k: None)
stub("imageio_ffmpeg", get_ffmpeg_exe=lambda: "ffmpeg")
stub("dashscope")
stub("requests", post=lambda *a, **k: None, get=lambda *a, **k: None)
class _OpenAI:
    def __init__(self, *a, **k): self.api_key = k.get("api_key", "")
stub("openai", OpenAI=_OpenAI)
stub("bson", ObjectId=object)
class _App:
    def __init__(self, *a, **k): pass
    def register_blueprint(self, *a, **k): pass
    def route(self, *a, **k):
        def d(fn): return fn
        return d
class _Req:
    headers = {}
    form = {}
    files = None
    def get_json(self, *a, **k): return {}
stub("flask", Flask=_App, render_template=lambda *a, **k: "", jsonify=lambda *a, **k: ("json", a, k), request=_Req())
class _SIO:
    def __init__(self, *a, **k): pass
    def on(self, *a, **k):
        def d(fn): return fn
        return d
    def emit(self, *a, **k): pass
    def sleep(self, *a, **k): pass
    def start_background_task(self, fn, *a, **k): pass   # 不跑链式接话，隔离首条回复
stub("flask_socketio", SocketIO=_SIO, join_room=lambda *a, **k: None)
stub("db_mongo", get_db=lambda: None, JWT_SECRET="x")
stub("auth_routes", auth_bp=object(), verify_token=lambda t: None)

import app  # noqa

# ── Fake DB ──
class Coll:
    def __init__(self): self.docs = []
    def find(self, q=None, sort=None, limit=None):
        res = list(self.docs)
        if sort:
            k, d = sort[0]; res.sort(key=lambda x: x.get(k) or 0, reverse=(d < 0))
        if limit: res = res[:limit]
        return res
    def find_one(self, q=None, sort=None):
        r = self.find(q, sort); return r[0] if r else None
    def insert_one(self, doc):
        self.docs.append(doc)
        return types.SimpleNamespace(inserted_id="fakeid")
    def count_documents(self, q=None): return len(self.docs)
    def update_one(self, *a, **k): pass
    def create_index(self, *a, **k): pass
class DB:
    def __init__(self): self.c = {}
    def __getitem__(self, k): return self.c.setdefault(k, Coll())

FAKE = DB()
app.get_mdb = lambda: FAKE

# ── mock LLM / TTS / 图片 / socket ──
CALLS = {"tokens": [], "audio_text": []}
EMITS = []
REPLY = "不是 我这不是还没反应过来嘛 ||| 你拆教授论证这么大事我肯定得认真看啊 ||| 先让我把你那堆看完"
def fake_llm(role, messages, max_tokens=200, temperature=1.0, system_prompt=""):
    CALLS["tokens"].append(max_tokens)
    return REPLY
def fake_tts(text, role):
    CALLS["audio_text"].append(text)
    return "/static/audio/fake.mp3"
app.call_llm = fake_llm
app.generate_audio_url = fake_tts
app.search_image_url = lambda *a, **k: ""
app.socketio.emit = lambda event, payload, **k: EMITS.append(payload)
app.socketio.sleep = lambda *a, **k: None
app.socketio.start_background_task = lambda fn, *a, **k: None

# ── 评测：强制语音的私聊回复 ──
def run_case():
    CALLS["tokens"].clear(); CALLS["audio_text"].clear(); EMITS.clear()
    app.trigger_dm_reply("ENTP", "发条语音说说你的想法", "u1", "u1")
    full = [e for e in EMITS if e.get("mode") == "full"]
    text_bubbles = [e for e in full if e.get("content") and not e.get("audio") and not e.get("image")]
    voice_msgs = [e for e in full if e.get("audio")]
    return full, text_bubbles, voice_msgs

full, text_bubbles, voice_msgs = run_case()
print("=" * 60)
print("发出的正式消息条数 :", len(full))
print("纯文字气泡          :", [e["content"] for e in text_bubbles])
print("带语音的消息        :", [(e.get("content"), e.get("audio")) for e in voice_msgs])
print("传给TTS的文本        :", CALLS["audio_text"])
print("call_llm 的 max_tokens:", CALLS["tokens"])
print("=" * 60)

# ── 断言（期望修复后成立）──
fails = []
if len(voice_msgs) != 1:
    fails.append(f"应恰好1条语音消息，实际{len(voice_msgs)}")
if text_bubbles:
    fails.append(f"语音模式下不应再单独发文字气泡，实际发了{len(text_bubbles)}条：{[e['content'] for e in text_bubbles]}")
if voice_msgs and CALLS["audio_text"]:
    tts = CALLS["audio_text"][-1].strip()
    trans = (voice_msgs[0].get("content") or "").strip()
    if tts != trans:
        fails.append(f"音频文本≠转写：\n  TTS ='{tts}'\n  转写='{trans}'")
if CALLS["tokens"] and min(CALLS["tokens"]) < 110:
    fails.append(f"max_tokens 太低会截断：{CALLS['tokens']}（应≥110）")

print("结果:", "[PASS] 全部通过" if not fails else "[FAIL] 失败:")
for f in fails: print("   -", f)

# ── 群聊(单人AI群)路径同样验证 ──
print("\n--- 群聊路径 ---")
try:
    EMITS.clear(); CALLS["audio_text"].clear(); CALLS["tokens"].clear()
    app.chat_counter = 0
    app.trigger_ai_reply("ENTP", "INFJ", "发条语音说说你的想法", "u1", "u1")
    gfull = [e for e in EMITS if e.get("mode") == "full"]
    gtext = [e for e in gfull if e.get("content") and not e.get("audio") and not e.get("image")]
    gvoice = [e for e in gfull if e.get("audio")]
    print("正式消息=%d 文字气泡=%d 语音=%d" % (len(gfull), len(gtext), len(gvoice)))
    print("传给TTS:", CALLS["audio_text"])
    g = []
    if len(gvoice) != 1: g.append(f"应1条语音，实际{len(gvoice)}")
    if gtext: g.append(f"语音模式不应发文字气泡，实际{len(gtext)}")
    if gvoice and CALLS["audio_text"] and CALLS["audio_text"][-1].strip() != (gvoice[0].get("content") or "").strip():
        g.append("音频文本≠转写")
    print("群聊结果:", "[PASS]" if not g else "[FAIL] " + "; ".join(g))
    fails += g
except Exception as e:
    import traceback; traceback.print_exc(); print("群聊用例运行异常:", e)

# ── 非语音路径不能回归：应发文字气泡、无音频、不调用TTS ──
print("\n--- 非语音(文字)路径 ---")
_orig_rand = app.random.random
app.random.random = lambda: 0.99   # 压掉18%/8%随机语音
try:
    EMITS.clear(); CALLS["audio_text"].clear()
    app.trigger_dm_reply("ENTP", "你怎么看这事", "u1", "u1")
    tfull = [e for e in EMITS if e.get("mode") == "full"]
    ttext = [e for e in tfull if e.get("content") and not e.get("audio")]
    tvoice = [e for e in tfull if e.get("audio")]
    print("文字气泡=%d 语音=%d TTS调用=%d" % (len(ttext), len(tvoice), len(CALLS["audio_text"])))
    t = []
    if len(ttext) < 1: t.append("文字模式应至少发1条文字气泡")
    if tvoice: t.append("文字模式不应出现语音")
    if CALLS["audio_text"]: t.append("文字模式不应调用TTS")
    print("非语音结果:", "[PASS]" if not t else "[FAIL] " + "; ".join(t))
    fails += t
finally:
    app.random.random = _orig_rand

sys.exit(1 if fails else 0)

