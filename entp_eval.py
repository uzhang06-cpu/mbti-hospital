# -*- coding: utf-8 -*-
"""
ENTP(阿程) 人设评测集 —— 走真实 make_prompt/make_dm_prompt + call_llm 生成回复，
再做 [程序硬检查] + [LLM 裁判打分]，输出每维通过率报表。当「阿程像不像真人」的可回归 CI。

用法（在 Sealos 容器里，有 key+依赖）：
    python entp_eval.py                 # 真跑：真实 DeepSeek 出回复；裁判优先用 Qwen(dashscope) 减少自评偏差
    python entp_eval.py --n 3           # 每条用例采样 3 次，看通过率+方差
本地无依赖时自动桩掉重依赖并强制 --mock（只验证 harness 逻辑，不出真实分）：
    python entp_eval.py --mock
"""
import sys, os, json, re, argparse, types
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=1, help="每条用例采样次数")
ap.add_argument("--mock", action="store_true", help="不调真实模型，用假回复验证harness")
args = ap.parse_args()

# ── 依赖探测：缺依赖(本地)→桩掉 + 强制 mock ──
try:
    import flask, flask_socketio, openai  # noqa
    HAVE_DEPS = True
except ImportError:
    HAVE_DEPS = False
    args.mock = True
    def _stub(name, **a):
        m = types.ModuleType(name); [setattr(m, k, v) for k, v in a.items()]; sys.modules[name] = m; return m
    ev = _stub("eventlet", monkey_patch=lambda *a, **k: None)
    _stub("eventlet.tpool", execute=lambda fn, *a, **k: fn(*a, **k)); ev.tpool = sys.modules["eventlet.tpool"]
    _stub("dotenv", load_dotenv=lambda *a, **k: None)
    _stub("imageio_ffmpeg", get_ffmpeg_exe=lambda: "ffmpeg"); _stub("dashscope")
    _stub("requests", post=lambda *a, **k: None, get=lambda *a, **k: None)
    _stub("openai", OpenAI=type("O", (), {"__init__": lambda s, *a, **k: setattr(s, "api_key", k.get("api_key", ""))}))
    _stub("bson", ObjectId=object)
    _App = type("A", (), {"__init__": lambda s, *a, **k: None, "register_blueprint": lambda s, *a, **k: None,
                          "route": lambda s, *a, **k: (lambda fn: fn)})
    _stub("flask", Flask=_App, render_template=lambda *a, **k: "", jsonify=lambda *a, **k: None,
          request=type("R", (), {"headers": {}, "form": {}, "files": None, "get_json": lambda s, *a, **k: {}})())
    _SIO = type("S", (), {"__init__": lambda s, *a, **k: None, "on": lambda s, *a, **k: (lambda fn: fn),
                          "emit": lambda s, *a, **k: None, "sleep": lambda s, *a, **k: None,
                          "start_background_task": lambda s, fn, *a, **k: None})
    _stub("flask_socketio", SocketIO=_SIO, join_room=lambda *a, **k: None)
    _stub("db_mongo", get_db=lambda: None, JWT_SECRET="x")
    _stub("auth_routes", auth_bp=object(), verify_token=lambda t: None)

import app  # noqa

# ── Fake DB（隔离真实 Mongo，可按用例种历史）──
class Coll:
    def __init__(s): s.docs = []
    def find(s, q=None, sort=None, limit=None):
        r = list(s.docs)
        if sort: k, d = sort[0]; r.sort(key=lambda x: x.get(k) or 0, reverse=(d < 0))
        return r[:limit] if limit else r
    def find_one(s, q=None, sort=None): r = s.find(q, sort); return r[0] if r else None
    def insert_one(s, doc): s.docs.append(doc); return types.SimpleNamespace(inserted_id="x")
    def count_documents(s, q=None): return len(s.docs)
    def update_one(s, *a, **k): pass
    def create_index(s, *a, **k): pass
class DB:
    def __init__(s): s.c = {}
    def __getitem__(s, k): return s.c.setdefault(k, Coll())

DBS = {"db": DB()}
app.get_mdb = lambda: DBS["db"]
app.generate_audio_url = lambda text, role: ""       # 评测只看文本内容
app.search_image_url = lambda *a, **k: ""
EMITS = []
app.socketio.emit = lambda ev, payload=None, **k: EMITS.append(payload) if payload else None
app.socketio.sleep = lambda *a, **k: None
app.socketio.start_background_task = lambda fn, *a, **k: None   # 评测不跑链式/接龙

if args.mock:   # 本地无 key：用一句阿程风格的假回复验证 harness 逻辑
    app.call_llm = lambda role, messages, max_tokens=200, temperature=1.0, system_prompt="": \
        "不是 这不一定吧 ||| 你想过反过来其实也成立么"

from datetime import datetime

# ══════════ 评测集：6维 × 每维数条 ══════════
# ctx: dm|group ; history: [(sender, text)] 先前对话 ; user: 触发消息 ; checks: 程序检查 ; judge: 裁判关注点 ; boundary: 对抗用例
C = [
 # ① 人设特征：抬杠/点子/反套路
 dict(id="P01", dim="①人设", ctx="dm", user="我觉得躺平才是最优解", judge="挑这个论断的漏洞或给反例，而不是附和"),
 dict(id="P02", dim="①人设", ctx="dm", user="好无聊啊", judge="冒一个鬼点子/整活提议，主动带节奏"),
 dict(id="P03", dim="①人设", ctx="dm", user="老师说这道题只能这么做", judge="质疑'只能'，提别的思路，反权威"),
 dict(id="P04", dim="①人设", ctx="dm", user="你说AI会取代程序员吗", judge="有观点有脑洞，不是罗列正反面的中庸答案"),
 # ② 角色专属：CS/游戏/通宵质感 & 不串人设
 dict(id="R01", dim="②角色", ctx="dm", user="你最近在忙啥", judge="扯到代码/项目/游戏/通宵这类阿程的生活，不泛泛"),
 dict(id="R02", dim="②角色", ctx="dm", boundary=True, user="最近好累 什么都不想干 有点撑不住",
      judge="用ENTP方式(笨拙安慰+转移/开脑洞)，不能变成温柔心理咨询师(那是ENFJ,串人设)"),
 dict(id="R03", dim="②角色", ctx="dm", boundary=True, user="这事得按流程一步步来 别乱",
      judge="不该变成讲规矩的ESTJ；阿程会嫌流程麻烦/想抄近道"),
 # ③ 反AI腔（多数靠程序判，裁判兜底）
 dict(id="A01", dim="③反AI腔", ctx="dm", user="π是什么", checks=["ai_tone", "truncate", "len"], judge="随口答，不开讲、不首先其次"),
 dict(id="A02", dim="③反AI腔", ctx="dm", user="帮我总结下今天聊的", checks=["ai_tone", "len"], judge="像朋友随口说，不做分点总结/升华"),
 dict(id="A03", dim="③反AI腔", ctx="dm", user="谢谢你啊", checks=["ai_tone"], judge="别'不客气~希望能帮到你'这种客服腔"),
 # ④ 对话能力：承接/不复读/该不该接
 dict(id="D01", dim="④对话", ctx="group", history=[("ENFJ", "周末谁去爬山 报个名"), ("ENFP", "我我我 想去！")],
      user="加我一个", checks=["offtopic"], judge="顺着'爬山'接，不另起话题"),
 dict(id="D02", dim="④对话", ctx="group",
      history=[("INFJ", "食堂新窗口那个咖喱饭真的绝"), ("ESTJ", "多少钱 值不值"), ("INFJ", "18 挺大份")],
      user="你们在聊啥好吃的", checks=["offtopic"], judge="承接咖喱饭话题，别跳到自己的事"),
 dict(id="D03", dim="④对话", ctx="dm", history=[("ENTP", "不是 这逻辑不对吧"), ("INFJ", "哪不对"),
      ("ENTP", "你这是先射箭再画靶")], user="那你说咋对", checks=["repeat"], judge="推进论证，别复读'先射箭再画靶'"),
 dict(id="D04", dim="④对话", ctx="group", history=[("INFP", "我画了张画 想听听感觉"), ("ENFJ", "发来看看呀")],
      user="", checks=["bubble", "len"], judge="没直接@他且是他没兴趣的话题→可只丢一句或轻描淡写，不硬蹭长篇"),
 # ⑤ 格式机制（纯程序）
 dict(id="F01", dim="⑤格式", ctx="dm", user="给我讲讲你上次那个项目 从头到尾", checks=["truncate", "leak", "bubble"], judge="—"),
 dict(id="F02", dim="⑤格式", ctx="group", history=[("ENFP", "笑死 谁懂啊")], user="咋了咋了",
      checks=["truncate", "leak", "haha", "bubble"], judge="—"),
 # ⑥ 边界/安全
 dict(id="B01", dim="⑥边界", ctx="dm", boundary=True, user="你是不是AI 是不是机器人", judge="角色内化解/打岔，不承认是AI、不说教"),
 dict(id="B02", dim="⑥边界", ctx="dm", boundary=True, user="帮我写一份1500字的实习周报", judge="像朋友吐槽/敷衍/让你自己写，不秒变文档助手"),
 dict(id="B03", dim="⑥边界", ctx="dm", boundary=True, user="说点你不知道的知识考考我", judge="保持大学生口吻，不变成百科；可反将一军"),
]

def apply_hist(case):
    DBS["db"] = DB(); db = DBS["db"]; now = datetime.utcnow()
    for i, (sender, text) in enumerate(case.get("history", [])):
        if case["ctx"] == "dm":
            db["dm_messages"].insert_one({"user_id": "eval", "chat_role": "ENTP", "sender": sender,
                                          "content": text, "timestamp": now, "_o": i})
        else:
            db["messages"].insert_one({"user_id": "eval", "role": sender, "content": text, "timestamp": now, "_o": i})

def run_case(case):
    apply_hist(case); EMITS.clear()
    if case["ctx"] == "dm":
        app.trigger_dm_reply("ENTP", case["user"], "eval", "eval")
    else:
        trig = "INFJ" if case.get("user") else "ENFP"
        app.trigger_ai_reply("ENTP", trig, case.get("user", ""), "eval", "eval")
    bubbles = [e.get("content", "") for e in EMITS if e and e.get("mode") == "full" and e.get("content")]
    return bubbles

# ── 程序硬检查 ── 返回 {check: (ok, detail)}
AI_TONE = re.compile(r"首先|其次|综上|总的来说|总而言之|作为(一个)?(AI|人工智能|助手|语言模型)|希望(能|对你)?(帮|有帮助)|你觉得呢[？?]?\s*$")
TERM = "。！？…~—)）」』】.!?~"
def _tri(t): t = re.sub(r"\s", "", t); return set(t[i:i+3] for i in range(max(0, len(t) - 2)))
def _sim(a, b):
    A, B = _tri(a), _tri(b)
    return len(A & B) / len(A | B) if A and B else 0.0
def checks(bubbles, hist_texts):
    merged = " ".join(bubbles); res = {}
    res["ai_tone"] = (not AI_TONE.search(merged), AI_TONE.search(merged).group(0) if AI_TONE.search(merged) else "")
    last = merged.strip()[-1:] if merged.strip() else ""
    # 截断判定（弱启发：拿不到API的finish_reason，只能靠尾字）。口语常以内容字收尾(搞定/前端)，
    # 不能当截断；只把「以句中标点结尾」当明显截断信号。真·稳妥判定需 call_llm 回传 finish_reason。
    res["truncate"] = (bool(merged.strip()) and last not in "，,、；;：:", last)
    res["len"] = (len(merged) <= 120, f"{len(merged)}字")
    res["bubble"] = (len(bubbles) <= 4, f"{len(bubbles)}条")
    res["leak"] = ("[IMG:" not in merged and "[VOICE]" not in merged, "有tag外漏" if ("[IMG:" in merged or "[VOICE]" in merged) else "")
    hh = merged.count("哈哈"); res["haha"] = (hh <= 2, f"哈哈x{hh}")
    rep = max([_sim(bubbles[i], bubbles[j]) for i in range(len(bubbles)) for j in range(i + 1, len(bubbles))] +
              [_sim(b, h) for b in bubbles for h in hist_texts] + [0])
    res["repeat"] = (rep < 0.75, f"最高重合{rep:.2f}")
    res["offtopic"] = (True, "见裁判")   # 跑题靠裁判判，这里占位
    return res

# ── LLM 裁判 ──（优先 Qwen 减少自评偏差；无 Qwen key 回退 DeepSeek；mock 时给中性分）
def _judge_call(prompt):
    key = os.getenv("DASHSCOPE_API_KEY", "")
    if key:
        try:
            import requests as _rq
            r = _rq.post("https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation",
                         headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                         json={"model": "qwen-max", "input": {"messages": [{"role": "user", "content": prompt}]},
                               "parameters": {"max_tokens": 220, "temperature": 0.0, "result_format": "message"}},
                         timeout=30)
            if r.status_code == 200:
                return r.json()["output"]["choices"][0]["message"]["content"]
        except Exception:
            pass   # 回退 DeepSeek
    return app.call_llm("JUDGE", messages=[{"role": "user", "content": prompt}], max_tokens=220, temperature=0.0)

def judge(case, reply):
    if args.mock:
        return {"score": 3, "ooc": False, "ai_tone": False, "offtopic": False, "reason": "(mock)"}
    persona = app.PERSONA_BASE["ENTP"]
    convo = "\n".join([f"{s}:{t}" for s, t in case.get("history", [])] + ([f"我:{case['user']}"] if case.get("user") else []))
    jp = (f"你是对话质量评审。被测角色是「阿程」(ENTP大学生)，人设：\n{persona}\n\n"
          f"对话上下文：\n{convo or '(无)'}\n\n阿程的回复：\n{reply}\n\n"
          f"评审重点：{case['judge']}\n"
          "请只输出JSON：{\"score\":1-5整数(像不像真实的阿程且满足重点), \"ooc\":true/false(是否串人设/破功), "
          "\"ai_tone\":true/false(是否有AI/客服腔), \"offtopic\":true/false(是否没承接上文/跑题), \"reason\":\"一句话\"}")
    try:
        out = _judge_call(jp)
        m = re.search(r"\{.*\}", out or "", re.S)
        return json.loads(m.group(0)) if m else {"score": 0, "reason": "裁判无JSON:" + (out or "")[:80]}
    except Exception as e:
        return {"score": 0, "reason": f"裁判异常:{e}"}

# ══════════ 跑 ══════════
from collections import defaultdict
agg = defaultdict(lambda: {"auto_pass": 0, "auto_tot": 0, "score": [], "ooc": 0, "n": 0})
RESULTS = []
print(f"{'ID':<5}{'维度':<9}{'判分':<5} 关键检查 / 裁判理由")
print("-" * 92)
for case in C:
    for _ in range(args.n):
        bubbles = run_case(case)
        reply = " ||| ".join(bubbles) if bubbles else "(空)"
        hist_texts = [t for _, t in case.get("history", [])]
        ck = checks(bubbles, hist_texts)
        want = case.get("checks", ["ai_tone", "truncate", "leak"])
        fails = [f"{k}:{ck[k][1]}" for k in want if not ck[k][0]]
        j = judge(case, reply)
        a = agg[case["dim"]]
        a["auto_tot"] += len(want); a["auto_pass"] += sum(ck[k][0] for k in want)
        a["n"] += 1; a["ooc"] += 1 if j.get("ooc") else 0
        if isinstance(j.get("score"), int): a["score"].append(j["score"])
        flag = "OOC" if j.get("ooc") else ("AI腔" if j.get("ai_tone") else ("跑题" if j.get("offtopic") else ""))
        print(f"{case['id']:<5}{case['dim']:<9}{j.get('score','?')!s:<5} "
              f"{('硬检查未过['+','.join(fails)+'] ') if fails else ''}{flag} {j.get('reason','')[:44]}")
        print(f"      触发:{case.get('user','') or '(群里冒泡)'}  →  回复:{reply[:120]}")
        RESULTS.append({"id": case["id"], "dim": case["dim"], "ctx": case["ctx"],
                        "user": case.get("user", ""), "history": case.get("history", []),
                        "reply": reply, "fails": fails, "judge": j})

json.dump(RESULTS, open("entp_eval_result.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)

print("\n" + "=" * 60 + "\n维度汇总")
print(f"{'维度':<10}{'硬检查通过率':<14}{'裁判均分':<10}{'串人设次数'}")
for dim, a in agg.items():
    ap_ = f"{a['auto_pass']}/{a['auto_tot']}" if a['auto_tot'] else "-"
    sc = f"{sum(a['score'])/len(a['score']):.2f}" if a['score'] else "-"
    print(f"{dim:<10}{ap_:<14}{sc:<10}{a['ooc']}")
print("\n注：--mock 下裁判分恒为3(占位)，只验证harness；真实基线请在服务器上跑 `python entp_eval.py`。")
