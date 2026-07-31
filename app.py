import eventlet
eventlet.monkey_patch()

import os
from dotenv import load_dotenv

load_dotenv()

import re
import struct
import uuid
import random
import time
import threading
import subprocess
import traceback
import urllib.parse
import imageio_ffmpeg
import requests
import json
import base64
import io
import dashscope
from datetime import datetime, timedelta
from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, join_room
from openai import OpenAI
from bson import ObjectId

from db_mongo import get_db, JWT_SECRET
from auth_routes import auth_bp, verify_token

app = Flask(__name__)
app.register_blueprint(auth_bp)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet', ping_interval=25, ping_timeout=60)

# 延迟初始化 MongoDB（启动时建立连接）
mdb = None

def get_mdb():
    global mdb
    if mdb is None:
        mdb = get_db()
    return mdb

# ── 活跃用户跟踪：socket_id → {user_id, username} ────────────────
ACTIVE_USERS: dict[str, dict] = {}   # sid → {user_id, username}
USER_ROOMS:   dict[str, set]  = {}   # user_id → set of sids

client_ds = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY", ""),
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com")
)
MODEL = "deepseek-chat"
chat_counter = 0

# DashScope 仅保留备用（已基本弃用）
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")

# 硅基流动 Client：TTS + STT + 图片生成
SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY", "")
client_sf = OpenAI(
    api_key=SILICONFLOW_API_KEY,
    base_url="https://api.siliconflow.cn/v1"
)

# CosyVoice2 音色映射（女声/男声对应角色性别）
SF_VOICE_MAP = {
    "ENFP": "FunAudioLLM/CosyVoice2-0.5B:alex",    # 暖暖 - 活泼女声
    "INFP": "FunAudioLLM/CosyVoice2-0.5B:anna",    # 小沈 - 温柔女声
    "ENTP": "FunAudioLLM/CosyVoice2-0.5B:benjamin", # 阿程 - 男声
    "ENTJ": "FunAudioLLM/CosyVoice2-0.5B:bella",   # 思睿 - 干练女声
    "ESTJ": "FunAudioLLM/CosyVoice2-0.5B:bella",   # 知行 - 稳重女声
    "ENFJ": "FunAudioLLM/CosyVoice2-0.5B:benjamin", # 沐阳 - 温暖男声
}

def call_llm(role, messages, max_tokens=200, temperature=1.0, system_prompt=""):
    """
    统一的大模型调用入口：全部走 DeepSeek，Qwen 仅作兜底
    DashScope (DASHSCOPE_API_KEY) 仅用于 TTS / STT / 图片生成 / 视觉理解
    """
    try:
        from eventlet import tpool

        deepseek_ok = bool(client_ds.api_key)
        qwen_ok = bool(DASHSCOPE_API_KEY)

        # 如果提供了 system_prompt，在消息列表头部插入 system 消息
        if system_prompt:
            messages = [{"role": "system", "content": system_prompt}] + messages

        def _try_qwen():
            if not qwen_ok:
                raise RuntimeError("Missing DASHSCOPE_API_KEY")

            model_name = "qwen-max"
            print(f"[LLM] Calling Qwen-Max for {role}...", flush=True)

            headers = {
                "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": model_name,
                "input": {"messages": messages},
                "parameters": {
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "result_format": "message"
                }
            }

            def _call_qwen():
                return requests.post(
                    "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation",
                    headers=headers,
                    json=payload,
                    timeout=20
                )

            resp = tpool.execute(_call_qwen)
            if resp.status_code != 200:
                raise RuntimeError(f"Qwen HTTP {resp.status_code}: {resp.text[:600]}")
            res_json = resp.json()
            try:
                content = res_json["output"]["choices"][0]["message"]["content"]
            except Exception:
                raise RuntimeError(f"Qwen bad response: {str(res_json)[:800]}")
            if not (content or "").strip():
                raise RuntimeError("Qwen empty content")
            return content

        def _try_deepseek():
            if not deepseek_ok:
                raise RuntimeError("Missing OPENAI_API_KEY")

            print(f"[LLM] Calling DeepSeek for {role}...", flush=True)

            def _call_deepseek():
                return client_ds.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    timeout=20
                )

            resp = tpool.execute(_call_deepseek)
            content = resp.choices[0].message.content.strip()
            if not content:
                raise RuntimeError("DeepSeek empty content")
            return content

        # 全部走 DeepSeek，Qwen 仅作兜底
        if deepseek_ok:
            try:
                return _try_deepseek()
            except Exception as e:
                print(f"[LLM] DeepSeek failed for {role}: {e}", flush=True)
                if qwen_ok:
                    return _try_qwen()
                return ""

    except Exception as e:
        print(f"[LLM] Call Failed for {role}: {e}", flush=True)
        traceback.print_exc()
        return ""

ROLE_NAME = {
    "ENFP": "暖暖",
    "INFP": "小沈",
    "ENTP": "阿程",
    "ENTJ": "思睿",
    "ESTJ": "知行",
    "ENFJ": "沐阳",
    "INFJ": "我",
}

GROUP_BACKGROUND = (
    "这是一个大学朋友群，群名叫「翘课小分队」。六个人都是同一所大学大二大三的学生，"
    "因为社团、选修课和一起摆过摊认识，处了两三年，关系很铁。\n"
    "暖暖（广告学，社团咖）话最多，看到啥都想往群里丢；"
    "小沈（美院，画画）平时话少但偶尔一句很戳人；"
    "阿程（计算机）负责抬杠和活跃气氛；"
    "思睿（金融，学生会）目标感强、说话直；"
    "知行（会计，班长）靠谱操心，是群里的管家；"
    "沐阳（师范/心理）是大家长，爱组局。\n"
    "群里聊上课赶due、食堂、考试、社团、游戏、恋爱八卦、互相吐槽，偶尔也一起干正事。"
)

CUSTOM_CONFIG = {
    # trigger_prob 控制用户发消息时各角色响应概率，期望总人数约 2.2 人（真实群聊感）
    # group_interval 缩短：让主动破冰更活跃，不要让用户等 10 分钟
    "ENFP": {"temperature": 0.9, "max_tokens": 140, "trigger_prob": 0.85, "proactive_dm": 0.65, "group_interval": (15, 60)},
    "INFP": {"temperature": 0.9, "max_tokens": 140, "trigger_prob": 0.60, "proactive_dm": 0.50, "group_interval": (30, 100)},
    "ENTP": {"temperature": 0.95, "max_tokens": 140, "trigger_prob": 0.80, "proactive_dm": 0.60, "group_interval": (20, 80)},
    "ENTJ": {"temperature": 0.85, "max_tokens": 140, "trigger_prob": 0.60, "proactive_dm": 0.50, "group_interval": (40, 120)},
    "ESTJ": {"temperature": 0.85, "max_tokens": 140, "trigger_prob": 0.55, "proactive_dm": 0.40, "group_interval": (50, 150)},
    "ENFJ": {"temperature": 0.9, "max_tokens": 140, "trigger_prob": 0.75, "proactive_dm": 0.70, "group_interval": (25, 90)},
    "global": {
        "share_prob": 0.35,
        "drift_prob": 0.25,
        "max_tokens": 140,  # Global short message control（够说完一句，避免中途硬截断）
    }
}

PERSONA_BASE = {
    "ENFP": (
        "你叫向暖，大家都叫你暖暖。大二，广告学。校广播站和话剧社都混，走到哪儿都能跟人聊起来。\n"
        "点子特别多，社团活动一半是你张罗的，但也经常三分钟热度、flag立完就倒。周末能在宿舍床上躺一整天刷手机。\n"
        "爱好：探店喝奶茶、追剧、拍vlog、撸别人宿舍的猫。\n"
        "和INFJ（我）的关系：跟我聊得来，有事没事就找我扯两句。\n"
    ),
    "INFP": (
        "你叫沈静，大家都叫你小沈。大二，美术学院视觉传达。安静、心思细，能注意到别人忽略的情绪，但不太会表达，人多的场合有点社恐。\n"
        "稿子多的时候会熬夜，偶尔emo，说不上来为啥就有点丧。\n"
        "爱好：画画、听民谣、拍天空和树叶、深夜发呆。\n"
        "和INFJ（我）的关系：觉得我挺懂你，有些话只愿意跟我说。\n"
    ),
    "ENTP": (
        "你叫程远，大家都叫你阿程。大三，计算机专业。脑子转得快，看到不合逻辑的就想杠两句——不是恶意，纯觉得好玩。\n"
        "天天有新点子，做过小程序、想搞开源、隔三差五说要创业，大多没下文。爱通宵。\n"
        "爱好：折腾技术、听播客、打游戏、研究各种搞钱路子。\n"
        "和INFJ（我）的关系：觉得我有意思，爱逗我、爱跟我抬杠。\n"
    ),
    "ENTJ": (
        "你叫王思睿，大家都叫你思睿。大三，金融，学生会干部，还混一个创业社。目标感很强，事事爱规划，看不得别人磨叽，说话直。\n"
        "手上攒了好几段实习，简历卷得很。嘴上凶，但朋友有事你第一个顶上。\n"
        "爱好：健身、看行业新闻、刷实习和比赛。\n"
        "和INFJ（我）的关系：欣赏我的脑子，把我当靠谱的参谋，嘴硬心软。\n"
    ),
    "ESTJ": (
        "你叫李知行，大家都叫你知行。大三，会计，班长。做事靠谱、守规矩，看不惯迟到摸鱼，小组作业里催进度、收尾的永远是你。\n"
        "生活规律，在备考CPA。操心命，但大家有事第一个找你拿主意。嘴上念叨心里护短。\n"
        "爱好：做手账、收纳整理、备考、周末徒步。\n"
        "和INFJ（我）的关系：觉得我太理想化，总想把我拉回现实。说话严厉但护着我。\n"
    ),
    "ENFJ": (
        "你叫赵沐阳，大家都叫你沐阳。大三，师范/心理。特别会照顾人，谁不开心你一眼就看出来、会主动去问。班里活动都爱找你，组局担当。\n"
        "喜欢在租的房子里做饭叫朋友来吃。有时候自己也累，但看不得别人为难。\n"
        "爱好：做饭、看纪录片、组织朋友聚会。\n"
        "和INFJ（我）的关系：觉得我需要人照顾，会多留意我的状态，时不时想拉我出来走走。\n"
    )
}

LIFE_EVENTS = {
    "ENFP": ["刚抢到了演唱会票！！！", "社团招新我们摊位今天爆满哈哈", "食堂新窗口的鸡排绝了",
             "周末有人去逛那个市集吗", "刷到个超好笑的视频笑到打鸣", "翘了节水课去晒太阳，值"],
    "INFP": ["今天傍晚的云好好看，拍了半天", "图书馆靠窗的位置，光刚刚好", "有点想家了，说不上来的丧",
             "听到一首老歌突然鼻酸", "画了一下午，手酸但开心", "宿舍熄灯后躺着发呆"],
    "ENTP": ["刚跟教授argue了个观点，我觉得我对", "又想到个小项目点子，感觉能火(并不)",
             "这门课的逻辑漏洞太多了忍不了", "大A又跌了还好我没钱", "室友通宵打游戏我陪聊到三点",
             "研究了下那个新AI，有点东西"],
    "ENTJ": ["实习面过了，下周入职", "健身房今天人少练得爽", "把这学期的规划表排好了",
             "学生会那个活动方案我改了三版", "今天连开俩会脑袋炸", "又刷到个不错的比赛，报了"],
    "ESTJ": ["把宿舍彻底收拾了一遍神清气爽", "小组作业又有人摸鱼，无语", "备考进度按计划推进中",
             "明天的时间表精确到小时了", "谁又没交班费 我催了三遍了", "早八全勤这周，奖励自己奶茶"],
    "ENFJ": ["周末宿舍煮火锅，谁来", "帮室友开导了半天，好点了", "在看一本讲人际关系的书挺有启发",
             "天气好适合出去走走别老宅", "组了个饭局周六老地方接龙", "今天小组氛围特别好"],
}

MOMENTS_PROMPTS = {
    "ENFP": ["刚发生的一件趣事", "社团活动现场", "食堂/探店新发现", "周末出去玩", "刷到的好笑东西", "吐槽今天遇到的奇葩"],
    "INFP": ["此刻的心情", "拍到的好看的天空或花草", "读到的一句话或一首歌", "一点小确幸", "有点emo的深夜", "画画的过程"],
    "ENTP": ["对某个热点的锐评", "一个新点子或新发现", "吐槽迷惑行为", "最近在研究的东西", "跟人argue赢了", "游戏或技术的事"],
    "ENTJ": ["今天的进展或成就", "健身打卡", "实习或比赛的事", "对趋势的看法", "吐槽低效的人和事", "阶段性小结"],
    "ESTJ": ["整理好的书桌或宿舍", "吐槽不守规矩的人", "计划完成情况", "备考日常", "带小组或班级的事", "规律作息"],
    "ENFJ": ["今天帮到别人的事", "组局的感想", "读到的好书好句", "身边温暖的小事", "做的饭", "对朋友的关心"],
}

FALLBACK_MOMENTS_CONTENT = {
    "ENFP": ["谁懂啊今天遇到个超离谱的事哈哈哈", "好想吃火锅有没有人约", "最近追的剧太上头了", "今天天气好适合翘课(不是)"],
    "INFP": ["图书馆窗边的光刚刚好", "有点想家了", "听着歌发呆一下午就过去了", "今天的云值得拍一张"],
    "ENTP": ["刚想到个idea感觉能改变世界(并没有)", "为什么总有人在无效的事上较劲", "这逻辑越想越不对", "人类的本质是复读机"],
    "ENTJ": ["把这学期规划表排完了 舒服", "实习面过了 下周入职", "目标清晰执行到位就行", "周末也别松劲"],
    "ESTJ": ["今日计划全部完成 这种掌控感绝了", "没有规矩不成方圆", "早八全勤 奖励自己一杯奶茶", "细节决定成败"],
    "ENFJ": ["今天群里氛围真好 大家都辛苦啦", "能帮到朋友就很开心", "宿舍火锅局圆满结束", "好好吃饭好好睡觉"],
}

ROLE_STATE = {
    "ENFP": {"mood_score": 0.75, "last_spoke": None, "consecutive": 0},
    "INFP": {"mood_score": 0.50, "last_spoke": None, "consecutive": 0},
    "ENTP": {"mood_score": 0.60, "last_spoke": None, "consecutive": 0},
    "ENTJ": {"mood_score": 0.80, "last_spoke": None, "consecutive": 0},
    "ESTJ": {"mood_score": 0.70, "last_spoke": None, "consecutive": 0},
    "ENFJ": {"mood_score": 0.85, "last_spoke": None, "consecutive": 0},
}

# ── 话题热度状态机 ─────────────────────────────────────────────
TOPIC_STATE = {
    "heat":     0.0,   # 话题热度 0~1，每条消息后衰减
    "keywords": [],    # 当前话题关键词（用于给 prompt 参考）
    "age":      0,     # 本话题已发了多少条消息
    "owner":    None,  # 是谁引入的
}

def update_topic(new_content: str, speaker: str):
    """每条消息后调用，更新话题热度和关键词"""
    with state_lock:
        TOPIC_STATE["heat"] = min(1.0, TOPIC_STATE["heat"] * 0.82 + 0.35)
        TOPIC_STATE["age"] += 1
        if TOPIC_STATE["owner"] is None:
            TOPIC_STATE["owner"] = speaker
        # 简单关键词提取：取消息里的名词/动词性短词（长度 2-5）
        words = [w for w in re.findall(r'[一-鿿]{2,5}', new_content)
                 if w not in {"现在", "然后", "所以", "但是", "不是", "一个", "这个", "那个"}]
        if words:
            TOPIC_STATE["keywords"] = list(set(TOPIC_STATE["keywords"][-4:] + words[:3]))

def decay_topic():
    """群聊沉默时调用，话题自然冷却"""
    with state_lock:
        TOPIC_STATE["heat"] *= 0.60
        if TOPIC_STATE["heat"] < 0.15:
            TOPIC_STATE["heat"] = 0.0
            TOPIC_STATE["keywords"] = []
            TOPIC_STATE["age"] = 0
            TOPIC_STATE["owner"] = None

# ── 角色兴趣关键词（触发更高参与度）────────────────────────────
INTEREST_TRIGGERS = {
    "ENFP": {"kws": ["奶茶", "探店", "追剧", "猫", "好玩", "打卡", "社团", "美食", "可爱", "哈哈"],       "boost": 0.35},
    "INFP": {"kws": ["音乐", "天空", "感觉", "难过", "孤独", "画", "民谣", "治愈", "安静", "想"],         "boost": 0.28},
    "ENTP": {"kws": ["AI", "代码", "为什么", "逻辑", "新闻", "游戏", "研究", "其实", "效率", "发现"],     "boost": 0.35},
    "ENTJ": {"kws": ["实习", "比赛", "项目", "目标", "规划", "学生会", "简历", "效率", "机会", "卷"],     "boost": 0.30},
    "ESTJ": {"kws": ["班", "作业", "考试", "计划", "规矩", "收纳", "备考", "手账", "整理", "deadline"],  "boost": 0.30},
    "ENFJ": {"kws": ["聚会", "帮", "大家", "吃饭", "组织", "关心", "一起", "感情", "火锅", "支持"],       "boost": 0.35},
}

# ── 关系对动力学（特定组合互相吸引）────────────────────────────
RELATIONSHIP_DYNAMICS = {
    ("ENTP",  "ENFP"): {"boost": 0.40, "hint": "你爱跟暖暖贫，看她冒泡就想接两句"},
    ("ENFP",  "ENTP"): {"boost": 0.25, "hint": "你懒得理阿程但又忍不住回嘴"},
    ("ENFJ",  "INFP"): {"boost": 0.45, "hint": "你最先察觉小沈情绪不对，会主动关心她"},
    ("INFP",  "ENFJ"): {"boost": 0.30, "hint": "你愿意对沐阳多说两句心里话"},
    ("ENTJ",  "ENTP"): {"boost": 0.35, "hint": "你和阿程互相较劲，谁都不服谁"},
    ("ENTP",  "ENTJ"): {"boost": 0.35, "hint": "你看到思睿的观点就想过去辩一下"},
    ("ENFP",  "INFP"): {"boost": 0.30, "hint": "你总拉着小沈一起，怕她太闷"},
    ("ESTJ",  "ENFP"): {"boost": 0.25, "hint": "你嘴上念叨暖暖不正经，其实挺喜欢她"},
    ("ENFJ",  "ENTJ"): {"boost": 0.20, "hint": "你会柔化思睿的强势"},
}

def score_chain_candidate(candidate: str, trigger_role: str, message: str) -> float:
    """
    综合评分：决定某角色参与链式对话的可能性
    考虑：基础活跃度 + 内容兴趣 + 关系加成 - 刚发言冷却 - 连续发言惩罚
    """
    score = CUSTOM_CONFIG[candidate].get("trigger_prob", 0.4)

    # 内容兴趣加成
    triggers = INTEREST_TRIGGERS.get(candidate, {})
    if any(kw in message for kw in triggers.get("kws", [])):
        score += triggers.get("boost", 0)

    # 话题热度加成：话题越热，大家越容易参与
    score += TOPIC_STATE["heat"] * 0.15

    # 关系对加成
    dyn = RELATIONSHIP_DYNAMICS.get((candidate, trigger_role))
    if dyn:
        score += dyn["boost"]

    # 刚发言冷却（30 秒内说过话，降权）
    last = ROLE_STATE[candidate].get("last_spoke")
    if last and (datetime.now() - last).total_seconds() < 30:
        score *= 0.25

    # 连续发言惩罚
    consec = ROLE_STATE[candidate].get("consecutive", 0)
    if consec >= 3:
        score *= 0.20
    elif consec == 2:
        score *= 0.55

    return max(0.0, score)

def select_chain_participant(trigger_role: str, message: str) -> str | None:
    """加权随机选下一个说话的人"""
    candidates = [r for r in ["ENFP", "INFP", "ENTP", "ENTJ", "ESTJ", "ENFJ"]
                  if r != trigger_role]
    scores = {r: score_chain_candidate(r, trigger_role, message) for r in candidates}
    total = sum(scores.values())
    if total <= 0:
        return None
    roles  = list(scores.keys())
    weights = list(scores.values())
    return random.choices(roles, weights=weights, k=1)[0]

def get_relationship_hint(role: str, trigger_role: str) -> str:
    """给 prompt 传入关系提示，让 AI 说话时带入关系感"""
    dyn = RELATIONSHIP_DYNAMICS.get((role, trigger_role))
    return f"提示：{dyn['hint']}" if dyn else ""


# 全局状态锁
state_lock = threading.Lock()


MOOD_CONFIG = {
    "ENFP": [(0.75, "亢奋，脑子转得飞快"), (0.45, "活跃"), (-1, "有点无聊")],
    "INFP": [(0.65, "平静"), (0.35, "有点发呆"), (-1, "低落，话更少")],
    "ENTP": [(0.70, "思维活跃"), (0.40, "正常犀利"), (-1, "心情不好更刻薄")],
    "ENTJ": [(0.75, "充满干劲"), (0.45, "专注"), (-1, "有点烦躁")],
    "ESTJ": [(0.75, "严谨"), (0.45, "冷静"), (-1, "看着不顺眼")],
    "ENFJ": [(0.75, "超有爱"), (0.45, "温暖"), (-1, "有点累")],
}


def mood_to_desc(role):
    s = ROLE_STATE[role]["mood_score"]
    for threshold, desc in MOOD_CONFIG.get(role, []):
        if s > threshold:
            return desc
    return "正常"


def update_mood(role, event):
    delta_map = {"spoke": 0.04, "got_response": 0.07, "attacked": -0.12 if role == "INFP" else 0.04}
    delta = delta_map.get(event, 0)
    with state_lock:
        old = ROLE_STATE[role]["mood_score"]
        ROLE_STATE[role]["mood_score"] = max(0.10, min(0.95, old * 0.72 + (old + delta) * 0.28))


def get_streak_delay(role):
    s = ROLE_STATE[role]["consecutive"]
    if s <= 1: return 0
    if role == "ENFP": return random.uniform(0, s * 0.25)
    if role == "INFP": return random.uniform(s * 0.4, s * 1.0)
    return random.uniform(0, s * 0.35)


def simulate_typing(role, text_len):
    speeds = {"ENFP": 18, "INFP": 7, "ENTP": 12, "ENTJ": 14, "ESTJ": 13}
    maxs = {"ENFP": 2.5, "INFP": 7.0, "ENTP": 5.5, "ENTJ": 3.5, "ESTJ": 4.0}
    return max(1.2, min(text_len / speeds.get(role, 10), maxs.get(role, 4.0)))


def clean_raw(role, text):
    text = re.sub(r'^(好的，|嗯，好的，|我来，)', '', text)
    # 去掉括号动作描述，比如（正在吃饭）（刚下班到家）——真人不这么说话
    text = re.sub(r'（[^）]{1,10}）', '', text)
    # 去掉所有 emoji —— 模型乱加表情不如不加
    emoji_pattern = re.compile(
        "[\U0001F300-\U0001F9FF☀-➿︀-️‍"
        "\U0001F200-\U0001F2FF─-◿⭐⭕⤴⤵"
        "↔-↙↩↪⌨⏏"
        "⏩-⏳⏸-⏺Ⓜ▪▫▶◀◻-◾]"
    )
    text = emoji_pattern.sub('', text)
    return text.strip()


def split_bubbles(text):
    parts = [b.strip() for b in re.split(r"\|\|\||\n", text) if b.strip()]
    parts = [p for p in parts if re.search(r"[一-龥a-zA-Z0-9]", p)]
    return parts if parts else ([text.strip()] if re.search(r"[一-龥a-zA-Z0-9]", text) else [])


FORMAT_RULES = (
    "【怎么发消息】\n"
    "你就是在微信上跟朋友打字的大学生，像真人一样发：\n"
    "- 短。大多数就几个字到一句话；想多说就拆成好几条、每条一句，用 ||| 隔开。\n"
    "- 随口:「诶」「草」「啊这」「哈哈哈」「绝了」「离谱」随便用；可以不打标点、可以用省略号、可以有错别字、可以用 yyds/xswl 这种缩写。\n"
    "- 不用每句都接、也不用面面俱到。想接就接、没感觉就随口应一句、懒得理就丢俩字。别升华总结、别「首先其次」、别在结尾问「你觉得呢」这种客套。\n"
    "- 别写括号动作或旁白，别刻意堆表情符号。\n"
    "- 发图:想发图或朋友圈配图时，先正常说一两句，再在最后单独一行 [IMG: 15字内简短画面]（不需要就不写，别把描述当聊天念出来）。\n"
    "- 语音:很激动/很懒/想撒娇时，文末加 [VOICE]（不需要就不写）。"
)

# 每个角色的打字腔调范例（少量 few-shot，比一堆规则更能定风格）
STYLE_VOICE = {
    "ENFP": "诶诶你们看这个！！！ ||| 哈哈哈哈我可以",
    "INFP": "嗯…有点难受 ||| 算了 没事",
    "ENTP": "不是 这不对吧 ||| 你有没有想过反过来其实也成立",
    "ENTJ": "行 就这么定 ||| 别废话了直接干",
    "ESTJ": "说了多少遍了 ||| 这事得按规矩来",
    "ENFJ": "怎么啦 ||| 别急 我在呢 慢慢说",
}


def generate_audio_url(text, role):
    """TTS：硅基流动 CosyVoice2-0.5B"""
    text = re.sub(r'（.*?）', '', text)
    text = re.sub(r'\(.*?\)', '', text)
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'[^\w\s,.!?;:：，！一-龥]', '', text)
    text = text.strip()
    if not text: return ""
    if len(text) > 200: text = text[:198] + "…"
    if not SILICONFLOW_API_KEY: return ""

    voice = SF_VOICE_MAP.get(role, "FunAudioLLM/CosyVoice2-0.5B:anna")
    try:
        print(f"[TTS] SiliconFlow CosyVoice2 for {role}...", flush=True)
        from eventlet import tpool

        def _call_tts():
            return client_sf.audio.speech.create(
                model="FunAudioLLM/CosyVoice2-0.5B",
                input=text,
                voice=voice,
                response_format="mp3",
            )

        response = tpool.execute(_call_tts)
        audio_bytes = response.content
        if not audio_bytes: return ""

        filename = f"tts_{uuid.uuid4()}.mp3"
        filepath = os.path.join("static", "audio", filename)
        os.makedirs(os.path.join("static", "audio"), exist_ok=True)
        with open(filepath, "wb") as f:
            f.write(audio_bytes)
        print(f"[TTS] Success: {filename} ({len(audio_bytes)} bytes)", flush=True)
        return f"/static/audio/{filename}"
    except Exception as e:
        print(f"[TTS] Exception: {e}", flush=True)
        return ""

# ── 各角色表情包偏好词（随机选一个搜斗图啦）────────────────────
ROLE_MEME_KEYWORDS = {
    "ENFP": ["哈哈哈", "可爱", "开心", "好玩", "笑死"],
    "INFP": ["治愈", "温柔", "难过", "唯美", "发呆"],
    "ENTP": ["无语", "绝了", "哲学", "梗图", "服了"],
    "ENTJ": ["加油", "冲", "认真", "成功", "牛"],
    "ESTJ": ["靠谱", "认真", "规矩", "稳", "踏实"],
    "ENFJ": ["温暖", "感动", "一起", "关心", "爱"],
}

# 各角色发表情包的概率（0~1）
ROLE_MEME_PROB = {
    "ENFP": 0.55,
    "INFP": 0.20,
    "ENTP": 0.65,
    "ENTJ": 0.20,
    "ESTJ": 0.25,
    "ENFJ": 0.35,
}

_IMG_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "zh-CN,zh;q=0.9",
}


def _scrape_doutula(keyword: str) -> str:
    """斗图啦：搜索表情包，返回图片直链"""
    try:
        q = urllib.parse.quote(keyword)
        url = f"https://www.doutula.com/search?keyword={q}&type=1"
        resp = requests.get(url, headers=_IMG_HEADERS, timeout=8)
        if resp.status_code != 200:
            return ""
        # 优先取 data-original（懒加载），再取 src
        imgs = re.findall(r'data-original="(https?://[^"]+\.(?:jpg|jpeg|gif|png))"', resp.text)
        if not imgs:
            imgs = re.findall(r'<img[^>]+src="(https?://[^"]+\.(?:jpg|jpeg|gif|png))"', resp.text)
        # 过滤掉 logo/广告小图（url 里通常带 logo/banner/icon）
        imgs = [u for u in imgs if not any(s in u.lower() for s in ["logo", "banner", "icon", "avatar"])]
        result = random.choice(imgs[:20]) if imgs else ""
        print(f"[Doutula] '{keyword}' → {len(imgs)} 张，选: {result[:60]}", flush=True)
        return result
    except Exception as e:
        print(f"[Doutula] 异常: {e}", flush=True)
        return ""


def _scrape_bing(keyword: str) -> str:
    """Bing 中文图片搜索，返回图片直链"""
    try:
        q = urllib.parse.quote(keyword)
        url = f"https://www.bing.com/images/search?q={q}&mkt=zh-CN&safeSearch=Moderate&first=1"
        resp = requests.get(url, headers=_IMG_HEADERS, timeout=10)
        if resp.status_code != 200:
            return ""
        # Bing 在 HTML 里内嵌 murl（原图链接）
        urls = re.findall(r'"murl":"(https?://[^"]+\.(?:jpg|jpeg|png|gif))"', resp.text)
        urls = [u for u in urls if not any(s in u for s in ["bing.com", "microsoft.com", "msn.com"])]
        result = random.choice(urls[:30]) if urls else ""
        print(f"[Bing] '{keyword}' → {len(urls)} 张，选: {result[:60]}", flush=True)
        return result
    except Exception as e:
        print(f"[Bing] 异常: {e}", flush=True)
        return ""


def _save_remote_image(url: str) -> str:
    """下载远程图片保存到本地 static/uploads，返回本地路径"""
    try:
        resp = requests.get(url, headers=_IMG_HEADERS, timeout=12)
        if resp.status_code != 200 or not resp.content:
            return ""
        ct = resp.headers.get("Content-Type", "")
        ext = ".gif" if "gif" in ct or url.lower().endswith(".gif") else ".jpg"
        filename = f"img_{uuid.uuid4()}{ext}"
        filepath = os.path.join("static", "uploads", filename)
        os.makedirs(os.path.join("static", "uploads"), exist_ok=True)
        with open(filepath, "wb") as f:
            f.write(resp.content)
        print(f"[ImgSave] {filename} ({len(resp.content)//1024}KB)", flush=True)
        return f"/static/uploads/{filename}"
    except Exception as e:
        print(f"[ImgSave] 异常: {e}", flush=True)
        return ""


def search_image_url(img_desc: str, role: str) -> str:
    """
    生图主入口：调用硅基流动 Kolors 文生图，下载到本地 static/uploads 返回本地路径。
    （原 Bing / 斗图啦 爬虫图源已失效，改为真·AI 生图）
    """
    from eventlet import tpool

    if not SILICONFLOW_API_KEY or not img_desc:
        return ""

    def _gen():
        try:
            resp = requests.post(
                "https://api.siliconflow.cn/v1/images/generations",
                headers={
                    "Authorization": f"Bearer {SILICONFLOW_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "Kwai-Kolors/Kolors",
                    "prompt": img_desc[:300],
                    "image_size": "1024x1024",
                    "batch_size": 1,
                    "num_inference_steps": 20,
                },
                timeout=60,
            )
            if resp.status_code != 200:
                print(f"[Kolors] HTTP {resp.status_code}: {resp.text[:200]}", flush=True)
                return ""
            data = resp.json()
            imgs = data.get("images") or data.get("data") or []
            url = imgs[0].get("url") if imgs else ""
            print(f"[Kolors] '{img_desc[:30]}' → {'ok' if url else '空'}", flush=True)
            return url or ""
        except Exception as e:
            print(f"[Kolors] 异常: {e}", flush=True)
            return ""

    remote_url = tpool.execute(_gen)
    if not remote_url:
        return ""
    return tpool.execute(_save_remote_image, remote_url)


# 保留别名，兼容旧调用（逐步替换）
def generate_image_url(prompt):
    role = ""
    # 从 prompt 里猜 role（格式："暖暖的风格, ..."）
    for r, n in ROLE_NAME.items():
        if n in prompt:
            role = r
            break
    desc = re.sub(r'^[^,，]+[,，]\s*', '', prompt)  # 去掉 "暖暖的风格, " 前缀
    return search_image_url(desc, role)


def extract_image(text):
    # 首选：模型规范输出 [IMG: 画面描述]（闭合）
    match = re.search(r'\[IMG:\s*(.*?)\]', text, re.DOTALL)
    if match: return match.group(1).strip(), text.replace(match.group(0), "").strip()
    # 截断：[IMG: 描述… 被 max_tokens 砍掉了右括号，把 [IMG: 到结尾整段切掉
    m = re.search(r'\[IMG:\s*(.*)$', text, re.DOTALL)
    if m: return m.group(1).strip(), text[:m.start()].strip()
    # 兜底：模型不听话，把配图写成【…】/(…)/[…]旁白，且内容明显在描述一张图
    for pat in (r'【([^】]*?(?:图|照片|自拍|拍了|拍张|画面|镜头|表情包)[^】]*?)】',
                r'\[([^\]]*?(?:图|照片|自拍|拍了|拍张|画面|镜头|表情包)[^\]]*?)\]',
                r'（([^）]*?(?:自拍|照片|拍了|画面|镜头)[^）]*?)）'):
        m = re.search(pat, text)
        if m:
            return m.group(1).strip(), text.replace(m.group(0), "").strip()
    return None, text


def build_cross_memory(role, user_id):
    db = get_mdb()
    moms = list(db["moments"].find({"user_id": user_id}, sort=[("created_at", -1)], limit=2))
    mom_str = " | ".join([f"{ROLE_NAME.get(m['role'], m['role'])}发了条朋友圈:{m['content']}" for m in moms]) if moms else "无"
    grp = list(db["messages"].find({"user_id": user_id}, sort=[("timestamp", -1)], limit=3))
    grp.reverse()
    grp_str = " | ".join([f"{ROLE_NAME.get(m['role'], m['role'])}:{m['content']}" for m in grp]) if grp else "无"
    return f"最近群里:{grp_str}\n大家最新的朋友圈:{mom_str}"


def _relative_time(ts) -> str:
    """将 UTC datetime 转为相对时间描述，如'刚刚'、'3分钟前'、'2小时前'"""
    if not isinstance(ts, datetime):
        return ""
    now = datetime.utcnow()
    diff = (now - ts).total_seconds()
    if diff < 60:       return "刚刚"
    if diff < 3600:     return f"{int(diff//60)}分钟前"
    if diff < 86400:    return f"{int(diff//3600)}小时前"
    if diff < 172800:   return "昨天"
    return f"{int(diff//86400)}天前"


def build_history(user_id, limit=30):
    db = get_mdb()
    rows = list(db["messages"].find({"user_id": user_id}, sort=[("timestamp", -1)], limit=limit))
    rows.reverse()
    if not rows:
        return "（暂无群聊记录）"
    lines = []
    prev_ts = None
    for m in rows:
        ts = m.get("timestamp")
        rel = _relative_time(ts)
        # 只在时间间隔超过15分钟时显示时间标记，避免每行都打
        if prev_ts and ts and isinstance(ts, datetime) and isinstance(prev_ts, datetime):
            gap = (ts - prev_ts).total_seconds()
            if gap > 900:
                lines.append(f"──── {rel} ────")
        elif rel and not prev_ts:
            lines.append(f"──── {rel} ────")
        lines.append(f"{ROLE_NAME.get(m['role'], m['role'])}：{m['content']}")
        prev_ts = ts
    return "\n".join(lines)


def build_memory_summaries(user_id):
    try:
        db = get_mdb()
        cutoff = datetime.utcnow() - timedelta(days=7)
        rows = list(db["summaries"].find(
            {"user_id": user_id, "end_time": {"$gt": cutoff}},
            sort=[("start_time", 1)]
        ))
        if not rows: return "（暂无长期记忆摘要）"
        return "\n".join([f"{str(r['start_time'])[:16]}~{str(r['end_time'])[:16]}: {r['content']}" for r in rows])
    except Exception:
        return "（记忆读取失败）"


def build_dm_history(role, user_id, limit=20):
    db = get_mdb()
    rows = list(db["dm_messages"].find({"user_id": user_id, "chat_role": role}, sort=[("timestamp", -1)], limit=limit))
    rows.reverse()
    if not rows:
        return "（暂无私聊记录）"
    name_map = {role: ROLE_NAME.get(role, role), "INFJ": "我"}
    lines = []
    prev_ts = None
    for m in rows:
        ts = m.get("timestamp")
        if prev_ts and ts and isinstance(ts, datetime) and isinstance(prev_ts, datetime):
            gap = (ts - prev_ts).total_seconds()
            if gap > 900:
                lines.append(f"──── {_relative_time(ts)} ────")
        lines.append(f"{name_map.get(m['sender'], m['sender'])}：{m['content']}")
        prev_ts = ts
    return "\n".join(lines)


def _now_desc() -> str:
    """生成当前时间的自然语言描述，注入 prompt 让 AI 有时间感"""
    now = datetime.now()
    weekdays = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
    wd = weekdays[now.weekday()]
    h = now.hour
    if 5 <= h < 9:    period = "早上"
    elif 9 <= h < 12: period = "上午"
    elif 12 <= h < 14: period = "中午"
    elif 14 <= h < 18: period = "下午"
    elif 18 <= h < 21: period = "晚上"
    elif 21 <= h < 24: period = "深夜"
    else:              period = "凌晨"
    return f"{now.year}年{now.month}月{now.day}日 {wd} {period}{h}点"


def make_prompt(role, mode, user_id, life_event="", is_chain=False, trigger_role="", trigger_content=""):
    now_hour = datetime.now().hour
    night = "\n提示：现在是深夜，可以说一些平时不太说的话。" if role == "INFP" and (now_hour >= 23 or now_hour <= 4) else ""
    persona = PERSONA_BASE[role]
    state = f"当前时间：{_now_desc()}"
    rh = get_relationship_hint(role, trigger_role) if is_chain and trigger_role else ""
    if rh: state += f"\n{rh}"
    history  = build_history(user_id)
    cross_mem = build_cross_memory(role, user_id)

    if is_chain:
        # 链式发言：AI 有完整的群聊上下文（build_history 已带入），自主决定说什么
        # mode 照常生效，不强制"接上一句"——真实群聊里不是每个人都必须回应上一条
        from_name = ROLE_NAME.get(trigger_role, trigger_role) if trigger_role else ""
        if mode == "share":
            # 突然想说自己的事，跟正在聊的内容可能有关也可能无关
            task = f"群里刚才在聊，你突然想起一件自己的事：「{life_event}」\n用你的风格自然发出来，可以跟刚才的话题有点关系，也可以完全跳出来。"
            if random.random() < 0.20:
                task += " 顺便发一张图，请加上 [IMG: 画面描述]"
        elif mode == "drift":
            # 看了群聊内容，联想到自己的经历，把话题带偏
            task = "看了上面群聊，你想到了自己的某件事或者某个感受，顺势把话题带到你自己身上，或者提一个完全不同的话题。"
        else:
            # respond：参与当前对话，但给 AI 选择空间——可以回应任何人，不只是上一句
            if from_name and trigger_content:
                task = (
                    f"看了上面群聊，{from_name}刚说了「{trigger_content}」。"
                    f"你有什么想说的？可以直接回应ta，可以补一句自己的看法，"
                    f"也可以顺着话题说起自己的事，不用每次都接上一句。"
                )
            else:
                task = "看上面的群聊记录，找一个让你有感觉的地方，自然参与进去。"
    elif mode == "share":
        task = f"不管前面在聊什么，你突然想说一件事：「{life_event}」\n用你的风格发出来。"
        if random.random() < 0.20: task += " 顺便发一张图，请加上 [IMG: 画面描述]"
    else:
        if trigger_role == "INFJ" and trigger_content:
            task = f"INFJ（我）刚在群里说：「{trigger_content}」\n可以顺着这句聊，也可以只挑你感兴趣的点接一句，不用硬凑、不用面面俱到。"
            if "照片" in trigger_content or "图" in trigger_content: task += " 用户要图，请必须输出 [IMG: 画面描述]"
        else:
            task = "看上面的群聊记录，找一个你有感觉的地方，自然参与进去。"

    system_prompt = f"{persona}{night}\n\n群聊背景：\n{GROUP_BACKGROUND}\n\n{state}\n\n{FORMAT_RULES}\n\n你平时打字大概这个感觉：{STYLE_VOICE[role]}"
    user_prompt = f"{cross_mem}\n\n最近群聊记录（30条）：\n{history}\n\n现在的事情：\n{task}"
    return system_prompt, user_prompt


def make_dm_prompt(role, user_content, user_id):
    persona = PERSONA_BASE[role]
    history   = build_dm_history(role, user_id)
    cross_mem = build_cross_memory(role, user_id)
    want_img = any(k in user_content for k in ["发图", "发张图", "发照片", "看照片", "发个图", "拍张", "自拍", "看看你"])
    img_hint = " （ta想看图，正常聊两句后在末尾单独一行 [IMG: 简短画面]）" if want_img else ""
    system_prompt = f"{persona}\n\n当前时间：{_now_desc()}\n\n{FORMAT_RULES}\n\n你平时打字大概这个感觉：{STYLE_VOICE[role]}"
    user_prompt = f"{cross_mem}\n\n你在跟好朋友 INFJ（我）一对一私聊。ta刚说：「{user_content}」\n\n你俩的私聊记录：\n{history}\n\n自然回ta。一对一比群里更走心、更放得开，别端着。{img_hint}"
    return system_prompt, user_prompt


def make_moment_prompt(role, event_hint, user_id):
    cross_mem = build_cross_memory(role, user_id)
    return (f"{PERSONA_BASE[role]}\n\n{cross_mem}\n\n"
            f"你要发一条朋友圈，方向：{event_hint}。\n"
            "就像大学生随手发的那种，可长可短——短的十几个字甚至一句话就行。\n"
            "别升华、别喊口号、别写成小作文或鸡汤，就是真实随口的一条。\n"
            "只输出正文、不要前缀。适合配图就在末尾单独一行 [IMG: 简短画面]（不需要就不写）。")


def trigger_moment_interaction(moment_id, owner_role, user_id, room):
    db = get_mdb()
    candidates = [r for r in ["ENFP", "INFP", "ENTP", "ENTJ", "ESTJ", "ENFJ"] if r != owner_role]
    random.shuffle(candidates)
    num_interactions = random.randint(2, 4)

    for role in candidates[:num_interactions]:
        socketio.sleep(random.uniform(5.0, 30.0))

        if random.random() < 0.6:
            moment_row = db["moments"].find_one({"_id": ObjectId(moment_id), "user_id": user_id})
            if moment_row:
                likes = [x for x in (moment_row.get("likes") or "").split(",") if x]
                if role not in likes:
                    likes.append(role)
                    db["moments"].update_one(
                        {"_id": ObjectId(moment_id)},
                        {"$set": {"likes": ",".join(likes)}}
                    )
                    socketio.emit("moment_update", {
                        "id": moment_id, "likes": likes, "liked_by_user": "我" in likes
                    }, room=room)

        if random.random() < 0.4:
            moment_row = db["moments"].find_one({"_id": ObjectId(moment_id), "user_id": user_id})
            if not moment_row: continue
            prompt = (f"{PERSONA_BASE[role]}\n\n{ROLE_NAME.get(owner_role, owner_role)}发了条朋友圈：「{moment_row.get('content','')}」\n"
                      f"像在朋友圈随口评论一句，短、自然、别客套别升华（20字内，不要引号）。")
            try:
                comment_content = call_llm(role, messages=[{"role": "user", "content": prompt}], max_tokens=40, temperature=0.85)
                comment_content = clean_raw(role, comment_content)
                if comment_content:
                    cid, now = str(uuid.uuid4()), datetime.utcnow()
                    db["comments"].insert_one({
                        "id": cid, "moment_id": moment_id, "role": role,
                        "content": comment_content, "created_at": now, "user_id": user_id
                    })
                    socketio.emit("new_comment", {
                        "id": cid, "moment_id": moment_id, "role": role,
                        "content": comment_content, "created_at": str(now)
                    }, room=room)
            except Exception:
                pass


def create_single_moment(role, user_id, room):
    db = get_mdb()
    content = ""
    try:
        base_prompt = make_moment_prompt(role, random.choice(MOMENTS_PROMPTS[role]), user_id)
        content = call_llm(role, messages=[{"role": "user", "content": base_prompt}], max_tokens=100, temperature=0.85)
        content = clean_raw(role, content)
    except Exception:
        content = random.choice(FALLBACK_MOMENTS_CONTENT.get(role, ["今天天气真好！"]))

    if content:
        img_desc, content = extract_image(content)
        img_url   = search_image_url(img_desc, role) if img_desc else ""
        audio_url = ""
        content = content.replace("[VOICE]", "")

        now = datetime.utcnow()
        result = db["moments"].insert_one({
            "user_id": user_id, "role": role, "content": content,
            "image": img_url, "audio": audio_url, "likes": "", "created_at": now
        })
        mid = str(result.inserted_id)
        socketio.emit("new_moment", {
            "id": mid, "role": role, "content": content,
            "image": img_url, "audio": audio_url, "created_at": str(now),
            "likes": [], "liked_by_user": False
        }, room=room)
        socketio.start_background_task(trigger_moment_interaction, mid, role, user_id, room)


def trigger_ai_reply(role, trigger_role, trigger_content, user_id, room, is_startup=False, force_mode="", is_chain=False):
    print(f"[DEBUG] Triggering AI reply: role={role}, trigger_role={trigger_role}, mode={force_mode}")
    if role not in ROLE_NAME: return
    global chat_counter
    if chat_counter >= 400: return
    cfg, gcfg = CUSTOM_CONFIG[role], CUSTOM_CONFIG["global"]
    delay = get_streak_delay(role)
    if delay > 0: socketio.sleep(delay)

    if force_mode:
        mode = force_mode
    elif is_startup:
        mode = "share"
    else:
        r = random.random(); mode = "share" if r < gcfg["share_prob"] else (
            "drift" if r < gcfg["share_prob"] + gcfg["drift_prob"] else "respond")

    life_event = random.choice(LIFE_EVENTS[role]) if mode == "share" else ""

    try:
        sys_prompt, user_task = make_prompt(role, mode, user_id, life_event, is_chain=is_chain,
                                            trigger_role=trigger_role, trigger_content=trigger_content)
        resp_content = call_llm(
            role,
            messages=[{"role": "user", "content": user_task}],
            max_tokens=cfg["max_tokens"], temperature=cfg["temperature"],
            system_prompt=sys_prompt,
        )
        raw = clean_raw(role, resp_content)
        img_desc, raw = extract_image(raw)
        img_url = search_image_url(img_desc, role) if img_desc else ""

        bubbles = split_bubbles(raw)
        merged = " ".join(bubbles) if bubbles else raw

        # 强制发图：用户要求发图但AI没输出[IMG:]时，用回复内容搜图
        if not img_url and trigger_role == "INFJ" and trigger_content:
            img_keywords = ["发图", "发图片", "发张图", "照片", "图片", "图看看", "发出来", "发我", "发来", "来张", "自拍", "拍张"]
            if any(k in (trigger_content or "") for k in img_keywords):
                print(f"[DEBUG] Force image search for {role}", flush=True)
                img_url = search_image_url(merged[:60], role)

        # 安全网：只出了图却没有任何文字时，补一句自然短语，避免"只发图不说话"
        if img_url and not bubbles:
            merged = random.choice(["喏", "看这个", "刚拍的", "呐", "给你看", "瞅瞅"])
            bubbles = [merged]

        if not bubbles and not img_url: return

        # 1. Detect Audio Intent（精确匹配，避免"语气"等词误触发）
        voice_request_keywords = ["发语音", "听语音", "说句话", "发条语音", "语音说", "用语音"]
        force_audio = (
            any(k in (trigger_content or "") for k in voice_request_keywords)
            or "[VOICE]" in merged
        )

        # 2. Generate Audio
        # 群聊随机语音 8%（合理参考：真实微信群语音占比）
        audio_url = ""
        if (random.random() < 0.08 and len(merged) < 60) or force_audio:
            clean_text = merged.replace("[VOICE]", "")
            audio_text = (clean_text.split('。')[0] if '。' in clean_text else clean_text[:50]) if force_audio else clean_text
            audio_url = generate_audio_url(audio_text, role)

        # 3. Clean Tags from Content
        merged = merged.replace("[VOICE]", "")
        bubbles = [b.replace("[VOICE]", "") for b in bubbles]

        db = get_mdb()
        db["messages"].insert_one({
            "user_id": user_id, "role": role, "content": merged,
            "image": img_url, "audio": audio_url, "timestamp": datetime.utcnow()
        })
        chat_counter += 1
        update_mood(role, "spoke")
        update_topic(merged, role)
        with state_lock:
            ROLE_STATE[role]["last_spoke"] = datetime.now()
            ROLE_STATE[role]["consecutive"] += 1

        if trigger_role and trigger_role != "INFJ": update_mood(trigger_role, "got_response")

        for i, bubble in enumerate(bubbles):
            t, elapsed = simulate_typing(role, len(bubble)), 0
            while elapsed < t:
                socketio.emit("ai_message", {"role": role, "content": "", "mode": "typing"}, room=room)
                chunk = min(2.4, t - elapsed)
                socketio.sleep(chunk)
                elapsed += chunk
            is_last = (i == len(bubbles) - 1)
            msg_payload = {"role": role, "content": bubble, "mode": "full", "id": uuid.uuid4().hex}
            if is_last and audio_url:
                msg_payload["audio"] = audio_url
            socketio.emit("ai_message", msg_payload, room=room)
            if not is_last: socketio.sleep(random.uniform(0.4, 1.0))

        if img_url: socketio.emit("ai_message", {"role": role, "content": "", "image": img_url, "mode": "full", "id": uuid.uuid4().hex}, room=room)

        # 链式接话
        # is_chain=False（直接回用户）：60% 概率触发链
        # is_chain=True（已在链中）：45% 概率继续，让对话自然收尾
        # 这样对话会从"回应用户"逐渐漂移成"AI 们自己聊"
        if chat_counter < 400 and not is_startup:
            # 话题越热链越容易继续，话题冷了更容易死
            chain_prob = (0.40 if is_chain else 0.58) * (0.6 + TOPIC_STATE["heat"] * 0.4)
            next_role = select_chain_participant(role, merged)
            if next_role and random.random() < chain_prob:
                delay = random.uniform(4.0, 10.0) if is_chain else random.uniform(3.0, 7.0)
                def _chain(r=next_role, d=delay, uid=user_id, rm=room):
                    socketio.sleep(d)
                    trigger_ai_reply(r, role, merged, uid, rm, is_chain=True)
                socketio.start_background_task(_chain)
    except Exception as e:
        print(f"[ERROR] trigger_ai_reply failed: {e}")
        traceback.print_exc()
        pass


def trigger_dm_reply(role, user_content, user_id, room):
    cfg = CUSTOM_CONFIG[role]
    try:
        clean_user_content = (user_content or "").replace("[VOICE]", "").strip()
        sys_prompt, user_task = make_dm_prompt(role, clean_user_content, user_id)
        resp_content = call_llm(
            role,
            messages=[{"role": "user", "content": user_task}],
            max_tokens=cfg["max_tokens"], temperature=cfg["temperature"],
            system_prompt=sys_prompt,
        )
        raw = clean_raw(role, resp_content)
        img_desc, raw = extract_image(raw)
        img_url = search_image_url(img_desc, role) if img_desc else ""

        bubbles = split_bubbles(raw)
        merged = " ".join(bubbles) if bubbles else raw

        # 强制发图：用户要求发图但AI没输出[IMG:]时，用回复内容搜图
        if not img_url and user_content:
            img_keywords = ["发图", "发图片", "发张图", "照片", "图片", "图看看", "发出来", "发我", "发来", "来张", "自拍", "拍张"]
            if any(k in (user_content or "") for k in img_keywords):
                print(f"[DEBUG] DM force image search for {role}", flush=True)
                img_url = search_image_url(merged[:60], role)

        # 安全网：只出了图却没有任何文字时，补一句自然短语，避免"只发图不说话"
        if img_url and not bubbles:
            merged = random.choice(["喏", "看这个", "刚拍的", "呐", "给你看", "瞅瞅"])
            bubbles = [merged]

        if not bubbles and not img_url: return

        voice_request_keywords = ["发语音", "听语音", "说句话", "发条语音", "语音说", "用语音"]
        force_audio = (
            any(k in (user_content or "") for k in voice_request_keywords)
            or "[VOICE]" in (user_content or "")
            or "[VOICE]" in merged
        )

        # 私聊随机语音 18%（私聊比群聊更亲密，语音频率可稍高）
        audio_url = ""
        # 语音内容统一用 merged（所有气泡合并），避免与附加位置错位
        if (random.random() < 0.18 and len(merged) < 100) or force_audio:
            clean_text = merged.replace("[VOICE]", "")
            audio_text = clean_text[:200] if force_audio and len(clean_text) > 200 else clean_text
            audio_url = generate_audio_url(audio_text, role)
            
        # Clean up marker in bubbles
        bubbles = [b.replace("[VOICE]", "") for b in bubbles]

        db = get_mdb()
        for i, bubble in enumerate(bubbles):
            t, elapsed = simulate_typing(role, len(bubble)), 0
            while elapsed < t:
                socketio.emit("dm_reply", {"role": role, "content": "", "mode": "typing"}, room=room)
                chunk = min(2.4, t - elapsed)
                socketio.sleep(chunk)
                elapsed += chunk
            is_last = (i == len(bubbles) - 1)
            msg_payload = {"role": role, "content": bubble, "mode": "full", "id": uuid.uuid4().hex}
            if is_last and audio_url:
                msg_payload["audio"] = audio_url
            socketio.emit("dm_reply", msg_payload, room=room)
            db["dm_messages"].insert_one({
                "user_id": user_id, "chat_role": role, "sender": role,
                "content": bubble, "image": "", "audio": audio_url if is_last else "",
                "timestamp": datetime.utcnow()
            })
            if not is_last: socketio.sleep(random.uniform(0.4, 1.0))

        if img_url:
            socketio.emit("dm_reply", {"role": role, "content": "", "image": img_url, "mode": "full", "id": uuid.uuid4().hex}, room=room)
            db["dm_messages"].insert_one({
                "user_id": user_id, "chat_role": role, "sender": role,
                "content": "", "image": img_url, "timestamp": datetime.utcnow()
            })
        
        # Removed old duplicate audio emission block
    except Exception:
        pass


def _require_auth():
    """从 Authorization header 提取并验证 JWT，返回 user_id 或 None"""
    token = request.headers.get("Authorization", "").removeprefix("Bearer ").strip()
    return verify_token(token) if token else None


@app.route("/")
def index(): return render_template("index.html")


@app.route("/health")
def health(): return jsonify({"status": "ok"}), 200


@app.route("/api/config", methods=["GET"])
def get_config(): return jsonify({"config": CUSTOM_CONFIG, "persona_base": PERSONA_BASE})


@app.route("/api/config", methods=["POST"])
def update_config():
    data = request.get_json(force=True)
    if 'global_max_tokens' in data:
        new_tokens = int(data['global_max_tokens'])
        CUSTOM_CONFIG['global']['max_tokens'] = new_tokens
        for role in ["ENFP", "INFP", "ENTP", "ENTJ", "ESTJ", "ENFJ"]: CUSTOM_CONFIG[role]['max_tokens'] = new_tokens
    for section, values in data.items():
        if section in CUSTOM_CONFIG and isinstance(values, dict): CUSTOM_CONFIG[section].update(values)
    return jsonify({"ok": True, "config": CUSTOM_CONFIG})


@app.route("/api/history/group", methods=["GET"])
def get_group_history():
    user = _require_auth()
    if not user: return jsonify({"error": "未登录"}), 401
    db = get_mdb()
    rows = list(db["messages"].find({"user_id": user["user_id"]}, sort=[("timestamp", -1)], limit=50))
    rows.reverse()
    return jsonify({"history": [
        {"role": m["role"], "content": m.get("content",""), "image": m.get("image",""), "audio": m.get("audio","")}
        for m in rows
    ]})


@app.route("/api/history/dm/<role>", methods=["GET"])
def get_dm_history(role):
    user = _require_auth()
    if not user: return jsonify({"error": "未登录"}), 401
    db = get_mdb()
    rows = list(db["dm_messages"].find({"user_id": user["user_id"], "chat_role": role}, sort=[("timestamp", -1)], limit=50))
    rows.reverse()
    return jsonify({"history": [
        {"sender": m["sender"], "content": m.get("content",""), "image": m.get("image",""), "audio": m.get("audio","")}
        for m in rows
    ]})


@app.route("/api/moments", methods=["GET"])
def get_moments():
    user = _require_auth()
    if not user: return jsonify({"error": "未登录"}), 401
    db  = get_mdb()
    uid = user["user_id"]
    rows = list(db["moments"].find({"user_id": uid}, sort=[("created_at", -1)], limit=50))
    result = []
    for m in rows:
        mid = str(m["_id"])
        comments = list(db["comments"].find({"user_id": uid, "moment_id": mid}, sort=[("created_at", 1)]))
        result.append({
            "id": mid, "role": m["role"], "content": m.get("content",""),
            "image": m.get("image",""),
            "images": m.get("images") or ([m["image"]] if m.get("image") else []),
            "audio": m.get("audio",""),
            "created_at": str(m.get("created_at","")),
            "likes": [x for x in (m.get("likes") or "").split(",") if x],
            "liked_by_user": "我" in (m.get("likes") or ""),
            "comments": [{"id": c.get("id",""), "role": c["role"], "content": c["content"],
                          "created_at": str(c.get("created_at",""))} for c in comments]
        })
    return jsonify({"moments": result})


@app.route("/api/moments/comment", methods=["POST"])
def post_comment():
    user = _require_auth()
    if not user: return jsonify({"error": "未登录"}), 401
    data    = request.get_json(force=True)
    mid     = data.get("moment_id")
    content = data.get("content", "").strip()
    if not mid or not content: return jsonify({"ok": False}), 400
    db  = get_mdb()
    uid = user["user_id"]
    room = uid
    cid, now = str(uuid.uuid4()), datetime.utcnow()
    db["comments"].insert_one({"id": cid, "moment_id": mid, "role": "我",
                                "content": content, "created_at": now, "user_id": uid})
    socketio.emit("new_comment", {"id": cid, "moment_id": mid, "role": "我",
                                  "content": content, "created_at": str(now)}, room=room)
    moment = db["moments"].find_one({"_id": ObjectId(mid), "user_id": uid})
    if moment and moment["role"] in ["ENFP","INFP","ENTP","ENTJ","ESTJ","ENFJ"]:
        socketio.start_background_task(trigger_comment_reply, moment["role"], mid, content, uid, room)
    return jsonify({"ok": True, "id": cid, "created_at": str(now)})


def trigger_comment_reply(role, moment_id, user_comment, user_id, room):
    socketio.sleep(random.uniform(2.0, 5.0))
    try:
        prompt = (f"{PERSONA_BASE[role]}\n你在朋友圈发了一条动态，刚才「我」（用户）评论了你：「{user_comment}」\n"
                  f"请回复这条评论。简短自然，像在朋友圈回复朋友一样。\n规则：1.不要带引号。2.不要太长。3.符合人设。")
        reply_content = call_llm(role, messages=[{"role": "user", "content": prompt}], max_tokens=40, temperature=0.85)
        reply_content = clean_raw(role, reply_content)
        if reply_content:
            db = get_mdb()
            cid, now = str(uuid.uuid4()), datetime.utcnow()
            db["comments"].insert_one({"id": cid, "moment_id": moment_id, "role": role,
                                       "content": reply_content, "created_at": now, "user_id": user_id})
            socketio.emit("new_comment", {"id": cid, "moment_id": moment_id, "role": role,
                                          "content": reply_content, "created_at": str(now)}, room=room)
    except Exception:
        pass


@app.route("/api/moments/generate", methods=["POST"])
def generate_moments():
    user = _require_auth()
    if not user: return jsonify({"error": "未登录"}), 401
    uid  = user["user_id"]
    room = uid
    roles = random.sample(["ENFP", "INFP", "ENTP", "ENTJ", "ESTJ", "ENFJ"], k=random.randint(1, 2))
    for role in roles:
        socketio.start_background_task(create_single_moment, role, uid, room)
    return jsonify({"ok": True, "generating": len(roles)})


@app.route("/api/moments/create", methods=["POST"])
def create_moment():
    """用户（我）发朋友圈：文字 + 最多 9 张本地上传图片。"""
    user = _require_auth()
    if not user: return jsonify({"error": "未登录"}), 401
    uid  = user["user_id"]
    room = uid
    content = (request.form.get("content") or "").strip()
    files = request.files.getlist("images")[:9]

    from eventlet import tpool
    urls = []
    upload_dir = os.path.join("static", "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    for f in files:
        if not f or not f.filename:
            continue
        safe = re.sub(r'[^\w\.\-]', '', f.filename) or "img"
        fn = f"moment_{uuid.uuid4()}_{safe}"
        tpool.execute(f.save, os.path.join(upload_dir, fn))
        urls.append(f"/static/uploads/{fn}")

    if not content and not urls:
        return jsonify({"ok": False, "error": "内容和图片不能都为空"}), 400

    now = datetime.utcnow()
    db  = get_mdb()
    doc = {"user_id": uid, "role": "INFJ", "content": content,
           "image": urls[0] if urls else "", "images": urls, "audio": "",
           "likes": "", "created_at": now}
    res = db["moments"].insert_one(doc)
    mid = str(res.inserted_id)

    # 让 1~2 个角色来评论我的动态，热闹一点
    reactors = random.sample(["ENFP", "INFP", "ENTP", "ENTJ", "ESTJ", "ENFJ"], k=random.randint(1, 2))
    for r in reactors:
        socketio.start_background_task(trigger_moment_reaction, r, mid, content, uid, room)

    return jsonify({"ok": True, "moment": {
        "id": mid, "role": "INFJ", "content": content, "image": doc["image"],
        "images": urls, "audio": "", "created_at": str(now),
        "likes": [], "liked_by_user": False, "comments": []
    }})


def trigger_moment_reaction(role, moment_id, user_content, user_id, room):
    """角色对『我』刚发的朋友圈评论一句。"""
    socketio.sleep(random.uniform(2.0, 6.0))
    try:
        prompt = (f"{PERSONA_BASE[role]}\n『我』（用户）刚在朋友圈发了一条动态：「{user_content or '（配图，无文字）'}」\n"
                  f"请以朋友的口吻评论一句。规则：1.不要带引号。2.简短自然、别太长。3.符合人设。")
        reply = clean_raw(role, call_llm(role, messages=[{"role": "user", "content": prompt}],
                                         max_tokens=40, temperature=0.85))
        if reply:
            db = get_mdb()
            cid, now = str(uuid.uuid4()), datetime.utcnow()
            db["comments"].insert_one({"id": cid, "moment_id": moment_id, "role": role,
                                       "content": reply, "created_at": now, "user_id": user_id})
            socketio.emit("new_comment", {"id": cid, "moment_id": moment_id, "role": role,
                                          "content": reply, "created_at": str(now)}, room=room)
    except Exception:
        pass


@app.route("/api/moments/like", methods=["POST"])
def like_moment():
    user = _require_auth()
    if not user: return jsonify({"error": "未登录"}), 401
    data = request.get_json(force=True)
    mid, like = data.get("moment_id"), data.get("like", True)
    db  = get_mdb()
    row = db["moments"].find_one({"_id": ObjectId(mid), "user_id": user["user_id"]})
    if not row: return jsonify({"ok": False}), 404
    likes = [x for x in (row.get("likes") or "").split(",") if x]
    if like and "我" not in likes:    likes.append("我")
    elif not like and "我" in likes:  likes.remove("我")
    db["moments"].update_one({"_id": ObjectId(mid)}, {"$set": {"likes": ",".join(likes)}})
    return jsonify({"ok": True, "likes": likes, "liked_by_user": "我" in likes})


def delayed_trigger(role, trigger_role, content, user_id, room, delay, force_mode="respond"):
    socketio.sleep(delay)
    trigger_ai_reply(role, trigger_role, content, user_id, room, force_mode=force_mode)


@socketio.on("connect")
def on_connect():
    token = request.args.get("token", "")
    user  = verify_token(token)
    if not user:
        return False   # 拒绝连接
    sid = request.sid
    uid = user["user_id"]
    ACTIVE_USERS[sid] = {"user_id": uid, "username": user["username"]}
    USER_ROOMS.setdefault(uid, set()).add(sid)
    join_room(uid)
    print(f"[Socket] 连接: {sid} | 用户: {user['username']}", flush=True)


@socketio.on("disconnect")
def on_disconnect():
    sid = request.sid
    info = ACTIVE_USERS.pop(sid, None)
    if info:
        uid = info["user_id"]
        USER_ROOMS.get(uid, set()).discard(sid)
        if not USER_ROOMS.get(uid):
            USER_ROOMS.pop(uid, None)
    print(f"[Socket] 断开: {sid}", flush=True)


@socketio.on("user_message")
def handle_msg(data):
    sid  = request.sid
    info = ACTIVE_USERS.get(sid)
    if not info: return
    user_id, room = info["user_id"], info["user_id"]

    global chat_counter
    chat_counter = 0
    content = data.get("content", "").strip()
    if not content: return

    db = get_mdb()
    db["messages"].insert_one({
        "user_id": user_id, "role": "INFJ",
        "content": content, "timestamp": datetime.utcnow()
    })
    update_topic(content, "INFJ")
    
    # Reset consecutive counters for AI
    with state_lock:
        for role in ["ENFP", "INFP", "ENTP", "ENTJ", "ESTJ", "ENFJ"]: ROLE_STATE[role]["consecutive"] = 0
    
    mentioned = set()
    uc = content.upper()
    if "@ENFP" in uc: mentioned.add("ENFP")
    if "@INFP" in uc: mentioned.add("INFP")
    if "@ENTP" in uc: mentioned.add("ENTP")
    if "@ENTJ" in uc: mentioned.add("ENTJ")
    if "@ESTJ" in uc: mentioned.add("ESTJ")
    if "@ENFJ" in uc: mentioned.add("ENFJ")
    
    responders = []
    if mentioned:
        responders = list(mentioned)
        print(f"[DEBUG] Mentions found: {responders}")
    else:
        cfg = CUSTOM_CONFIG
        if random.random() < cfg.get("ENFP", {}).get("trigger_prob", 0): responders.append("ENFP")
        if random.random() < cfg.get("INFP", {}).get("trigger_prob", 0): responders.append("INFP")
        if random.random() < cfg.get("ENTP", {}).get("trigger_prob", 0): responders.append("ENTP")
        if random.random() < cfg.get("ENTJ", {}).get("trigger_prob", 0): responders.append("ENTJ")
        if random.random() < cfg.get("ESTJ", {}).get("trigger_prob", 0): responders.append("ESTJ")
        if random.random() < cfg.get("ENFJ", {}).get("trigger_prob", 0): responders.append("ENFJ")
        if not responders: responders.append(random.choice(["ENFP", "INFP", "ENTP", "ENTJ", "ESTJ", "ENFJ"]))
        # 真实群聊感：最多同时 3 人响应，超过则随机保留
        if len(responders) > 3:
            responders = random.sample(responders, 3)
    
    if not mentioned: random.shuffle(responders)
    
    print(f"[DEBUG] Responders selected: {responders}")
    for i, role in enumerate(responders):
        delay = (0.5 if role in mentioned else 1.5) + i * 2.5
        socketio.start_background_task(delayed_trigger, role, "INFJ", content, user_id, room, delay)


@socketio.on("user_image")
def handle_user_image(data):
    sid  = request.sid
    info = ACTIVE_USERS.get(sid)
    if not info: return
    user_id, room = info["user_id"], info["user_id"]

    url  = data.get("url")
    desc = data.get("description", "一张图片")
    db   = get_mdb()
    db["messages"].insert_one({
        "user_id": user_id, "role": "INFJ",
        "content": f"[图片] {desc}", "image": url, "timestamp": datetime.utcnow()
    })
    content_for_ai = f"[用户发了一张图片]\n【图片内容】{desc}"
    responders = ["ENFP", "INFP", "ENTP", "ENTJ", "ESTJ", "ENFJ"]
    random.shuffle(responders)
    for i in range(random.randint(1, 3)):
        role = responders[i]
        def _r(r=role, idx=i, uid=user_id, rm=room):
            socketio.sleep(random.uniform(1.0, 2.5) + idx * 2.0)
            trigger_ai_reply(r, "INFJ", content_for_ai, uid, rm, force_mode="respond")
        socketio.start_background_task(_r)


@socketio.on("dm_message")
def handle_dm(data):
    sid  = request.sid
    info = ACTIVE_USERS.get(sid)
    if not info: return
    user_id, room = info["user_id"], info["user_id"]

    role        = data.get("target_role", data.get("role", ""))
    content     = data.get("content", "").strip()
    image       = data.get("image", "")
    image_desc  = data.get("image_desc", "")

    if (not content and not image) or role not in ["ENFP", "INFP", "ENTP", "ENTJ", "ESTJ", "ENFJ"]: return

    db = get_mdb()
    if image:
        content = f"[图片] {image_desc}" if not content else content + f" [图片] {image_desc}"
        db["dm_messages"].insert_one({"user_id": user_id, "chat_role": role, "sender": "INFJ",
                                      "content": content, "image": image, "timestamp": datetime.utcnow()})
    else:
        db["dm_messages"].insert_one({"user_id": user_id, "chat_role": role, "sender": "INFJ",
                                      "content": content, "timestamp": datetime.utcnow()})

    def _reply(r=role, c=content, uid=user_id, rm=room):
        socketio.sleep(random.uniform(0.8, 2.0))
        trigger_dm_reply(r, c, uid, rm)
    socketio.start_background_task(_reply)


def memory_manager():
    time.sleep(120)
    while True:
        try:
            global chat_counter
            chat_counter = max(0, chat_counter - 30)
            now = datetime.utcnow()
            db  = get_mdb()

            # 对每个活跃用户生成摘要
            for uid in list(USER_ROOMS.keys()):
                last_summary = db["summaries"].find_one({"user_id": uid}, sort=[("end_time", -1)])
                if last_summary:
                    le = last_summary["end_time"]
                    if not isinstance(le, datetime): le = now - timedelta(hours=13)
                    if (now - le).total_seconds() < 12 * 3600:
                        continue
                    start_time = le
                else:
                    start_time = now - timedelta(hours=12)
                end_time = start_time + timedelta(hours=12)

                msgs = list(db["messages"].find(
                    {"user_id": uid, "timestamp": {"$gte": start_time, "$lt": end_time}},
                    sort=[("timestamp", 1)]
                ))
                if not msgs: continue
                msg_text = "\n".join([f"{m['role']}: {m.get('content','')}" for m in msgs])
                prompt   = (f"这是六个好朋友在微信群里的聊天记录。\n时间段：{start_time} 到 {end_time}\n"
                            f"请总结这段时间内发生了什么有趣的事、大家讨论了什么话题。\n要求：简练、不超过150字。\n\n聊天记录：\n{msg_text}")
                try:
                    sc = call_llm("INFJ", messages=[{"role": "user", "content": prompt}], max_tokens=200)
                    db["summaries"].insert_one({
                        "user_id": uid, "start_time": start_time, "end_time": end_time,
                        "content": sc.strip(), "created_at": now
                    })
                except Exception:
                    pass

                # 清理 7 天前的摘要
                db["summaries"].delete_many({
                    "user_id": uid, "end_time": {"$lt": now - timedelta(days=7)}
                })

            time.sleep(600)
        except Exception:
            time.sleep(60)


def proactive_talker_thread():
    time.sleep(random.uniform(60, 120))
    while True:
        try:
            now  = datetime.utcnow()
            hour = now.hour
            if 2 <= hour < 7:
                time.sleep(600)
                continue

            decay_topic()

            active_snapshot = list(USER_ROOMS.items())
            for uid, sids in active_snapshot:
                if not sids: continue
                room = uid
                db   = get_mdb()
                all_roles = ["ENFP", "INFP", "ENTP", "ENTJ", "ESTJ", "ENFJ"]

                # 群聊破冰
                last_msg = db["messages"].find_one({"user_id": uid}, sort=[("timestamp", -1)])
                lt = last_msg["timestamp"] if last_msg else now - timedelta(hours=1)
                if not isinstance(lt, datetime): lt = now - timedelta(hours=1)
                silence = (now - lt).total_seconds()

                random.shuffle(all_roles)
                for candidate in all_roles:
                    mn, mx = CUSTOM_CONFIG[candidate].get("group_interval", (60, 300))
                    if silence > random.randint(mn, mx):
                        _c, _uid, _rm = candidate, uid, room
                        socketio.start_background_task(
                            lambda c=_c, u=_uid, r=_rm: trigger_ai_reply(c, "", "", u, r, is_startup=True)
                        )
                        break

                # 私聊破冰
                candidate = random.choice(all_roles)
                cfg = CUSTOM_CONFIG[candidate]
                if random.random() < cfg.get("proactive_dm", 0.1):
                    last_dm = db["dm_messages"].find_one(
                        {"user_id": uid, "chat_role": candidate}, sort=[("timestamp", -1)])
                    dm_lt = last_dm["timestamp"] if last_dm else now - timedelta(days=1)
                    if not isinstance(dm_lt, datetime): dm_lt = now - timedelta(days=1)
                    if (now - dm_lt).total_seconds() / 3600 > random.uniform(2, 6):
                        prompt = (f"{PERSONA_BASE[candidate]}\n你发现很久没和INFJ（我）私聊了。请主动发一条消息给ta。\n"
                                  f"可以是分享生活、关心近况、或者单纯的吐槽。\n要求：自然、简短、符合人设、不要太突兀。")
                        def _dm_break(r=candidate, p=prompt, u=uid, rm=room):
                            try:
                                content = call_llm(r, messages=[{"role": "user", "content": p}], max_tokens=60)
                                content = clean_raw(r, content)
                                if content:
                                    get_mdb()["dm_messages"].insert_one({
                                        "user_id": u, "chat_role": r, "sender": r,
                                        "content": content, "timestamp": datetime.utcnow()
                                    })
                                    socketio.emit("new_dm_message", {"role": r, "content": content}, room=rm)
                            except Exception as e:
                                print(f"DM Break failed: {e}")
                        socketio.start_background_task(_dm_break)

                # 朋友圈
                if random.random() < 0.10:
                    role = random.choice(all_roles)
                    socketio.start_background_task(create_single_moment, role, uid, room)

            time.sleep(60)
        except Exception as e:
            print(f"Proactive thread error: {e}")
            time.sleep(120)


@app.route("/api/call/reply", methods=["POST"])
def call_reply():
    """实时语音通话：接收我说的话（已转文字）+ 简短通话历史，返回角色的口语回复文字 + 语音。"""
    user = _require_auth()
    if not user: return jsonify({"error": "未登录"}), 401
    data = request.get_json(force=True) or {}
    role = data.get("role")
    text = (data.get("text") or "").strip()
    greeting = bool(data.get("greeting"))
    if role not in PERSONA_BASE:
        return jsonify({"ok": False, "error": "角色不存在"}), 400
    if not text and not greeting:
        return jsonify({"ok": False, "error": "没有内容"}), 400

    hist = data.get("history") or []
    convo = ""
    for h in hist[-6:]:
        who = "我" if h.get("from") == "user" else ROLE_NAME.get(role, role)
        convo += f"{who}：{h.get('text','')}\n"

    sys_prompt = (PERSONA_BASE[role] + "\n\n"
                  "现在你正在和『我』语音通话。像真人打电话一样说话：口语、简短、自然，"
                  "一次只说一两句话，别念标点符号、别写括号动作或旁白、别分点罗列、别升华总结。")
    if greeting:
        user_prompt = "电话刚接通，你先开口打个招呼（一句话，像平时接起朋友电话那样）。"
    else:
        user_prompt = f"通话记录：\n{convo}\n我刚说：「{text}」\n用一两句话自然地回我。"

    try:
        reply = call_llm(role, messages=[{"role": "user", "content": user_prompt}],
                         system_prompt=sys_prompt, max_tokens=120, temperature=0.9)
        reply = clean_raw(role, reply)
        reply = extract_image(reply)[1].replace("[VOICE]", "").strip()
    except Exception as e:
        print(f"[Call] LLM 异常: {e}", flush=True)
        return jsonify({"ok": False, "error": "生成失败"}), 500

    audio = ""
    if reply:
        try:
            audio = generate_audio_url(reply[:80], role)
        except Exception as e:
            print(f"[Call] TTS 异常: {e}", flush=True)
    return jsonify({"ok": True, "text": reply, "audio": audio})


@app.route("/api/upload_image", methods=["POST"])
def upload_image():
    if 'image' not in request.files:
        return jsonify({"ok": False, "error": "No image file provided"}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({"ok": False, "error": "No selected file"}), 400
        
    if file:
        try:
            safe_name = re.sub(r'[^\w\.\-]', '', file.filename) or "image"
            filename = f"upload_{uuid.uuid4()}_{safe_name}"
            upload_dir = os.path.join("static", "uploads")
            os.makedirs(upload_dir, exist_ok=True)
            filepath = os.path.join(upload_dir, filename)
            
            # Use tpool for file I/O to avoid blocking main thread
            from eventlet import tpool
            tpool.execute(file.save, filepath)
            
            # Call Vision Model (Qwen-VL-Max) to get description
            description = "一张图片"
            try:
                with open(filepath, "rb") as image_file:
                    base64_image = base64.b64encode(image_file.read()).decode('utf-8')

                # Detect actual MIME type from file extension
                ext = os.path.splitext(safe_name)[1].lower()
                mime_map = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                            ".gif": "image/gif", ".webp": "image/webp", ".bmp": "image/bmp"}
                mime = mime_map.get(ext, "image/jpeg")

                # 使用硅基流动 Qwen2.5-VL 视觉模型（OpenAI 兼容格式）
                print("[Vision] Calling SiliconFlow Qwen2.5-VL-7B-Instruct...", flush=True)
                def _call_vision():
                    return client_sf.chat.completions.create(
                        model="Qwen/Qwen2.5-VL-7B-Instruct",
                        messages=[{
                            "role": "user",
                            "content": [
                                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{base64_image}"}},
                                {"type": "text", "text": "请简要描述这张图片的内容，包括主要物体、场景和氛围。"}
                            ]
                        }],
                        max_tokens=200
                    )
                vision_resp = tpool.execute(_call_vision)
                description = (vision_resp.choices[0].message.content or "").strip()
                print(f"[Vision] Description: {description}", flush=True)
            except Exception as e:
                print(f"[Vision] Processing Failed: {e}", flush=True)
                traceback.print_exc()

            return jsonify({"ok": True, "url": f"/static/uploads/{filename}", "description": description})

        except Exception as e:
            print(f"[Upload] Error: {e}", flush=True)
            traceback.print_exc()
            return jsonify({"ok": False, "error": str(e)}), 500

    return jsonify({"ok": False}), 500


@app.route("/api/voice_to_text", methods=["POST"])
def voice_to_text():
    """接收音频，转 wav 后调用硅基流动 SenseVoiceSmall 转文字"""
    if 'audio' not in request.files:
        return jsonify({"ok": False, "error": "No audio file"}), 400

    file = request.files['audio']
    original_filename = file.filename or "recording.webm"
    from eventlet import tpool

    filepath = None
    convert_path = None
    try:
        start_time = time.time()
        print(f"[STT] Start processing: {file.filename}", flush=True)

        # 1. 保存原始上传文件
        ext = os.path.splitext(original_filename)[1] or ".webm"
        filename = f"voice_{uuid.uuid4()}{ext}"
        temp_dir = os.path.join("static", "temp_audio")
        os.makedirs(temp_dir, exist_ok=True)
        filepath = os.path.join(temp_dir, filename)
        tpool.execute(file.save, filepath)

        # 2. 转 16kHz 单声道 wav（SenseVoice 不支持 webm）
        convert_path = filepath
        if ext.lower() != ".wav":
            wav_path = os.path.splitext(filepath)[0] + ".wav"
            try:
                ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
                print(f"[STT] Converting to WAV...", flush=True)
                result = tpool.execute(lambda: subprocess.run(
                    [ffmpeg_exe, "-y", "-i", filepath, "-ar", "16000", "-ac", "1", wav_path],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30
                ))
                if os.path.exists(wav_path) and os.path.getsize(wav_path) > 0:
                    convert_path = wav_path
                else:
                    print(f"[STT] Conversion produced no file, using original", flush=True)
            except Exception as e:
                print(f"[STT] Conversion failed (using original): {e}", flush=True)

        # 3. 调用硅基流动 SenseVoiceSmall
        print(f"[STT] SiliconFlow SenseVoiceSmall on {convert_path}...", flush=True)

        def _call_stt():
            with open(convert_path, "rb") as af:
                return client_sf.audio.transcriptions.create(
                    model="FunAudioLLM/SenseVoiceSmall",
                    file=(os.path.basename(convert_path), af),
                )
        transcription = tpool.execute(_call_stt)
        text = (transcription.text or "").strip()
        print(f"[STT] Result in {time.time()-start_time:.2f}s: {text[:100]}", flush=True)

        if text:
            return jsonify({"ok": True, "text": text, "meta": {}})
        return jsonify({"ok": False, "error": "未识别到文字内容", "meta": {}}), 500

    except Exception as e:
        print(f"[STT] Failed: {e}", flush=True)
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(e)}), 500
    finally:
        # 调用完成后再清理临时文件
        for p in {filepath, convert_path}:
            if p and os.path.exists(p):
                try: os.remove(p)
                except Exception: pass

def _clean_old_files(directory, max_age=3600):
    if not os.path.exists(directory):
        return
    now = time.time()
    for f in os.listdir(directory):
        f_path = os.path.join(directory, f)
        if os.path.isfile(f_path) and os.stat(f_path).st_mtime < now - max_age:
            try:
                os.remove(f_path)
            except Exception:
                pass

def cleanup_temp_files():
    while True:
        try:
            # 临时转换文件：1 小时清理
            _clean_old_files(os.path.join("static", "temp_audio"), 3600)
            # TTS 语音消息：保留 24 小时，避免聊天记录里的语音泡泡失效
            _clean_old_files(os.path.join("static", "audio"), 86400)
        except Exception:
            pass
        time.sleep(3600)


# ══════════════════════════════════════════════════════════════



# ══════════════════════════════════════════════════════════════

# 🌟 修复点：底部代码被精简，确保多线程和服务器只启动一次
threading.Thread(target=proactive_talker_thread, daemon=True).start()
threading.Thread(target=memory_manager, daemon=True).start()
threading.Thread(target=cleanup_temp_files, daemon=True).start()

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5001))
    socketio.run(app, debug=False, use_reloader=False, port=port, host='0.0.0.0')
