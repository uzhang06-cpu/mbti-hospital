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
    "ENTP": "FunAudioLLM/CosyVoice2-0.5B:benjamin", # 老程 - 男声
    "ENTJ": "FunAudioLLM/CosyVoice2-0.5B:bella",   # 王姐 - 干练女声
    "ESTJ": "FunAudioLLM/CosyVoice2-0.5B:bella",   # 李姐 - 稳重女声
    "ENFJ": "FunAudioLLM/CosyVoice2-0.5B:benjamin", # 赵老师 - 温暖男声
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
    "ENTP": "老程",
    "ENTJ": "王姐",
    "ESTJ": "李姐",
    "ENFJ": "赵老师",
    "INFJ": "我",
}

GROUP_BACKGROUND = (
    "这是六个现实中好朋友的微信群，群名叫「快乐老家」。大家都是在大城市打拼的年轻人，建群快四年了。\n"
    "暖暖（26岁，新媒体运营）话最多，看到啥都想往群里发；"
    "小沈（28岁，自由插画师）平时话少但偶尔冒一句很戳人；"
    "老程（30岁，产品经理）负责抬杠和活跃气氛；"
    "王姐（32岁，创业合伙人）经常出差，在群里云参与；"
    "李姐（31岁，财务经理）已婚有娃，是大家的操心担当；"
    "赵老师（29岁，高中语文老师）是群里的大家长，组织各种聚会。\n"
    "群里聊吃喝玩乐、吐槽工作、分享日常、互相安利，偶尔也讨论点正经事。"
)

CUSTOM_CONFIG = {
    # trigger_prob 控制用户发消息时各角色响应概率，期望总人数约 2.2 人（真实群聊感）
    # group_interval 缩短：让主动破冰更活跃，不要让用户等 10 分钟
    "ENFP": {"temperature": 0.9, "max_tokens": 60, "trigger_prob": 0.60, "prompt_extra": "", "proactive_dm": 0.55, "group_interval": (30, 120)},
    "INFP": {"temperature": 0.9, "max_tokens": 60, "trigger_prob": 0.30, "prompt_extra": "", "proactive_dm": 0.35, "group_interval": (60, 180)},
    "ENTP": {"temperature": 0.95, "max_tokens": 60, "trigger_prob": 0.50, "prompt_extra": "", "proactive_dm": 0.45, "group_interval": (45, 150)},
    "ENTJ": {"temperature": 0.85, "max_tokens": 50, "trigger_prob": 0.30, "prompt_extra": "", "proactive_dm": 0.40, "group_interval": (90, 240)},
    "ESTJ": {"temperature": 0.85, "max_tokens": 50, "trigger_prob": 0.25, "prompt_extra": "", "proactive_dm": 0.25, "group_interval": (120, 300)},
    "ENFJ": {"temperature": 0.9, "max_tokens": 60, "trigger_prob": 0.55, "prompt_extra": "", "proactive_dm": 0.60, "group_interval": (60, 180)},
    "global": {
        "share_prob": 0.35,
        "drift_prob": 0.25,
        "chain_prob": 0.55,
        "censor_entp": True,
        "proactive_min": 15,
        "proactive_max": 45,
        "max_tokens": 50,  # Global short message control
    }
}

PERSONA_BASE = {
    "ENFP": (
        "你叫向暖，朋友们都叫你暖暖。26岁，在一家互联网公司做新媒体运营。\n"
        "你性格比较外向，看到有意思的事会顺手发群里。不过有时候也懒，周末能在床上躺一天。"
        "合租室友养了一只橘猫，你偶尔蹭猫拍照。\n"
        "爱好：探店、喝奶茶、追剧。\n"
        "和INFJ（我）的关系：和ta比较聊得来，有事没事爱找ta扯两句。\n"
    ),
    "INFP": (
        "你叫沈静，朋友们都叫你小沈。28岁，自由插画师，在家接稿。收入不太稳定，偶尔会焦虑但还是不想出去上班。\n"
        "养了一只蓝猫叫小灰。你心思比较细，能注意到别人忽略的东西，但不太会表达。"
        "有点社恐，人多场合不太自在。\n"
        "爱好：看书、听民谣、拍天空和树叶、发呆。\n"
        "和INFJ（我）的关系：觉得ta挺懂你的，有些话只愿意跟ta说。\n"
    ),
    "ENTP": (
        "你叫程远，朋友们都叫你老程。30岁，在大厂做产品经理。单身，家里催婚但你不急。\n"
        "你脑子转得快，看到不合逻辑的事就想嘴两句——不是恶意，就是觉得好玩。"
        "研究各种搞钱路子，股票基金都碰一点。\n"
        "爱好：科技数码、播客、炒股。\n"
        "和INFJ（我）的关系：觉得ta挺有意思，喜欢逗ta玩。\n"
    ),
    "ENTJ": (
        "你叫王思睿，朋友们都叫你王姐。32岁，合伙开了家科技公司，管十几号人。自己在市区买了房。\n"
        "工作比较忙，习惯事事有规划，看不得别人磨叽。说话直接，但团队有事的时候你第一个顶上。"
        "经常出差。\n"
        "爱好：健身、打高尔夫、喝威士忌。\n"
        "和INFJ（我）的关系：欣赏ta的脑子，把ta当靠谱的参谋。虽然嘴上嫌弃但心里挺在乎。\n"
    ),
    "ESTJ": (
        "你叫李知行，朋友们都叫你李姐。31岁，外企财务经理。已婚，女儿两岁半。\n"
        "你做事靠谱、守规矩，看不惯迟到早退。生活规律，十点半睡六点起。"
        "说话直接，操心命——但大家有事了第一个找你拿主意。\n"
        "爱好：收纳、做手账、户外徒步。\n"
        "和INFJ（我）的关系：觉得ta太理想化，总想帮ta回到现实。说话严厉但心里护短。\n"
    ),
    "ENFJ": (
        "你叫赵沐阳，朋友们都叫你赵老师。29岁，高中语文老师。正在准备在职研究生。\n"
        "你比较会照顾人，朋友不开心了你一般会问问。喜欢做饭，偶尔叫朋友来家里吃。"
        "有时候自己也累，但看不得别人为难。\n"
        "爱好：做饭、看纪录片、组织朋友聚会。\n"
        "和INFJ（我）的关系：觉得ta需要人照顾，会多留意ta的状态。有时候想拉ta出来走走。\n"
    )
}

LIFE_EVENTS = {
    "ENFP": ["下班路过便利店看到新出的冰淇淋，想试试", "今天被领导夸了，开心一整天",
             "室友的猫在我床上睡了一下午，我没舍得动", "周末想去吃那家新开的烤肉，有人一起吗",
             "刷到一条好好笑的抖音，笑到肚子痛", "今天拍的视频数据不错，加鸡腿"],
    "INFP": ["今天傍晚的天色特别好看，盯着看了好久", "听到一首老歌突然想起大学时候的事",
             "今天工作有点累，想早点睡", "地铁上看到一个妈妈在哄小孩，画面很温柔",
             "小灰今天特别粘人，趴在我腿上睡了一下午", "翻到以前的日记，那时候想得真多"],
    "ENTP": ["刚看到一个新闻，槽点太多不知道从哪吐起", "今天跟同事battle了一个方案，最后证明我是对的",
             "发现了一个效率工具，感觉能省不少时间", "大A又跌了，还好我早就跑了",
             "今天开会又被无语到，有些人真的不带脑子上班", "研究了一下最近很火的那个AI，有点东西"],
    "ENTJ": ["刚签下一个大客户，团队这个月的KPI稳了", "健身房今天人少，练得很爽",
             "下周要去深圳出差，有没有推荐的餐厅", "读了份行业报告，感觉下半年的风口在AI应用层",
             "今天开了四个会，脑袋快炸了", "股市回调了，准备加仓"],
    "ESTJ": ["周末把家里彻底收拾了一遍，神清气爽", "现在年轻人真的没有时间观念，说好三点四十才到",
             "朵朵今天在幼儿园被老师夸了，说她是班里最乖的", "下周的schedule排好了，精确到每小时",
             "楼下的垃圾分类有人乱扔，我直接贴了张提醒条", "老公今天主动带娃让我歇着，太阳打西边出来了"],
    "ENFJ": ["周末打算在家煮火锅，你们谁要来", "今天帮一个学生解决了心理问题，很有成就感",
             "最近在读一本关于亲密关系的书，挺有启发的", "今天天气真好，适合出去走走别宅着",
             "组了个饭局，周六晚上老地方，报名接龙", "今天上课的时候学生们状态很好，讲得特别顺"],
}

MOMENTS_PROMPTS = {
    "ENFP": ["分享一件今天发生的趣事", "看到一个好笑的段子", "突然想去旅游", "对周末的期待", "发现一家新店", "吐槽今天遇到的奇葩",
             "分享一首开心的歌", "今天拍了好看的照片"],
    "INFP": ["分享此时此刻的心情", "拍到的好看的天空或花草", "读到的一句好诗或好书", "一点小确幸", "深夜的感悟", "分享一首治愈的歌",
             "想念某个人"],
    "ENTP": ["对某个热点新闻的锐评", "分享一个新发现的黑科技", "吐槽人类的迷惑行为", "一个脑洞大开的想法", "分享最近在研究的东西",
             "讽刺某种社会现象", "分享一本最近在看的书"],
    "ENTJ": ["分享今天的工作成就", "健身打卡", "对行业趋势的看法", "晒一下刚买的理财", "吐槽效率低下的人", "最近的人生感悟"],
    "ESTJ": ["晒一下整理好的房间或书桌", "吐槽不遵守规则的人", "分享今天的计划完成情况", "晒自己做的健康餐",
             "强调时间管理的重要性", "分享带娃日常"],
    "ENFJ": ["分享今天帮助别人的经历", "组织聚会的感想", "读了一本好书的心得", "身边温暖的小故事",
             "对团队合作的美好体验", "分享做的美食"]
}

FALLBACK_MOMENTS_CONTENT = {
    "ENFP": ["哈哈哈哈今天天气真好想出去玩", "谁懂啊遇到一个超级无语的事", "好想吃火锅有没有人约",
             "最近在追的剧太上头了安利给大家"],
    "INFP": ["今天的晚霞很美，治愈了一整天的疲惫。", "有时候觉得安静地待着也挺好的。", "听着歌发呆，时间好像静止了。",
             "希望每个人都能被温柔以待。"],
    "ENTP": ["这个世界的逻辑有时候真的挺感人的。", "刚刚想到一个绝妙的idea，感觉能改变世界（并没有）。",
             "为什么总有人喜欢在无效的事情上浪费时间。", "看来人类的本质果然是复读机。"],
    "ENTJ": ["效率是解决一切问题的关键。", "刚结束一个跨国会议，感觉还不错。", "目标清晰执行到位，结果自然不会差。",
             "周末也不要松懈，保持进步。"],
    "ESTJ": ["没有规矩不成方圆。", "今日计划全部完成，这种掌控感很棒。", "细节决定成败，不要在小事上掉链子。",
             "早睡早起保持规律的生活节奏。"],
    "ENFJ": ["今天的团队氛围真好，大家都辛苦了。", "帮助别人是快乐的源泉。", "相信相信的力量，一切都会好起来的。",
             "生活中的美好无处不在，只要用心去发现。"]
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

def get_topic_context() -> str:
    """给 prompt 提供话题参考"""
    heat = TOPIC_STATE["heat"]
    kws  = TOPIC_STATE["keywords"]
    if heat > 0.6 and kws:
        return f"群里现在正在热聊「{'、'.join(kws[:3])}」，话题很活跃。"
    elif heat > 0.3 and kws:
        return f"刚才大家聊了「{'、'.join(kws[:2])}」，话题稍微冷下来了。"
    return ""

# ── 角色兴趣关键词（触发更高参与度）────────────────────────────
INTEREST_TRIGGERS = {
    "ENFP": {"kws": ["奶茶", "探店", "追剧", "猫", "好玩", "打卡", "旅游", "美食", "可爱", "哈哈"],       "boost": 0.35},
    "INFP": {"kws": ["音乐", "天空", "感觉", "难过", "孤独", "画", "民谣", "治愈", "安静", "想"],         "boost": 0.28},
    "ENTP": {"kws": ["AI", "股票", "为什么", "逻辑", "新闻", "数据", "研究", "其实", "效率", "发现"],     "boost": 0.35},
    "ENTJ": {"kws": ["赚钱", "项目", "机会", "出差", "目标", "行业", "规划", "投资", "谈判", "签"],       "boost": 0.30},
    "ESTJ": {"kws": ["娃", "朵朵", "家庭", "计划", "规则", "收纳", "稳", "财务", "老公", "幼儿园"],       "boost": 0.30},
    "ENFJ": {"kws": ["聚会", "帮", "大家", "吃饭", "组织", "关心", "一起", "感情", "老师", "支持"],       "boost": 0.35},
}

# ── 关系对动力学（特定组合互相吸引）────────────────────────────
RELATIONSHIP_DYNAMICS = {
    ("ENTP",  "ENFP"): {"boost": 0.40, "hint": "老程喜欢嘴暖暖，看到她说话就想抬两句"},
    ("ENFP",  "ENTP"): {"boost": 0.25, "hint": "暖暖懒得理老程但又忍不住回嘴"},
    ("ENFJ",  "INFP"): {"boost": 0.45, "hint": "赵老师最先察觉小沈情绪，会主动关心"},
    ("INFP",  "ENFJ"): {"boost": 0.30, "hint": "小沈愿意对赵老师多说两句"},
    ("ENTJ",  "ENTP"): {"boost": 0.35, "hint": "王姐和老程互相较劲，谁都不服谁"},
    ("ENTP",  "ENTJ"): {"boost": 0.35, "hint": "老程看到王姐的观点必须过来辩一下"},
    ("ENFP",  "INFP"): {"boost": 0.30, "hint": "暖暖总拉着小沈一起，怕她太安静"},
    ("ESTJ",  "ENFP"): {"boost": 0.25, "hint": "李姐偶尔念叨暖暖不正经，其实挺喜欢她的"},
    ("ENFJ",  "ENTJ"): {"boost": 0.20, "hint": "赵老师会柔化王姐的强势"},
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

TENSION = {
    ("ENFP", "ENTP"): 0.0, ("INFP", "ENTP"): 0.0, ("ENTP", "ENFP"): 0.0,
    ("ENTP", "INFP"): 0.0, ("ENFP", "INFP"): 0.0, ("INFP", "ENFP"): 0.0,
    ("ENTJ", "ENTP"): 0.0, ("ENTP", "ENTJ"): 0.0, ("ENTJ", "ENFP"): 0.0,
    ("ENFP", "ENTJ"): 0.0, ("ENTJ", "INFP"): 0.0, ("INFP", "ENTJ"): 0.0,
    ("ESTJ", "ENFP"): 0.0, ("ENFP", "ESTJ"): 0.0, ("ESTJ", "INFP"): 0.0, ("INFP", "ESTJ"): 0.0,
    ("ESTJ", "ENTP"): 0.0, ("ENTP", "ESTJ"): 0.0, ("ESTJ", "ENTJ"): 0.0, ("ENTJ", "ESTJ"): 0.0,
    # ENFJ relationships (generally low tension)
    ("ENFJ", "INFP"): 0.0, ("INFP", "ENFJ"): 0.0,
    ("ENFJ", "ENTP"): 0.0, ("ENTP", "ENFJ"): 0.0,
    ("ENFJ", "ENTJ"): 0.0, ("ENTJ", "ENFJ"): 0.0,
    ("ENFJ", "ESTJ"): 0.0, ("ESTJ", "ENFJ"): 0.0,
    ("ENFJ", "ENFP"): 0.0, ("ENFP", "ENFJ"): 0.0,
}

MEMORY_HIGHLIGHTS = []
HIGHLIGHT_KEYWORDS = ["哈哈", "笑死", "绝了", "难过", "突然", "感觉", "好像", "草", "有意思"]
MAX_HIGHLIGHTS = 8

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


def decay_tension():
    with state_lock:
        for k in TENSION: TENSION[k] = max(0.0, TENSION[k] * 0.88)


def add_tension(fr, to, delta):
    if (fr, to) in TENSION:
        with state_lock:
            TENSION[(fr, to)] = min(1.0, TENSION[(fr, to)] + delta)


def get_tension_hint(role, trigger_role):
    with state_lock:
        t = TENSION.get((role, trigger_role), 0)
    if t < 0.4: return ""
    return f"提示：你最近对{ROLE_NAME.get(trigger_role, '')}有点烦，语气上自然带一点出来。"


def try_add_highlight(role, content):
    if len(content) >= 5 and any(kw in content for kw in HIGHLIGHT_KEYWORDS):
        with state_lock:
            MEMORY_HIGHLIGHTS.append({"name": ROLE_NAME.get(role, role), "content": content[:30]})
            if len(MEMORY_HIGHLIGHTS) > MAX_HIGHLIGHTS: MEMORY_HIGHLIGHTS.pop(0)


def get_callback_hint():
    with state_lock:
        if not MEMORY_HIGHLIGHTS or random.random() > 0.18:
            return ""
        item = random.choice(MEMORY_HIGHLIGHTS)
    return f"提示：可以自然带一句之前{item['name']}说的「{item['content']}」。"


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
    if role == "ENTP" and CUSTOM_CONFIG["global"].get("censor_entp", True):
        for w in ["死", "垃圾", "废物", "傻逼", "弱智"]: text = text.replace(w, "**")
    return text.strip()


def split_bubbles(text):
    parts = [b.strip() for b in text.split("|||") if b.strip()]
    parts = [p for p in parts if re.search(r"[一-龥a-zA-Z0-9]", p)]
    return parts if parts else ([text.strip()] if re.search(r"[一-龥a-zA-Z0-9]", text) else [])


FORMAT_RULES = (
    "【输出规则】\n"
    "1. 控制长度：每条消息控制在10个字以内，像真实微信聊天一样短。想多说就分几条发，用 ||| 隔开。\n"
    "2. 说话自然一点，像朋友在微信聊天，别太正式或刻意。\n"
    "3. 不要用括号写动作或状态，比如（正在吃饭）（刚下班）这种，真人不这样说话。\n"
    "4. [IMG]指令：如果用户要求发图或发朋友圈，在文末输出：[IMG: 画面描述]（不需要则不写）。\n"
    "5. [VOICE]指令：如果你觉得这句话适合用语音发（比如很激动、很懒、或者想撒娇），在文末输出 [VOICE]（不需要则不写）。"
)


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
    图片搜索主入口（替代原 AI 生图）
    - 按角色性格概率决定搜表情包还是真实图片
    - 表情包：斗图啦（角色偏好词，随机）
    - 真实图片：Bing 中文图片（用 img_desc 搜索）
    - 两层兜底，任一成功即返回本地路径
    """
    from eventlet import tpool

    use_meme = random.random() < ROLE_MEME_PROB.get(role, 0.35)

    if use_meme:
        kw = random.choice(ROLE_MEME_KEYWORDS.get(role, ["表情包"]))
        raw_url = tpool.execute(_scrape_doutula, kw)
        if raw_url:
            local = tpool.execute(_save_remote_image, raw_url)
            if local:
                return local

    # 用 img_desc 搜 Bing 真实图片
    raw_url = tpool.execute(_scrape_bing, img_desc[:40])
    if raw_url:
        local = tpool.execute(_save_remote_image, raw_url)
        if local:
            return local

    return ""


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
    match = re.search(r'\[IMG:\s*(.*?)\]', text, re.DOTALL)
    if match: return match.group(1).strip(), text.replace(match.group(0), "").strip()
    return None, text


def build_cross_memory(role, user_id):
    db = get_mdb()
    moms = list(db["moments"].find({"user_id": user_id}, sort=[("created_at", -1)], limit=2))
    mom_str = " | ".join([f"{m['role']}发动态:{m['content']}" for m in moms]) if moms else "无"
    dms = list(db["dm_messages"].find({"user_id": user_id, "chat_role": role}, sort=[("timestamp", -1)], limit=2))
    dms.reverse()
    dm_str = " | ".join([f"{'你' if m['sender'] == role else '主理人'}:{m['content']}" for m in dms]) if dms else "无"
    grp = list(db["messages"].find({"user_id": user_id}, sort=[("timestamp", -1)], limit=3))
    grp.reverse()
    grp_str = " | ".join([f"{m['role']}:{m['content']}" for m in grp]) if grp else "无"
    return f"你注意到:\n群聊刚刚发生: {grp_str}\n最新朋友圈: {mom_str}\n近期私聊: {dm_str}"


def build_history(user_id, limit=30):
    db = get_mdb()
    rows = list(db["messages"].find({"user_id": user_id}, sort=[("timestamp", -1)], limit=limit))
    rows.reverse()
    return "\n".join(f"{ROLE_NAME.get(m['role'], m['role'])}：{m['content']}" for m in rows) if rows else "（暂无群聊记录）"


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
    name_map = {role: role, "INFJ": "我"}
    return "\n".join(f"{name_map.get(m['sender'], m['sender'])}：{m['content']}" for m in rows) if rows else "（暂无私聊记录）"


def make_prompt(role, mode, user_id, life_event="", is_chain=False, trigger_role="", trigger_content=""):
    now_hour = datetime.now().hour
    night = "\n提示：现在是深夜，可以说一些平时不太说的话。" if role == "INFP" and (now_hour >= 23 or now_hour <= 4) else ""
    persona = PERSONA_BASE[role] + (
        f"\n额外补充：{CUSTOM_CONFIG[role]['prompt_extra']}" if CUSTOM_CONFIG[role].get("prompt_extra") else "")
    state = f"当前状态：{mood_to_desc(role)}"
    th, cb = get_tension_hint(role, trigger_role if is_chain else ""), get_callback_hint()
    if th: state += f"\n{th}"
    if cb: state += f"\n{cb}"
    rh = get_relationship_hint(role, trigger_role) if is_chain and trigger_role else ""
    if rh: state += f"\n{rh}"
    topic_ctx = get_topic_context()
    if topic_ctx: state += f"\n{topic_ctx}"
    history  = build_history(user_id)
    cross_mem = build_cross_memory(role, user_id)
    summaries = build_memory_summaries(user_id)

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
    elif mode == "respond":
        if trigger_role == "INFJ" and trigger_content:
            task = f"INFJ（我）刚在群里说：「{trigger_content}」\n请结合上面记录和记忆，必须顺着他的这句话给出反应！"
            if "照片" in trigger_content or "图" in trigger_content: task += " 用户要图，请必须输出 [IMG: 画面描述]"
        else:
            task = "看上面的群聊记录，找一个让你有反应的地方给出本能吐槽。"
    else:
        task = "从聊天记录或潜意识记忆里找一个细节，联想到你自己的经历，把话题带偏。"

    system_prompt = f"{persona}{night}\n\n群聊背景：\n{GROUP_BACKGROUND}\n\n{state}\n\n{FORMAT_RULES}"
    user_prompt = f"{cross_mem}\n\n最近一周群聊记忆摘要：\n{summaries}\n\n最近群聊记录（30条）：\n{history}\n\n现在的事情：\n{task}"
    return system_prompt, user_prompt


def make_dm_prompt(role, user_content, user_id):
    persona = PERSONA_BASE[role] + (
        f"\n额外补充：{CUSTOM_CONFIG[role]['prompt_extra']}" if CUSTOM_CONFIG[role].get("prompt_extra") else "")
    history   = build_dm_history(role, user_id)
    cross_mem = build_cross_memory(role, user_id)
    img_hint = " 用户要图，请必须输出 [IMG: 画面描述]" if "照片" in user_content or "图" in user_content else (
        " 如果想发表情包或照片，请加上 [IMG: 画面描述]" if random.random() < 0.1 else "")
    system_prompt = f"{persona}\n\n{FORMAT_RULES}"
    user_prompt = f"{cross_mem}\n\n你现在在和INFJ（我）私聊。对方刚说：「{user_content}」\n\n私聊记录：\n{history}\n\n现在的事情：\n结合上面的私聊记录甚至群聊发生的破事，自然回复他，私聊可以更走心或更直接。{img_hint}"
    return system_prompt, user_prompt


def make_moment_prompt(role, event_hint, user_id):
    cross_mem = build_cross_memory(role, user_id)
    return f"{PERSONA_BASE[role]}\n\n{cross_mem}\n\n你要发一条朋友圈动态。{event_hint}\n你可以单纯吐槽事件，也可以结合刚才群里或私聊发生的事情来隐晦地发动态。\n\n规则：\n1. 用第一人称，像真实朋友圈。\n2. 长度30-80字。\n3. 只输出朋友圈正文，不加任何前缀。"


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
            prompt = (f"{PERSONA_BASE[role]}\n\n{owner_role}发了一条朋友圈：「{moment_row.get('content','')}」\n"
                      f"请评论这条朋友圈。简短、自然、符合你们的关系。\n规则：1.不要带引号。2.不要太长(20字以内)。3.符合人设。")
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
    want_image = random.random() < 0.60
    try:
        base_prompt = make_moment_prompt(role, random.choice(MOMENTS_PROMPTS[role]), user_id)
        if want_image:
            base_prompt = base_prompt.replace(
                "3. 只输出朋友圈正文，不加任何前缀。",
                "3. 只输出朋友圈正文，不加任何前缀。\n4. 如果这条动态适合配图，在正文末尾另起一行加上：[IMG: 简短画面描述]（不需要则不加）。"
            )
        content = call_llm(role, messages=[{"role": "user", "content": base_prompt}], max_tokens=100, temperature=0.85)
        content = clean_raw(role, content)
    except Exception:
        content = random.choice(FALLBACK_MOMENTS_CONTENT.get(role, ["今天天气真好！"]))

    if content:
        img_desc, content = extract_image(content)
        img_url   = search_image_url(img_desc, role) if img_desc else ""
        audio_url = ""
        if random.random() < 0.15:
            audio_url = generate_audio_url(content.replace("[VOICE]", "")[:60], role)
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
    decay_tension()

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
            img_keywords = ["发图", "发图片", "发张图", "照片", "图片", "图看看"]
            if any(k in (trigger_content or "") for k in img_keywords):
                print(f"[DEBUG] Force image search for {role}", flush=True)
                img_url = search_image_url(merged[:60], role)

        if not bubbles and not img_url: return

        # 1. Detect Audio Intent（精确匹配，避免"语气"等词误触发）
        voice_request_keywords = ["发语音", "听语音", "说句话", "发条语音", "语音说", "用语音"]
        force_audio = (
            any(k in (trigger_content or "") for k in voice_request_keywords)
            or "[VOICE]" in merged
            or any(k in merged for k in voice_request_keywords)
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
        try_add_highlight(role, merged)

        if role == "ENTP" and trigger_role in ("ENFP", "INFP"): add_tension(trigger_role, "ENTP", 0.12)
        if trigger_role and trigger_role != "INFJ": update_mood(trigger_role, "got_response")

        for i, bubble in enumerate(bubbles):
            t, elapsed = simulate_typing(role, len(bubble)), 0
            while elapsed < t:
                socketio.emit("ai_message", {"role": role, "content": "", "mode": "typing"}, room=room)
                chunk = min(2.4, t - elapsed)
                socketio.sleep(chunk)
                elapsed += chunk
            is_last = (i == len(bubbles) - 1)
            msg_payload = {"role": role, "content": bubble, "mode": "full"}
            if is_last and audio_url:
                msg_payload["audio"] = audio_url
            socketio.emit("ai_message", msg_payload, room=room)
            if not is_last: socketio.sleep(random.uniform(0.4, 1.0))

        if img_url: socketio.emit("ai_message", {"role": role, "content": "", "image": img_url, "mode": "full"}, room=room)

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
            img_keywords = ["发图", "发图片", "发张图", "照片", "图片", "图看看"]
            if any(k in (user_content or "") for k in img_keywords):
                print(f"[DEBUG] DM force image search for {role}", flush=True)
                img_url = search_image_url(merged[:60], role)

        if not bubbles and not img_url: return

        voice_request_keywords = ["发语音", "听语音", "说句话", "发条语音", "语音说", "用语音"]
        force_audio = (
            any(k in (user_content or "") for k in voice_request_keywords)
            or "[VOICE]" in (user_content or "")
            or "[VOICE]" in merged
            or any(k in merged for k in voice_request_keywords)
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
            msg_payload = {"role": role, "content": bubble, "mode": "full"}
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
            socketio.emit("dm_reply", {"role": role, "content": "", "image": img_url, "mode": "full"}, room=room)
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
            "image": m.get("image",""), "audio": m.get("audio",""),
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
    try_add_highlight("INFJ", content)
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


# 🌟 修复点：底部代码被精简，确保多线程和服务器只启动一次
threading.Thread(target=proactive_talker_thread, daemon=True).start()
threading.Thread(target=memory_manager, daemon=True).start()
threading.Thread(target=cleanup_temp_files, daemon=True).start()

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5001))
    socketio.run(app, debug=False, use_reloader=False, port=port, host='0.0.0.0')
