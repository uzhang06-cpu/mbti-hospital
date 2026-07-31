"""
生成 6 个角色的 AI 头像（硅基流动 Kolors），存到 static/avatars/{ROLE}.png。
- 幂等：已存在则跳过（传 --force 覆盖重生）。
- 单独运行：  python gen_avatars.py [--force]
- 也被 app.py 启动时后台调用 ensure_avatars()（用服务器上的 SILICONFLOW_API_KEY）。
用户(INFJ) 和 群聊 图标保持前端内置 SVG，不在这里生成。
"""
import os
import sys
import requests

AVATAR_DIR = os.path.join("static", "avatars")
SF_KEY = os.getenv("SILICONFLOW_API_KEY", "")

# 统一画风前缀，保证 6 张头像风格一致
STYLE = ("扁平矢量插画头像，简约现代，柔和马卡龙配色，纯色干净背景，"
         "正脸头像特写，可爱亲和，统一画风，不要文字不要边框")

PROMPTS = {
    "ENFP": "活泼开朗、笑容灿烂的大二女生，扎马尾，暖橙色调",
    "INFP": "安静文艺、表情淡淡的长发女生，柔和蓝色调",
    "ENTP": "机灵爱笑、眼神狡黠的短发男生，紫色调",
    "ENTJ": "干练自信、眼神坚定的利落短发女生，红色调",
    "ESTJ": "认真稳重、干净整洁的学生班长，青绿色调",
    "ENFJ": "温暖亲和、微笑的男生，明黄色调",
}


def _gen_one(role: str, desc: str, force: bool = False) -> bool:
    path = os.path.join(AVATAR_DIR, f"{role}.png")
    if os.path.exists(path) and not force:
        return True
    if not SF_KEY:
        print("[Avatar] 缺少 SILICONFLOW_API_KEY，跳过生成", flush=True)
        return False
    try:
        resp = requests.post(
            "https://api.siliconflow.cn/v1/images/generations",
            headers={"Authorization": f"Bearer {SF_KEY}", "Content-Type": "application/json"},
            json={
                "model": "Kwai-Kolors/Kolors",
                "prompt": f"{STYLE}，{desc}",
                "image_size": "1024x1024",
                "batch_size": 1,
                "num_inference_steps": 20,
            },
            timeout=90,
        )
        if resp.status_code != 200:
            print(f"[Avatar] {role} HTTP {resp.status_code}: {resp.text[:150]}", flush=True)
            return False
        data = resp.json()
        imgs = data.get("images") or data.get("data") or []
        url = imgs[0].get("url") if imgs else ""
        if not url:
            print(f"[Avatar] {role} 返回无 url", flush=True)
            return False
        img = requests.get(url, timeout=60)
        if img.status_code != 200 or not img.content:
            return False
        os.makedirs(AVATAR_DIR, exist_ok=True)
        with open(path, "wb") as f:
            f.write(img.content)
        print(f"[Avatar] {role} 生成成功 ({len(img.content)//1024}KB)", flush=True)
        return True
    except Exception as e:
        print(f"[Avatar] {role} 异常: {e}", flush=True)
        return False


def ensure_avatars(force: bool = False):
    """幂等生成。首次(无 .genv2 标记)会重生一遍以套用新学生人设，之后跳过。
    生成失败时保留旧图作兜底，不写标记(下次重试)。"""
    os.makedirs(AVATAR_DIR, exist_ok=True)
    marker = os.path.join(AVATAR_DIR, ".genv2")
    do_force = force or not os.path.exists(marker)
    ok_all = True
    for role, desc in PROMPTS.items():
        if not _gen_one(role, desc, force=do_force):
            ok_all = False
    if do_force and ok_all and SF_KEY:
        try:
            with open(marker, "w") as f:
                f.write("v2")
        except Exception:
            pass


if __name__ == "__main__":
    ensure_avatars(force="--force" in sys.argv)
