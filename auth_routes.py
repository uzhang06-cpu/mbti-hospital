"""
用户注册 / 登录路由
POST /api/auth/register  →  { token, username }
POST /api/auth/login     →  { token, username }
"""
import os, jwt, bcrypt
from datetime import datetime, timedelta
from flask import Blueprint, request, jsonify
from db_mongo import get_db, JWT_SECRET

auth_bp = Blueprint("auth", __name__)


def _make_token(user_id: str, username: str) -> str:
    payload = {
        "user_id":  user_id,
        "username": username,
        "exp": datetime.utcnow() + timedelta(days=7),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")


def verify_token(token: str) -> dict | None:
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
    except Exception:
        return None


@auth_bp.route("/api/auth/register", methods=["POST"])
def register():
    data     = request.get_json(force=True)
    username = (data.get("username") or "").strip()
    password = (data.get("password") or "")

    if not username or not password:
        return jsonify({"error": "用户名和密码不能为空"}), 400
    if len(username) < 2 or len(username) > 20:
        return jsonify({"error": "用户名长度 2~20 位"}), 400
    if len(password) < 6:
        return jsonify({"error": "密码至少 6 位"}), 400

    db = get_db()
    if db["users"].find_one({"username": username}):
        return jsonify({"error": "用户名已存在"}), 400

    pw_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    result  = db["users"].insert_one({
        "username":    username,
        "password":    pw_hash,
        "created_at":  datetime.utcnow(),
    })
    user_id = str(result.inserted_id)
    return jsonify({"token": _make_token(user_id, username), "username": username})


@auth_bp.route("/api/auth/login", methods=["POST"])
def login():
    data     = request.get_json(force=True)
    username = (data.get("username") or "").strip()
    password = (data.get("password") or "")

    if not username or not password:
        return jsonify({"error": "用户名和密码不能为空"}), 400

    db   = get_db()
    user = db["users"].find_one({"username": username})
    if not user:
        return jsonify({"error": "用户名或密码错误"}), 401
    if not bcrypt.checkpw(password.encode(), user["password"].encode()):
        return jsonify({"error": "用户名或密码错误"}), 401

    user_id = str(user["_id"])
    return jsonify({"token": _make_token(user_id, username), "username": username})
