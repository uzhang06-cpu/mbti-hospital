"""
MongoDB 连接与集合定义
所有集合通过 user_id 字段隔离每个用户的数据
"""
import os
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import ConnectionFailure

MONGO_URI = os.getenv("MONGODB_URI", "")
JWT_SECRET = os.getenv("JWT_SECRET", "mbti_hospital_secret_2024")

_client = None
_db     = None

def get_db():
    global _client, _db
    if _db is not None:
        return _db
    if not MONGO_URI:
        raise RuntimeError("MONGODB_URI 环境变量未设置")
    _client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    _client.admin.command("ping")          # 连接检验
    _db = _client["mbti_hospital"]

    # 建索引（幂等）
    _db["users"].create_index("username", unique=True)
    _db["messages"].create_index([("user_id", ASCENDING), ("timestamp", DESCENDING)])
    _db["dm_messages"].create_index([("user_id", ASCENDING), ("chat_role", ASCENDING), ("timestamp", DESCENDING)])
    _db["moments"].create_index([("user_id", ASCENDING), ("created_at", DESCENDING)])
    _db["comments"].create_index([("user_id", ASCENDING), ("moment_id", ASCENDING)])
    _db["summaries"].create_index([("user_id", ASCENDING), ("end_time", DESCENDING)])
    print("[MongoDB] 连接成功", flush=True)
    return _db
