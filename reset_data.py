"""
清空所有账号的聊天记录与记忆（保留用户账号本身）。
用途：人设重塑后，旧对话里残留的过时设定（如"闺女/朵朵"等）会被模型读进上下文，
清一次即可。在服务器上运行： python reset_data.py --yes
不加 --yes 只做演练(dry-run)，打印将删除的条数，不真正删除。
"""
import sys
from db_mongo import get_db

# 要清空的会话/记忆集合；users(账号) 不动
COLLECTIONS = ["messages", "dm_messages", "moments", "comments", "summaries"]


def main():
    do = "--yes" in sys.argv
    db = get_db()
    print("=== reset_data " + ("(执行删除)" if do else "(演练 dry-run，加 --yes 才真正删除)") + " ===", flush=True)
    for c in COLLECTIONS:
        n = db[c].count_documents({})
        if do:
            db[c].delete_many({})
            print(f"  {c}: 删除 {n} 条", flush=True)
        else:
            print(f"  {c}: 将删除 {n} 条", flush=True)
    users = db["users"].count_documents({})
    print(f"  users(保留): {users} 个账号", flush=True)
    print("完成。" if do else "以上为演练，未删除。", flush=True)


if __name__ == "__main__":
    main()
