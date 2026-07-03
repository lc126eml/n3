import requests
import datetime
import pytz
from typing import Any, Union
# 替换为你的真实信息
SUPABASE_URL = "https://xvtowanhrprunhhbhtte.supabase.co"
API_KEY = "sb_publishable_-pAGDwqIrM2c7eAMHYjBXQ_oXxIpqIn"
def log_to_supabase(task: str, status: bool, message: str = "") -> None:
    """Log a kernel status row.

    Args:
        task: Kaggle kernel id, for example "owner/slug".
        status: Boolean Supabase status column. False means running; True means finished.
        message: Free-form status detail. Successful finished logs should contain "SUCCESS".
    """
    url = f"{SUPABASE_URL}/rest/v1/training_logs"
    headers = {
        "apikey": API_KEY,
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=representation"
    }
    payload = {
        "task": task,
        "status": status,
        "message": message
    }
    requests.post(url, headers=headers, json=payload)
# 使用示例
# task: kernel_id
# status:
    # false: running 
    # true: finished (message contains "SUCCESS" if success)
# try:
#     # 你的模型训练逻辑
#     log_to_supabase("owner/slug", True, "SUCCESS 模型已保存至 /output/model.pth")
# except Exception as e:
#     log_to_supabase("owner/slug", True, f"FAILED {e}")

# SUPABASE_URL = "你的Project URL"
# API_KEY = "你的ANON_KEY"
def fetch_new_logs(
    minutes: int | float | None = None,
    last_stamp: str | None = None,
    last_id: int | None = None,
    cleanup_minutes: int | float = 3,
) -> list[dict[str, Any]] | None:
    headers = {"apikey": API_KEY, "Authorization": f"Bearer {API_KEY}"}
    url = f"{SUPABASE_URL}/rest/v1/training_logs"
    end_stamp = (datetime.datetime.now(pytz.utc) - datetime.timedelta(minutes=cleanup_minutes)).isoformat()
    params: dict[str, Union[str, list[str]]] = {
        "order": "id.asc",
        "created_at": f"lt.{end_stamp}"
    }
    if minutes is not None:
        mins_ago = (datetime.datetime.now(pytz.utc) - datetime.timedelta(minutes=minutes)).isoformat()
        params["created_at"] = [f"gte.{mins_ago}", f"lt.{end_stamp}"] 
    elif last_id is not None:
        params["id"] = f"gt.{last_id}"
    elif last_stamp is not None:
        params["created_at"] = [f"gt.{last_stamp}", f"lt.{end_stamp}"] 
    else:
        return None

    response = requests.get(url, headers=headers, params=params)
    
    if response.status_code == 200:
        logs = response.json()
        return logs
    else:
        return None

# 每分钟执行一次
# logs = fetch_new_logs(minutes=10)

# for log in logs:
#     print(f"检测到状态更新: status time: {log['created_at']}; kernel_id: {log['task']};  log['status']}; {log['message']}")
    # 在这里执行你的本地处理逻辑