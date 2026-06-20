
import os
import requests
import time
import random

def generate_image_url(prompt):
    api_key = "sk-ce766e4bb72b40e29960801e235e7a7c"
    if not api_key:
        print("Error: API key not found")
        return ""
        
    headers = {
        "Authorization": f"Bearer {api_key}", 
        "X-DashScope-Async": "enable", 
        "Content-Type": "application/json"
    }
    data = {
        "model": "wanx-v1", 
        "input": {"prompt": prompt},
        "parameters": {"style": "<auto>", "size": "1024*1024", "n": 1}
    }
    
    try:
        print(f"Generating image: {prompt}", flush=True)
        resp = requests.post(
            "https://dashscope.aliyuncs.com/api/v1/services/aigc/text2image/image-synthesis",
            headers=headers, 
            json=data, 
            timeout=10
        )
        
        if resp.status_code != 200:
            print(f"Image Task Error: {resp.text}", flush=True)
            return ""
            
        task_id = resp.json().get('output', {}).get('task_id')
        if not task_id: return ""
        
        for _ in range(30):
            time.sleep(2)
            check_resp = requests.get(
                f"https://dashscope.aliyuncs.com/api/v1/tasks/{task_id}",
                headers={"Authorization": f"Bearer {api_key}"}, 
                timeout=10
            )
            if check_resp.status_code != 200: continue
            
            res = check_resp.json()
            status = res.get('output', {}).get('task_status')
            
            if status == 'SUCCEEDED':
                url = res['output']['results'][0]['url']
                print(f"Image Success: {url}", flush=True)
                return url
            elif status in ['FAILED', 'CANCELED']:
                print(f"Image Failed: {status}", flush=True)
                return ""
        return ""
    except Exception as e:
        print(f"Image Exception: {e}", flush=True)
        return ""

def download_and_save(url, filepath):
    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code == 200:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, "wb") as f:
                f.write(resp.content)
            print(f"Saved to {filepath}")
            return True
        else:
            print(f"Failed to download {url}")
            return False
    except Exception as e:
        print(f"Error saving {filepath}: {e}")
        return False

# ENFJ Prompt
prompt = "Hyper-realistic portrait of a warm and charismatic young woman, ENFJ personality, friendly smile, glowing skin, wearing a soft orange sweater, standing in a sunny park, natural lighting, approachable aura, 8k resolution, detailed texture, photography style."

print("--- Processing ENFJ ---")
url = generate_image_url(prompt)
if url:
    download_and_save(url, "static/images/avatar_ENFJ.png")
else:
    print("Skipping ENFJ due to generation failure")

print("Done!")
