import requests
import os
import uuid

API_KEY = "sk-ce766e4bb72b40e29960801e235e7a7c"

def test_tts(text, model):
    print(f"Testing TTS with model: {model}")
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    
    # Try Qwen-TTS Multimodal Endpoint
    data = {
        "model": "qwen-tts-latest",
        "input": {"text": text},
        "parameters": {
            "voice": "longxiaochun" 
        }
    }
    
    try:
        resp = requests.post("https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation",
                             headers=headers, json=data, timeout=30)
        
        print(f"Status Code: {resp.status_code}")
        print(f"Headers: {resp.headers}")
        
        if resp.status_code == 200:
            ct = resp.headers.get('Content-Type', '')
            if 'application/json' in ct:
                print(f"Response is JSON (Likely Error or Task): {resp.json()}")
            else:
                print(f"Response is Binary (Success). Size: {len(resp.content)} bytes")
                with open(f"test_{model}.wav", "wb") as f:
                    f.write(resp.content)
                print(f"Saved to test_{model}.wav")
        else:
            print(f"Error Response: {resp.text}")
            
    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    # Test ENFP voice
    test_tts("你好，我是快乐的小狗！", "sambert-zhitian-v1")
