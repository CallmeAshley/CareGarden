import json
import requests
import time
from services.config import TYPECAST_API_KEY, ACTOR_ID

API_URL = "https://typecast.ai/api/speak"

def request_tts(sentence, actor_id=ACTOR_ID):

    headers = {
        "Authorization": f"Bearer {TYPECAST_API_KEY}",
        "Content-Type": "application/json"
    }
    data = json.dumps({
        "text": sentence.strip(),
        "lang": "auto",
        "actor_id": actor_id,
        "xapi_hd": True,
        "model_version": "latest"
    })

    response = requests.post(API_URL, headers=headers, data=data)
    if response.status_code == 200:
        response_json = response.json()
        return response_json.get("result", {}).get("speak_v2_url")
    else:
        print(f"TTS 요청 실패: {response.status_code}, {response.text}")
        return None

def wait_for_audio(speak_v2_url):
    """TTS 변환이 완료될 때까지 대기"""
    headers = {"Authorization": f"Bearer {TYPECAST_API_KEY}"}

    for _ in range(10):  # 최대 10초 동안 상태 체크
        response = requests.get(speak_v2_url, headers=headers)
        if response.status_code == 200:
            response_json = response.json()
            status = response_json["result"].get("status", "")
            if status == "done":
                return response_json["result"]["audio_download_url"]
        time.sleep(1)
    print("음성 생성 시간 초과")
    return None
