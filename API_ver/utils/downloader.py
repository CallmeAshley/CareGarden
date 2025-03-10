import os
import requests

def download_audio(audio_url, file_path):
    """음성 파일을 다운로드"""
    response = requests.get(audio_url, stream=True)
    if response.status_code == 200:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"다운로드 완료: {file_path}")
    else:
        print(f"다운로드 실패: {response.status_code}")
