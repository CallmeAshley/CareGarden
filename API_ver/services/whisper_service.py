from openai import OpenAI
from services.config import OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)

def transcribe_audio(file_path):
    """Whisper를 이용하여 음성을 텍스트로 변환"""
    with open(file_path, "rb") as audio_file:
        response = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            temperature=0.2
        )
    return response.text.strip()





 