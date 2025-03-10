import time
import os
from services.whisper_service import transcribe_audio
from services.gpt_service import correct_text
from services.tts_service import request_tts, wait_for_audio
from utils.downloader import download_audio

# 데이터 폴더 설정
AUDIO_FILE_PATH = "C:/Users/001/Documents/간병인프로젝트/voice/dataset/train/custom_voice/고기중국어.wav"
OUTPUT_FOLDER = "C:/Users/001/Documents/간병인프로젝트/voice/model_result"
OUTPUT_FILE_PATH = os.path.join(OUTPUT_FOLDER, "seg1.wav")

def process_audio_to_tts(input_audio, output_file):
    """Whisper → GPT → TTS 순으로 실행"""
    start_time = time.time()

    # Whisper (음성 → 텍스트)
    w_s = time.time()                                                                             
    transcript = transcribe_audio(input_audio)
    print(transcript)
    w_e = time.time()

    # GPT (텍스트 교정)
    g_s = time.time()
    corrected_text = correct_text(transcript)
    print(corrected_text)
    g_e = time.time()

    # Typecast TTS (텍스트 → 음성)
    v_s = time.time()
    speak_v2_url = request_tts(corrected_text)
    if speak_v2_url:
        audio_url = wait_for_audio(speak_v2_url)
        if audio_url:
            download_audio(audio_url, output_file)
    v_e = time.time()

    print(f"실행 완료 (Whisper: {w_e-w_s:.2f}s, GPT: {g_e-g_s:.2f}s, TTS: {v_e-v_s:.2f}s, 총: {time.time() - start_time:.2f}s)")

if __name__ == "__main__":
    process_audio_to_tts(AUDIO_FILE_PATH, OUTPUT_FILE_PATH)
