import torch
from transformers import WhisperTokenizer, WhisperForConditionalGeneration
import librosa

# 모델 로드 함수
def load_model(model_path):
    # Whisper 모델과 tokenizer 로드
    tokenizer = WhisperTokenizer.from_pretrained(model_path)
    model = WhisperForConditionalGeneration.from_pretrained(model_path)

    return tokenizer, model

# 오디오 파일 로딩 함수 (librosa 사용)
def load_audio(audio_path):
    audio, _ = librosa.load(audio_path, sr=16000) 
    return audio

# 예측 함수
def transcribe_audio(audio_path, model_path):
    # 모델과 tokenizer 로드
    tokenizer, model = load_model(model_path)
    
    # 오디오 파일 로딩
    audio_input = load_audio(audio_path)

    # 텍스트 예측을 위한 tokenization
    input_features = tokenizer(audio_input, return_tensors="pt", padding=True, truncation=True)
    
    # 생성된 텍스트 IDs로 예측된 텍스트 추출
    generated_ids = model.generate(input_features["input_ids"])
    transcription = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    return transcription

if __name__ == "__main__":
    # 예시: 오디오 파일 경로
    audio_input_path = "C:/Users/001/Documents/간병인프로젝트/voice/dataset/test/남자환자.wav"
    model_path = "./lora_finetuned_whisper"  # fine-tuned 모델 경로 (로컬 경로)

    # 텍스트 예측
    transcription = transcribe_audio(audio_input_path, model_path)
    print("Predicted Transcription:", transcription)
