import os
import json
from datasets import Dataset, DatasetDict
from tqdm import tqdm
from transformers import WhisperTokenizer, WhisperForConditionalGeneration, Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig
import torch

# 데이터 경로 설정
train_audio_dir = 'C:/Users/001/Documents/간병인프로젝트/voice/dataset/train/voice_processed'
train_transcript_dir = 'C:/Users/001/Documents/간병인프로젝트/voice/dataset/train/label'
val_audio_dir = 'C:/Users/001/Documents/간병인프로젝트/voice/dataset/val/voice_processed'
val_transcript_dir = 'C:/Users/001/Documents/간병인프로젝트/voice/dataset/val/label'

# 오디오 파일 경로와 해당 JSON 파일 경로를 매칭하여 텍스트 데이터를 읽어옴
def load_transcripts(audio_dir, transcript_dir):
    audio_paths = []
    transcripts = []
    
    # JSON 파일을 읽어들여 텍스트 추출
    for audio_file in tqdm(os.listdir(audio_dir)):
        if audio_file.endswith(".wav"):  # 오디오 파일만 필터링
            audio_path = os.path.join(audio_dir, audio_file)
            json_file = os.path.splitext(audio_file)[0] + ".json"
            json_path = os.path.join(transcript_dir, json_file)
            
            # JSON 파일 읽기
            with open(json_path, 'r') as f:
                transcript_data = json.load(f)
                transcripts.append(transcript_data["text"])
                audio_paths.append(audio_path)
    
    return {"audio": audio_paths, "text": transcripts}

# 학습 데이터셋 로드
train_data = load_transcripts(train_audio_dir, train_transcript_dir)
val_data = load_transcripts(val_audio_dir, val_transcript_dir)

# Dataset 생성
train_dataset = Dataset.from_dict(train_data)
val_dataset = Dataset.from_dict(val_data)

# DatasetDict로 묶어서 반환
dataset_dict = DatasetDict({
    "train": train_dataset,
    "validation": val_dataset
})


# Whisper 모델과 tokenizer 로드
model_name = "openai/whisper-base"
tokenizer = WhisperTokenizer.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name)

# Whisper 모델의 모든 파라미터 freezing
for param in model.parameters():
    param.requires_grad = False  # Whisper 모델의 파라미터는 동결

# LoRA 설정
lora_config = LoraConfig(
    r=8,  # LoRA rank
    lora_alpha=16,  # 학습률 비율
    lora_dropout=0.1,  # 드롭아웃 비율
    target_modules=["encoder.block.*.attn.q_proj", "encoder.block.*.attn.k_proj", "encoder.block.*.attn.v_proj"],  # LoRA를 적용할 모듈
)

# LoRA 모델로 변환
lora_model = get_peft_model(model, lora_config)


# 데이터 전처리 함수
def preprocess_function(examples):
    transcriptions = examples["text"]
    
    # 텍스트 데이터만 tokenizing 처리
    input_features = tokenizer(transcriptions, padding=True, truncation=True, return_tensors="pt")
    return input_features

# 데이터셋 전처리
train_dataset = train_dataset.map(preprocess_function, remove_columns=["audio", "text"])
val_dataset = val_dataset.map(preprocess_function, remove_columns=["audio", "text"])

# TrainingArguments 설정
training_args = TrainingArguments(
    output_dir="./results",  # 결과 저장 위치
    per_device_train_batch_size=4,  # 배치 크기 (메모리에 맞게 조절)
    per_device_eval_batch_size=4,
    num_train_epochs=10,  # 학습 에폭 수
    logging_dir="./logs",  # 로그 저장 위치
    evaluation_strategy="epoch",  # 평가 전략
    save_strategy="epoch",  # 모델 저장 전략
    learning_rate=5e-5,  # 학습률
    weight_decay=0.01,  # 가중치 감쇠
)

# Trainer 인스턴스 생성
trainer = Trainer(
    model=lora_model,  # fine-tuning 된 모델
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
)

# 학습 시작
trainer.train()

# 학습 완료 후 모델 저장
lora_model.save_pretrained("./lora_finetuned_whisper")
tokenizer.save_pretrained("./lora_finetuned_whisper")

