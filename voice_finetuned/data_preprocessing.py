import os
import librosa
import numpy as np
import soundfile as sf

def convert_audio_safely(input_path, output_path, target_sr=16000):
    """
    - 스테레오 → 모노 변환 (더 강한 채널을 선택하거나, 동일하면 평균화)
    - 샘플링 레이트 변환 (16kHz)
    """
    # 오디오 로드 (스테레오 유지)
    audio, sr = librosa.load(input_path, sr=None, mono=False)

    # 스테레오 → 모노 변환 (더 강한 채널 선택)
    if len(audio.shape) > 1:  # 다중 채널(스테레오)일 경우
        left_channel = audio[0, :]  # L 채널
        right_channel = audio[1, :]  # R 채널

        # 각 채널의 에너지 계산 (절대값의 합)
        left_energy = np.sum(np.abs(left_channel))
        right_energy = np.sum(np.abs(right_channel))

        # 더 강한 채널 선택 (차이가 작으면 평균화)
        if abs(left_energy - right_energy) < 1e-3:  # 에너지가 거의 같으면 평균화
            audio = np.mean(audio, axis=0)
        elif left_energy > right_energy:
            audio = left_channel
        else:
            audio = right_channel
    else:
        # 오디오가 이미 모노인 경우 그대로 사용
        audio = audio

    # 샘플링 레이트 변환 (16kHz)
    if sr != target_sr:
        audio = librosa.resample(y=audio, orig_sr=sr, target_sr=target_sr)

    # 변환된 오디오 저장
    sf.write(output_path, audio, target_sr)

    print(f"변환 완료: {output_path}")


def remove_silence(audio, sr, min_silence_length=1.2, top_db=30):
    """
     일정한 간격을 두고 무음 제거
     min_silence_length: 유지할 최소 무음 길이 (초)
     top_db: 무음 감지 임계값 (낮을수록 작은 소리도 무음으로 감지)
    """
    non_silent_intervals = librosa.effects.split(audio, top_db=top_db)
    new_audio = []
    for i, (start, end) in enumerate(non_silent_intervals):
        new_audio.append(audio[start:end])
        if i < len(non_silent_intervals) - 1:
            silence_duration = min_silence_length * sr
            new_audio.append(np.zeros(int(silence_duration)))
    return np.concatenate(new_audio)


def split_audio_with_overlap(audio, sr, segment_length=30, overlap_ratio=0.25):
    """
     슬라이딩 윈도우 방식으로 30초 단위로 음성을 분할
     segment_length: 한 조각의 길이 (초)
     overlap_ratio: 슬라이딩 윈도우의 겹치는 비율 (0.0 ~ 1.0)
    """
    segment_samples = segment_length * sr
    step_size = int(segment_samples * (1 - overlap_ratio))
    segments = []
    for start in range(0, len(audio), step_size):
        end = min(start + segment_samples, len(audio))
        segments.append(audio[start:end])
        if end == len(audio):
            break
    return segments


def process_and_save_audio(input_dir, output_dir, sr=16000):
    """
     오디오 파일을 변환 후 전처리(무음 제거 + 30초 단위 분할) 후 저장
     input_dir: 원본 오디오 파일 경로
     output_dir: 전처리된 오디오 저장 경로
    """
    os.makedirs(output_dir, exist_ok=True)
    audio_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.wav')])
    
    for audio_file in audio_files:
        input_path = os.path.join(input_dir, audio_file)
        
        # Step 1: 오디오 변환 (모노화 및 샘플링 레이트 변환)
        temp_output_path = os.path.join(output_dir, f"temp_{audio_file}")
        convert_audio_safely(input_path, temp_output_path, target_sr=sr)

        # Step 2: 변환된 오디오 파일을 불러와 무음 제거 및 분할
        audio, sr = librosa.load(temp_output_path, sr=sr)
        audio = remove_silence(audio, sr)  # 무음 제거
        segments = split_audio_with_overlap(audio, sr)  # 30초 단위 분할
        
        # Step 3: 조각별로 저장
        base_name = os.path.splitext(audio_file)[0]
        for idx, segment in enumerate(segments):
            output_path = os.path.join(output_dir, f"{base_name}_seg{idx+1}.wav")
            sf.write(output_path, segment, sr)
            print(f"Saved: {output_path}")
        
        # 임시 변환 파일 삭제
        os.remove(temp_output_path)


input_audio_path = "C:/Users/001/Documents/간병인프로젝트/voice/dataset/train/voice/"  # 원본 오디오 폴더
output_audio_path = "C:/Users/001/Documents/간병인프로젝트/voice/dataset/train/preprocessed_voice"  # 저장할 폴더
process_and_save_audio(input_audio_path, output_audio_path)
