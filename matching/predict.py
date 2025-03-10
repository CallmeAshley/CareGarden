# src/predict.py

import pandas as pd
import numpy as np
from pytorch_tabnet.tab_model import TabNetRegressor
from database import get_db
import models
import os

# 매핑 모듈 임포트
from mapping import (
    map_region, map_sex, map_spot, map_canwalk, map_symtoms, 
    map_prefersex, map_smoking, map_pwalk,
    check_spot_match, check_gender_match, check_canwalk_match, check_smoking_match,
)

from train import WEIGHT_SUM

def load_test_data_from_db():
    """
    DB에서 환자와 간병인 데이터를 로드한 후, 크로스 조인을 통해 테스트 데이터셋(매칭 피처 벡터)을 생성합니다.
    """
    db = next(get_db())
    # 환자, 간병인 데이터를 DataFrame으로 로드
    patients_df = pd.read_sql(db.query(models.PatientUserInfo).statement, db.bind)
    caregivers_df = pd.read_sql(db.query(models.CaregiverUserInfo).statement, db.bind)
    
    # 크로스 조인을 위한 key 추가
    patients_df["key"] = 1
    caregivers_df["key"] = 1
    merged_df = pd.merge(patients_df, caregivers_df, on="key", suffixes=('_x', '_y')).drop(columns=["key"])
    
    # ID 재설정 (각각 patient_id, caregiver_id)
    merged_df.rename(columns={'id_x': 'patient_id', 'id_y': 'caregiver_id'}, inplace=True)
    
    # caregiver_df 전처리
    merged_df['region_x'] = merged_df['region_x'].apply(map_region)
    merged_df['sex_x'] = merged_df['sex_x'].apply(map_sex)
    merged_df['spot_x'] = merged_df['spot_x'].apply(map_spot)
    merged_df['canwalk_x'] = merged_df['canwalk_x'].apply(map_canwalk)
    merged_df['symptoms_x'] = merged_df['symptoms_x'].apply(map_symtoms)
    merged_df['prefersex_x'] = merged_df['prefersex_x'].apply(map_prefersex)
    merged_df['smoking_x'] = merged_df['smoking_x'].apply(map_smoking)

    # patient_df 전처리
    merged_df['region_y'] = merged_df['region_y'].apply(map_region)
    merged_df['spot_y'] = merged_df['spot_y'].apply(map_spot)
    merged_df['sex_y'] = merged_df['sex_y'].apply(map_sex)
    merged_df['symptoms_y'] = merged_df['symptoms_y'].apply(map_symtoms)
    merged_df['canwalk_y'] = merged_df['canwalk_y'].apply(map_pwalk)
    merged_df['prefersex_y'] = merged_df['prefersex_y'].apply(map_prefersex)
    merged_df['smoking_y'] = merged_df['smoking_y'].apply(map_smoking)
    
    # 지역 매칭 One-hot Encoding (예시)
    for i in range(19):
        merged_df[f"region_x_{i}"] = merged_df["region_x"].apply(lambda x: 1 if str(i) in x else 0)
        merged_df[f"region_y_{i}"] = merged_df["region_y"].apply(lambda x: 1 if str(i) in x.split(",") else 0)
    
    # 지역 매칭 여부 계산 (비트 OR 연산)
    merged_df['region_match'] = 0
    for i in range(19):
        merged_df['region_match'] = merged_df['region_match'] | (
            (merged_df[f"region_x_{i}"] == 1) & (merged_df[f"region_y_{i}"] == 1)
        )
    
    # Boolean 피처 생성
    merged_df["spot_match"] = merged_df.apply(check_spot_match, axis=1)
    merged_df["gender_match"] = merged_df.apply(check_gender_match, axis=1)
    merged_df["canwalk_match"] = merged_df.apply(check_canwalk_match, axis=1)
    merged_df["smoking_match"] = merged_df.apply(check_smoking_match, axis=1)
    merged_df["date_overlap"] = (pd.to_datetime(merged_df["startdate_x"]) <= pd.to_datetime(merged_df["enddate_y"])) & \
                                (pd.to_datetime(merged_df["startdate_y"]) <= pd.to_datetime(merged_df["enddate_x"]))
    
    # Boolean 데이터를 0/1로 변환
    for col in ["region_match", "spot_match", "gender_match", "canwalk_match", "smoking_match", "date_overlap"]:
        merged_df[col] = merged_df[col].astype(int)
    
    # 증상 매칭 점수 계산
    def compute_symptom_score(row):
        patient_symptoms = set(map(int, row["symptoms_x"].split(",")))
        caregiver_symptoms = set(map(int, row["symptoms_y"].split(",")))
        return len(patient_symptoms & caregiver_symptoms) / len(patient_symptoms)
    
    merged_df["symptom_match_score"] = merged_df.apply(compute_symptom_score, axis=1)
    
    return merged_df

# DB에서 테스트 데이터 로드 및 전처리
test_df = load_test_data_from_db()

# 예측에 필요한 컬럼 순서 (학습 시 사용한 순서와 일치해야 함)
columns = (
    ["patient_id", "caregiver_id"] +
    [f"region_x_{i}" for i in range(19)] +
    [f"region_y_{i}" for i in range(19)] +
    ["spot_x", "spot_y", "sex_x", "sex_y", "prefersex_x", "prefersex_y",
     "canwalk_x", "canwalk_y", "smoking_x", "smoking_y",
     "region_match", "spot_match", "gender_match", "canwalk_match", "smoking_match",
     "symptom_match_score", "date_overlap", 'preferstar', "star_0", 'star_1', 'star_2']
)

# 컬럼 순서 재정렬
test_df = test_df[columns]

# ID 컬럼 분리 (예측 결과 저장용)
id_df = test_df[["patient_id", "caregiver_id", "star_0", "star_1", "star_2"] +
                ["spot_x", "spot_y", "sex_x", "sex_y", "prefersex_x", "prefersex_y",
                 "canwalk_x", "canwalk_y", "smoking_x", "smoking_y"] +
                [f"region_x_{i}" for i in range(19)] +
                [f"region_y_{i}" for i in range(19)]]

# Feature 데이터 (ID 제외)
X_test = test_df.drop(columns=["patient_id", "caregiver_id"]).values.astype(np.float32)

# 모델 로드 (저장된 모델 파일)
best_model = TabNetRegressor()
best_model.load_model("models/tabnet_model.zip")

# 예측 수행
preds = best_model.predict(X_test)
matching_rate = (preds / WEIGHT_SUM) * 100

# 결과 DataFrame 구성
result_df = id_df.copy()
result_df["predicted_score"] = preds
result_df["matching_rate (%)"] = matching_rate

# 결과를 엑셀 파일로 저장
output_path = "predictions.xlsx"
with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
    result_df.to_excel(writer, index=False, sheet_name="Predictions")

print(f"예측 결과가 저장되었습니다: {os.path.abspath(output_path)}")
