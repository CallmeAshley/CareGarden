import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
import pickle

#  전체 랜덤 시드 고정 (재현 가능성 보장)
SEED = 819
random.seed(SEED)
np.random.seed(SEED)  # numpy 관련 연산이 있다면 추가

# 파일 경로 설정
PATIENTS_FILE = "C:/Users/001/Documents/간병인프로젝트/matching/dataset/gpt_1000_patients_dataset.xlsx"
CAREGIVERS_FILE = "C:/Users/001/Documents/간병인프로젝트/matching/dataset/gpt_1000_caregivers_dataset.xlsx"

#  1. 데이터 불러오기
def load_data():
    """ 엑셀 파일에서 환자와 간병인 데이터를 로드하는 함수 """
    patients_df = pd.read_excel(PATIENTS_FILE)
    caregivers_df = pd.read_excel(CAREGIVERS_FILE)
    return patients_df, caregivers_df


def generate_matching_data(patients_df, caregivers_df):
    """ 환자-간병인 매칭 벡터를 생성 (One-hot Encoding + Boolean 0/1 변환) """

    #  1) 환자와 간병인의 모든 조합 생성 (merge 사용)
    patients_df["key"] = 1
    caregivers_df["key"] = 1
    merged_df = pd.merge(patients_df, caregivers_df, on="key").drop(columns=["key"])  

    #  3) 지역 매칭 (One-hot Encoding 적용)
    for i in range(19):  # region_0 ~ region_18
        merged_df[f"region_x_{i}"] = (merged_df["region_x"] == i).astype(int)
        merged_df[f"region_y_{i}"] = merged_df["region_y"].apply(lambda x: 1 if str(i) in x.split(",") else 0)

    #  4) 지역 매칭 여부
    merged_df["region_match"] = merged_df.apply(lambda row: 1 if row["region_x"] in map(int, row["region_y"].split(",")) else 0, axis=1)

    #  5) Boolean Feature → 0/1 변환
    merged_df["spot_match"] = (merged_df["spot_x"] == merged_df["spot_y"]) | (merged_df["spot_y"] == 2)
    merged_df["gender_match"] = (merged_df["prefersex_x"].isin([merged_df["sex_y"], 2])) & \
                                (merged_df["prefersex_y"].isin([merged_df["sex_x"], 2]))
    merged_df["canwalk_match"] = (merged_df["canwalk_x"] == merged_df["canwalk_y"]) | (merged_df["canwalk_y"] == 2)
    merged_df["smoking_match"] = (merged_df["smoking_x"] == 2) | (merged_df["smoking_x"] == merged_df["smoking_y"])
    merged_df["date_overlap"] = (pd.to_datetime(merged_df["startdate_x"]) <= pd.to_datetime(merged_df["enddate_y"])) & \
                                (pd.to_datetime(merged_df["startdate_y"]) <= pd.to_datetime(merged_df["enddate_x"]))

    #  Boolean 데이터를 0 또는 1로 변환
    for col in ["region_match", "spot_match", "gender_match", "canwalk_match", "smoking_match", "date_overlap"]:
        merged_df[col] = merged_df[col].astype(int)

    #  6) 증상 매칭 점수 (교집합 / 환자의 증상 개수)
    def compute_symptom_score(row):
        patient_symptoms = set(map(int, row["symptoms_x"].split(",")))
        caregiver_symptoms = set(map(int, row["symptoms_y"].split(",")))
        return len(patient_symptoms & caregiver_symptoms) / len(patient_symptoms)

    merged_df["symptom_match_score"] = merged_df.apply(compute_symptom_score, axis=1)

    #  7) 최종 Feature 선택 (환자 및 간병인 ID 포함)
    feature_cols = (
        ["patient_id", "caregiver_id"] +  #  환자 및 간병인 ID 추가
        [f"region_x_{i}" for i in range(19)] +  # 환자 지역 One-hot Encoding
        [f"region_y_{i}" for i in range(19)] +  # 간병인 활동 가능 지역 One-hot Encoding
        ["spot_x", "spot_y", "sex_x", "sex_y", "prefersex_x", "prefersex_y",  # 기존 정보
         "canwalk_x", "canwalk_y", "smoking_x", "smoking_y",
         "region_match", "spot_match", "gender_match", "canwalk_match", "smoking_match",
         "symptom_match_score", "date_overlap",'preferstar', "star_0", 'star_1', 'star_2']  # 추가된 Matching 정보들
    )

    #  최종 데이터 변환 (numpy array)
    final_data = merged_df[feature_cols].to_numpy()

    return final_data




#  4. 데이터셋 저장
def save_dataset(train_data, test_data):
    """ 데이터를 Pickle 형식으로 저장하여 학습에 사용할 수 있도록 함 """
    with open("/mnt/dataset/train.pkl", "wb") as f:
        pickle.dump(train_data, f)
    with open("/mnt/dataset/test.pkl", "wb") as f:
        pickle.dump(test_data, f)

#  4. 메인 실행 함수
def main():
    print("데이터 로딩 중")
    patients_df, caregivers_df = load_data()

    print("데이터 전처리 중")
    matching_data = generate_matching_data(patients_df, caregivers_df)

    print("데이터셋 분할 (Train/Test)")
    
    train_data, test_data = train_test_split(matching_data, test_size=0.2, random_state=819)

    print("데이터 저장 중")
    save_dataset(train_data, test_data)

    print("데이터 로딩 및 전처리 완료! (train_tabnet.pkl / test_tabnet.pkl 저장됨)")

if __name__ == "__main__":
    main()