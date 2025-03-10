# 라이브러리 import
import argparse
import pickle
import pandas as pd
import numpy as np
import torch
import os
from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn.model_selection import train_test_split
from pytorch_tabnet.callbacks import Callback
from torch.utils.tensorboard import SummaryWriter

# SEED 고정
SEED = 819
np.random.seed(SEED)
torch.manual_seed(SEED)

# Argument 설정
def parse_args():
    parser = argparse.ArgumentParser(description="Train the caregiver-patient matching model.")

    # 모델 선택
    parser.add_argument("--model_type", type=str, default="tabnet", choices=["tabnet", "xgboost", "lightgbm"])

    # 하이퍼파라미터
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--virtual_batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=0.01)

    # 컬럼 가중치
    parser.add_argument("--w_region", type=float, default=3.0)
    parser.add_argument("--w_spot", type=float, default=2.0)
    parser.add_argument("--w_sex", type=float, default=2.0)
    parser.add_argument("--w_walk", type=float, default=2.0)
    parser.add_argument("--w_smoking", type=float, default=2.0)
    parser.add_argument("--w_symp", type=float, default=2.0)
    parser.add_argument("--w_date", type=float, default=2.0)
    parser.add_argument("--w_star0", type=float, default=0.5)
    parser.add_argument("--w_star1", type=float, default=0.5)
    parser.add_argument("--w_star2", type=float, default=0.5)


    # 데이터 경로
    parser.add_argument('--data_path', type=str, default='/mnt/dataset/gpt_1000_train.pkl')

    return parser.parse_args()

args = parse_args()



#  데이터 로드
with open(args.data_path, "rb") as f:
    train_data = pickle.load(f)

#  컬럼 정의
columns = (
    ["patient_id", "caregiver_id"] +
    [f"region_x_{i}" for i in range(19)] +
    [f"region_y_{i}" for i in range(19)] +
    ["spot_x", "spot_y", "sex_x", "sex_y", "prefersex_x", "prefersex_y",
     "canwalk_x", "canwalk_y", "smoking_x", "smoking_y",
     "region_match", "spot_match", "gender_match", "canwalk_match", "smoking_match",
     "symptom_match_score", "date_overlap", 'preferstar', "star_0", 'star_1', 'star_2']
)

# DataFrame 변환
train_df = pd.DataFrame(train_data, columns=columns)

X_train = train_df.drop(columns=["patient_id", "caregiver_id"]).values
X_train = X_train.astype(np.float32)

# y_train 가중치 조정 함수
def adjust_star_weights(row, w_star0, w_star1, w_star2):
    if row["preferstar"] == 0:
        return w_star0 * 1.0, w_star1 * 0.8, w_star2 * 0.8  # 성실성 선호 증가
    elif row["preferstar"] == 1:
        return w_star0 * 0.8, w_star1 * 1.0, w_star2 * 0.8  # 의사소통 선호 증가
    elif row["preferstar"] == 2:
        return w_star0 * 0.8, w_star1 * 0.8, w_star2 * 1.0  # 위생관념 선호 증가
    return w_star0, w_star1, w_star2  # 기본 가중치 유지

# 각 행의 preferstar에 따라 가중치 적용
adjusted_weights = train_df.apply(lambda row: adjust_star_weights(row, args.w_star0, args.w_star1, args.w_star2), axis=1)

# DataFrame으로 변환하여 개별 가중치 가져오기
train_df["adjusted_w_star0"] = adjusted_weights.apply(lambda x: x[0])
train_df["adjusted_w_star1"] = adjusted_weights.apply(lambda x: x[1])
train_df["adjusted_w_star2"] = adjusted_weights.apply(lambda x: x[2])


# y_train 생성 (조정된 가중치 적용)
y_train = (
    train_df["region_match"] * args.w_region +
    train_df["spot_match"] * args.w_spot +
    train_df["gender_match"] * args.w_sex +
    train_df["canwalk_match"] * args.w_walk +
    train_df["smoking_match"] * args.w_smoking +
    train_df["symptom_match_score"] * args.w_symp +
    train_df["date_overlap"] * args.w_date +
    train_df["star_0"] * train_df["adjusted_w_star0"] +
    train_df["star_1"] * train_df["adjusted_w_star1"] +
    train_df["star_2"] * train_df["adjusted_w_star2"]
).values.reshape(-1, 1)  #  (n_samples, 1) 형태로 변환

WEIGHT_SUM = args.w_region + args.w_spot + args.w_sex + args.w_walk + args.w_smoking + args.w_symp + args.w_date + train_df["adjusted_w_star0"] + train_df["adjusted_w_star1"] + train_df["adjusted_w_star2"]


# Train / Validation Split
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=SEED)
y_train = y_train.astype(np.float32)
y_val = y_val.astype(np.float32)



# TabNet 모델 초기화 
model = TabNetRegressor(
    optimizer_fn=torch.optim.Adam,
    optimizer_params={'lr': args.learning_rate},
    scheduler_params={"step_size": 10, "gamma": 0.9},
    scheduler_fn=torch.optim.lr_scheduler.StepLR,
    mask_type="entmax",
    verbose=1,
    device_name="cuda",  #  GPU 사용
    n_d=64, n_a=64,  #  모델 차원 (기본값 64)
    n_steps=5, gamma=1.5,  #  학습 안정화 파라미터
    n_shared=2, n_independent=2,  #  공유된/독립된 인코더 스텝 수
    momentum=0.02  #  학습 안정화를 위한 모멘텀
)


# TensorBoard Logger 콜백 정의
# Mean Absolute Error, Root Mean Squared Error --> 값이 작을수록 학습 잘됨
class TensorBoardLogger(Callback):
    def __init__(self, writer):
        self.writer = writer

    def on_epoch_end(self, epoch, logs=None):
        if logs:
            self.writer.add_scalar("Loss/Train", logs["loss"], epoch)
            self.writer.add_scalar("Metrics/MAE", logs["validation_mae"], epoch)
            self.writer.add_scalar("Metrics/RMSE", logs["validation_rmse"], epoch)
            self.writer.flush()  #  즉시 반영
            

# TensorBoard 설정
log_dir = "logs/"
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir=log_dir)

# Callback 객체 생성
tb_callback = TensorBoardLogger(writer)


# 모델 학습 (최적 모델 자동 저장 포함)
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric=["rmse", "mae"],
    max_epochs=args.epochs,
    patience=10,
    batch_size=args.batch_size,
    virtual_batch_size=args.virtual_batch_size,
    eval_name=["validation"],
    num_workers=64,
    callbacks=[tb_callback]
)

model.save_model("/mnt/tabnet_model.zip")

print(f" 학습 완료! 가장 낮은 RMSE 모델 저장됨: best_tabnet_model.zip")


# TensorBoard Writer 닫기
writer.close()


