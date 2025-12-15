import pandas as pd
import numpy as np

import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from tqdm import tqdm
from transformers import ElectraTokenizer, ElectraForSequenceClassification

# ⭐ 모델 및 데이터 경로 설정
MODEL_SAVE_PATH = "koelectra_hotel_multi_label" # 학습된 모델이 저장된 폴더
# ⭐ 전체 부정 리뷰 파일 경로 (15,186건)
FULL_DATA_PATH = "data/processed/negative_hotel_reviews_3_and_under.csv"
# ⭐ 분류 결과 저장할 최종 파일
OUTPUT_LABELED_PATH = "data/final_result/full_negative_reviews_classified.csv"

# 0. 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("사용하는 장치:", device)

# 1. 모델 및 토크나이저 로드
try:
    print("✅ 학습된 KOELECTRA 모델 로드 시작...")
    tokenizer = ElectraTokenizer.from_pretrained(MODEL_SAVE_PATH)
    # num_labels=4로 저장했기 때문에 자동으로 로드됩니다.
    model = ElectraForSequenceClassification.from_pretrained(MODEL_SAVE_PATH)
    model.to(device)
    model.eval() # 추론 모드로 전환
    print("✅ 모델 로드 및 추론 모드 설정 완료.")
except Exception as e:
    print(f"❌ 오류: 모델 로드 실패. '{MODEL_SAVE_PATH}' 폴더를 확인해주세요. 오류: {e}")
    exit()

# 2. 전체 데이터 로드
try:
    df_full = pd.read_csv(FULL_DATA_PATH)
    reviews = df_full['review'].astype(str).tolist()
    print(f"✅ 전체 데이터 로드 완료. 총 {len(reviews)}건.")
except FileNotFoundError:
    print(f"❌ 오류: 전체 데이터 파일 '{FULL_DATA_PATH}'을(를) 찾을 수 없습니다.")
    exit()

# 3. 데이터 토큰화
print("✅ 데이터 토큰화 시작...")
tokenized_data = tokenizer(reviews, truncation=True, max_length=256, add_special_tokens=True, padding="max_length")

input_ids = tokenized_data["input_ids"]
attention_mask = tokenized_data["attention_mask"]

# 4. DataLoader 설정
batch_size = 32
inputs = torch.tensor(input_ids)
masks = torch.tensor(attention_mask)

# 라벨이 없으므로 inputs와 masks만 데이터셋에 넣습니다.
inference_data = TensorDataset(inputs, masks)
inference_sampler = SequentialSampler(inference_data)
inference_dataloader = DataLoader(inference_data, batch_size=batch_size, sampler=inference_sampler)
print("✅ DataLoader 설정 완료.")

# 5. 추론 (Inference) 수행
print("✅ 전체 데이터에 대한 카테고리 분류(추론) 시작...")
all_preds = []

# tqdm으로 진행 상황을 시각화
for batch in tqdm(inference_dataloader, desc="Classifying Reviews"):
    batch_ids, batch_mask = tuple(t.to(device) for t in batch)

    with torch.no_grad():
        outputs = model(batch_ids, attention_mask=batch_mask)

    # 다중 라벨 예측 로직: Sigmoid를 적용하고 0.5 임계값으로 이진화
    probs = torch.sigmoid(outputs.logits)
    preds = (probs > 0.5).int()
    all_preds.extend(preds.cpu().numpy())

print("✅ 추론 완료.")

# 6. 결과 정리 및 파일 저장
predicted_labels_matrix = np.array(all_preds)
label_cols = ['label_A', 'label_B', 'label_C', 'label_D']

# 데이터프레임에 예측 결과 컬럼 추가
for i, col_name in enumerate(label_cols):
    df_full[col_name] = predicted_labels_matrix[:, i]

# 7. 최종 분석용 파일 저장
df_full.to_csv(OUTPUT_LABELED_PATH, index=False, encoding='utf-8')

print("\n--- 최종 결과 ---")
print(f"총 {len(df_full)}건의 리뷰에 대한 카테고리 분류를 완료했습니다.")
print(f"결과 파일 저장 경로: {OUTPUT_LABELED_PATH}")
print("\n--- 예측 결과 미리보기 ---")
print(df_full[['review', 'label_A', 'label_B', 'label_C', 'label_D']].head())