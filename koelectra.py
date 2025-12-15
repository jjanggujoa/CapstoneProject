import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
# ⭐ 모델 임포트는 그대로 유지합니다.
from transformers import ElectraTokenizer, ElectraForSequenceClassification
from transformers import get_linear_schedule_with_warmup, logging

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score

# ⭐ 정확한 모델 이름 정의 (수정된 부분)
MODEL_NAME = "monologg/koelectra-base-v3-discriminator"

# 0. 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("사용하는 장치:", device)

# 1. 학습 시 경고 메세지 제거
logging.set_verbosity_error()

# 2. 데이터 로드 및 확인
train_data_path = "data/labeled/review_sample_2500_encoded.csv"
try:
    dataset = pd.read_csv(train_data_path, sep=",")
except FileNotFoundError:
    print(f"❌ 오류: 데이터 파일 '{train_data_path}'을(를) 찾을 수 없습니다. JSON -> CSV 변환 단계를 먼저 완료했는지 확인해주세요.")
    exit()

# 리뷰 텍스트와 4개의 라벨 컬럼 추출
text = list(dataset['review'].values)

label_cols = ['label_A', 'label_B', 'label_C', 'label_D']
labels_matrix = dataset[label_cols].values
num_labels = len(label_cols)  # 라벨 개수: 4 (A, B, C, D)

print(f"### 데이터 확인 (다중 라벨) ###")
print(f"총 학습 데이터의 수: {len(text)}건")
print(f"라벨 개수 (A, B, C, D): {num_labels}개")

# 3. 텍스트 토큰화
# ⭐ 모델 이름 수정 적용 (monologg/koelectra-base-v3-discriminator)
tokenizer = ElectraTokenizer.from_pretrained(MODEL_NAME)
input = tokenizer(text, truncation=True, max_length=256, add_special_tokens=True, padding="max_length")

input_ids = input["input_ids"]
attention_mask = input["attention_mask"]

# 4. 데이터 분리 (학습 / 검증)
train_ids, validation_ids, train_y, validation_y = train_test_split(
    input_ids, labels_matrix, test_size=0.2, random_state=2025
)
train_masks, validation_masks, _, _ = train_test_split(
    attention_mask, labels_matrix, test_size=0.2, random_state=2025
)

# 5. DataLoader 설정
batch_size = 32

train_inputs = torch.tensor(train_ids)
train_labels = torch.tensor(train_y, dtype=torch.float)  # 다중 라벨을 위해 float 타입 사용
train_masks = torch.tensor(train_masks)

train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, batch_size=batch_size, sampler=train_sampler)

validation_inputs = torch.tensor(validation_ids)
validation_labels = torch.tensor(validation_y, dtype=torch.float)
validation_masks = torch.tensor(validation_masks)

validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(validation_data, batch_size=batch_size, sampler=validation_sampler)

# 6. 모델, 옵티마이저, 스케줄러 설정
# ⭐ 모델 이름 수정 적용 (monologg/koelectra-base-v3-discriminator)
model = ElectraForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_labels  # num_labels = 4
)
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-04, eps=1e-06, betas=(0.9, 0.999))

epoch = 4
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,
                                            num_training_steps=len(train_dataloader) * epoch)

epoch_result = []

# ==============================================================================
# 7. 모델 학습 루프
# ==============================================================================
for e in range(0, epoch):
    print(f"\n======== Epoch {e + 1}/{epoch} 시작 ========")
    model.train()
    total_train_loss = 0.0
    progress_bar = tqdm(train_dataloader, desc=f"Training Epoch {e + 1}", leave=False)

    for batch in progress_bar:
        batch_ids, batch_mask, batch_label = tuple(t.to(device) for t in batch)

        model.zero_grad()
        outputs = model(batch_ids, attention_mask=batch_mask, labels=batch_label)
        loss = outputs.loss

        total_train_loss += loss.item()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        progress_bar.set_postfix({'loss': loss.item()})

    avg_train_loss = total_train_loss / len(train_dataloader)

    # ==============================================================================
    # 8. 검증 및 성능 측정 (Validation)
    # ==============================================================================
    model.eval()
    val_preds = []
    val_true = []

    for batch in tqdm(validation_dataloader, desc=f"Evaluating Validation Epoch {e + 1}", leave=False):
        batch_ids, batch_mask, batch_label = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            outputs = model(batch_ids, attention_mask=batch_mask)

        # 다중 라벨 예측: Sigmoid 적용 후 0.5 임계값으로 이진화
        probs = torch.sigmoid(outputs.logits)
        preds = (probs > 0.5).int()

        val_preds.extend(preds.cpu().numpy())
        val_true.extend(batch_label.cpu().numpy())

    # 성능 지표 계산
    val_true_arr = np.array(val_true)
    val_preds_arr = np.array(val_preds)

    val_f1_micro = f1_score(val_true_arr, val_preds_arr, average='micro')
    validation_accuracy = accuracy_score(val_true_arr, val_preds_arr)

    epoch_result.append((avg_train_loss, validation_accuracy, val_f1_micro))

print("\n\n ===== 최종 학습 결과 요약 ===== ")
print(" (F1-Micro Score를 주요 성능 지표로 사용하세요.)")
for idx, (loss, val_acc, val_f1) in enumerate(epoch_result):
    print(
        f"Epoch: {idx + 1}, Train Loss: {loss:.4f}, Validation Acc: {val_acc:.4f}, Validation F1(micro): {val_f1:.4f}")

# ==============================================================================
# 9. 모델 저장 (다음 분석 단계에 사용)
# ==============================================================================
print("\n ===== 모델 저장 =====")
save_path = "koelectra_hotel_multi_label"
model.cpu()
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print(f"✅ 모델 저장 완료: {save_path} 폴더에 저장됨.")