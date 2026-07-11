"""
신종 게임 은어 및 악플 필터링 텍스트 분류기 - 학습 파이프라인

1) 데이터 로드 & 전처리 (kor_unsmile → 정상/악플 이진 분류)
2) kcBERT 미세조정 (Fine-Tuning)
3) 로컬 저장 + (선택) Hugging Face Hub 업로드
"""

from __future__ import annotations

import numpy as np
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

# ---------------------------------------------------------------------------
# 설정
# ---------------------------------------------------------------------------
DATASET_NAME = "smilegate-ai/kor_unsmile"
MODEL_NAME = "beomi/kcbert-base"  # 한국어 댓글/채팅에 적합한 BERT
OUTPUT_DIR = "./my-custom-model"
MAX_LENGTH = 128
NUM_LABELS = 2
LABEL2ID = {"정상": 0, "악플": 1}
ID2LABEL = {0: "정상", 1: "악플"}


# ---------------------------------------------------------------------------
# 1. 데이터 수집 및 전처리
# ---------------------------------------------------------------------------
def to_binary_label(example: dict) -> dict:
    """
    kor_unsmile은 다중 라벨(혐오 카테고리 + clean)입니다.
    챌린지용으로 단순화:
      - clean == 1  → 정상(0)
      - 그 외       → 악플(1)  (혐오/욕설/악플 포함)
    """
    example["label"] = 0 if example["clean"] == 1 else 1
    return example


def load_and_prepare_dataset():
    """Hugging Face datasets로 한국어 악플/혐오 데이터셋을 로드하고 이진 라벨로 변환합니다."""
    raw = load_dataset(DATASET_NAME)

    # train / valid 스플릿 사용 (kor_unsmile 기본 구성)
    dataset = raw.map(to_binary_label)
    dataset = dataset.rename_column("문장", "text")
    dataset = dataset.select_columns(["text", "label"])

    print("=== 데이터셋 미리보기 ===")
    print(dataset)
    print(dataset["train"][0])
    return dataset


def tokenize_function(examples, tokenizer):
    """padding/truncation을 적용해 텍스트를 모델 입력으로 변환합니다."""
    return tokenizer(
        examples["text"],
        truncation=True,
        padding=False,  # 배치 단위 패딩은 DataCollator가 처리 (더 효율적)
        max_length=MAX_LENGTH,
    )


def preprocess_dataset(dataset, tokenizer):
    tokenized = dataset.map(
        lambda batch: tokenize_function(batch, tokenizer),
        batched=True,
        remove_columns=["text"],
    )
    return tokenized


# ---------------------------------------------------------------------------
# 2. 모델 미세조정
# ---------------------------------------------------------------------------
def compute_metrics(eval_pred):
    """검증 세트에 대한 Accuracy / F1을 계산합니다."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="binary"),
    }


def train():
    # --- 데이터 ---
    dataset = load_and_prepare_dataset()

    # --- 토크나이저 ---
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenized = preprocess_dataset(dataset, tokenizer)

    # --- 분류 모델 (정상 / 악플) ---
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    # --- 학습 설정 (노트북/로컬 CPU·GPU 모두 가볍게 돌릴 수 있는 값) ---
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=2e-5,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=50,
        report_to="none",  # wandb 등 외부 로깅 비활성화
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["valid"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("=== 학습 시작 ===")
    trainer.train()

    print("=== 검증 세트 평가 ===")
    metrics = trainer.evaluate()
    print(metrics)

    # -----------------------------------------------------------------------
    # 3. 모델 평가 및 로컬 저장 (+ Hub 업로드 스니펫)
    # -----------------------------------------------------------------------
    print(f"=== 모델 저장: {OUTPUT_DIR} ===")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("로컬 저장 완료. 이제 `python app.py`로 Gradio UI를 실행할 수 있습니다.")

    # --- (선택) Hugging Face Hub 업로드 ---
    # 1) 터미널에서 `huggingface-cli login` 실행 후 토큰 입력
    # 2) 아래 주석을 해제하고 YOUR_HF_USERNAME을 본인 계정명으로 바꾸세요.
    #
    # repo_id = "YOUR_HF_USERNAME/game-chat-abuse-filter"
    # trainer.push_to_hub(repo_id)
    # # 또는:
    # # from huggingface_hub import HfApi
    # # api = HfApi()
    # # api.upload_folder(folder_path=OUTPUT_DIR, repo_id=repo_id, repo_type="model")


if __name__ == "__main__":
    train()
