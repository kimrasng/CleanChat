# 신종 게임 은어 및 악플 필터링 텍스트 분류기

고등학교 SW 챌린지용 한국어 악플/혐오 표현 이진 분류 프로젝트입니다.

- **데이터**: [`smilegate-ai/kor_unsmile`](https://huggingface.co/datasets/smilegate-ai/kor_unsmile)
- **베이스 모델**: [`beomi/kcbert-base`](https://huggingface.co/beomi/kcbert-base)
- **라벨**: `정상(0)` / `악플(1)` (`clean==1`이면 정상, 그 외 악플)
- **UI**: Gradio

## 프로젝트 구조

```
huggingface/
├── requirements.txt   # 의존성
├── train.py           # 데이터 전처리 + 미세조정 + 모델 저장
├── app.py             # Gradio 웹 UI
└── README.md          # 실행 가이드
```

## 사전 준비

- Python **3.10 이상**
- (권장) GPU가 있으면 학습이 훨씬 빠릅니다. CPU만으로도 동작하지만 시간이 오래 걸립니다.

---

## Step-by-Step 실행 가이드

### 1) 가상환경 만들기 (권장)

```bash
cd /Users/dohyunkim/dev/ebs/huggingface

python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 2) 라이브러리 설치

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> Apple Silicon(M1/M2/M3) Mac이면 PyTorch가 자동으로 MPS를 사용할 수 있습니다.  
> CUDA GPU PC라면 [PyTorch 공식 설치 가이드](https://pytorch.org/get-started/locally/)에 맞춰 `torch`를 먼저 설치해도 됩니다.

### 3) (선택) Hugging Face 로그인

모델/데이터셋 다운로드는 대부분 로그인 없이 가능합니다.  
**Hub에 모델을 업로드**하려면 로그인이 필요합니다.

1. [Hugging Face](https://huggingface.co/) 계정 생성
2. [Access Tokens](https://huggingface.co/settings/tokens)에서 **Write** 권한 토큰 발급
3. 터미널에서 로그인:

```bash
huggingface-cli login
```

토큰을 붙여넣고 Enter.  
또는 환경변수로:

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxx
```

### 4) 모델 학습 (Fine-Tuning)

```bash
python train.py
```

학습이 끝나면 `./my-custom-model/` 폴더에 모델과 토크나이저가 저장됩니다.

**예상 소요 시간 (대략)**

| 환경 | 예상 시간 |
|------|-----------|
| CUDA GPU | 수십 분 |
| Apple MPS | 1~수 시간 |
| CPU only | 수 시간 이상 |

빠르게 파이프라인만 확인하려면 `train.py`의 `num_train_epochs=3`을 `1`로,  
또는 `tokenized["train"]` 대신 `tokenized["train"].select(range(1000))`처럼 샘플을 줄여보세요.

### 5) Gradio 웹 UI 실행

```bash
python app.py
```

브라우저에서 `http://127.0.0.1:7860` 접속 후 게임 채팅 문장을 입력해 보세요.

발표용 공개 링크가 필요하면 `app.py`에서 `share=False`를 `share=True`로 바꾸면 됩니다.

### 6) (선택) Hugging Face Hub에 모델 업로드

`train.py` 하단 주석을 해제하고 `YOUR_HF_USERNAME`을 본인 계정명으로 바꾼 뒤 다시 학습하거나,  
이미 저장된 모델만 올릴 때는 아래를 사용하세요.

```python
from huggingface_hub import HfApi

api = HfApi()
api.upload_folder(
    folder_path="./my-custom-model",
    repo_id="YOUR_HF_USERNAME/game-chat-abuse-filter",
    repo_type="model",
)
```

---

## 파이프라인 요약

1. **Data**: `kor_unsmile` 로드 → `clean` 기준 이진 라벨화 → `AutoTokenizer`로 토큰화  
2. **Train**: `AutoModelForSequenceClassification`(라벨 2개) + `Trainer` (epochs=3, batch=8)  
3. **Save**: `./my-custom-model` 저장 (+ 선택적 `push_to_hub`)  
4. **Deploy**: `pipeline("text-classification")` + `gr.Interface`

## 주의사항

- 학습 데이터는 일반 혐오/악플 표현 중심이라, **최신 게임 은어**는 추가 데이터로 보강하면 성능이 좋아집니다.
- 필터링 결과는 보조 도구로 쓰고, 실제 서비스에서는 오탐/미탐을 반드시 검토하세요.
- `kor_unsmile` 데이터셋 라이선스와 사용 조건을 챌린지 제출 전에 확인하세요.
