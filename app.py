"""
신종 게임 은어 및 악플 필터링 텍스트 분류기 - Gradio 웹 UI

학습된 로컬 모델(./my-custom-model)을 pipeline으로 불러와
게임 채팅 텍스트의 정상/악플 여부와 확률(Score)을 보여줍니다.
"""

from __future__ import annotations

import gradio as gr
from transformers import pipeline

MODEL_PATH = "./my-custom-model"

# ---------------------------------------------------------------------------
# 4. Gradio 기반 웹 서비스 배포
# ---------------------------------------------------------------------------
print(f"모델 로딩 중: {MODEL_PATH}")
classifier = pipeline(
    "text-classification",
    model=MODEL_PATH,
    tokenizer=MODEL_PATH,
    top_k=None,  # 모든 라벨 점수 반환 (구버전: return_all_scores=True)
)
print("모델 로딩 완료.")


def predict(text: str) -> str:
    """입력 채팅 문장에 대해 정상/악플 판정과 확률을 반환합니다."""
    if not text or not text.strip():
        return "텍스트를 입력해 주세요."

    results = classifier(text)[0]  # [{'label': '정상', 'score': ...}, ...]
    # 점수가 높은 순으로 정렬
    results = sorted(results, key=lambda x: x["score"], reverse=True)
    top = results[0]

    lines = [
        f"판정: {top['label']}",
        f"Score: {top['score']:.4f}",
        "",
        "전체 점수:",
    ]
    for item in results:
        lines.append(f"  - {item['label']}: {item['score']:.4f}")
    return "\n".join(lines)


demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(
        lines=3,
        label="게임 채팅 입력",
        placeholder="예: 오늘 한 판 더 할까? / 너 진짜 쓰레기네",
    ),
    outputs=gr.Textbox(label="분류 결과", lines=8),
    title="게임 채팅 악플 필터링 분류기",
    description=(
        "한국어 게임 채팅/댓글 문장을 입력하면 "
        "정상 또는 악플 여부와 확률(Score)을 출력합니다. "
        "(기반 데이터: smilegate-ai/kor_unsmile, 모델: beomi/kcbert-base fine-tuned)"
    ),
    examples=[
        ["오늘 저녁에 랭크 한 판 어때?"],
        ["너 때문에 졌잖아 진짜 못하네"],
        ["ㅋㅋㅋ 개못하네 꺼져"],
        ["팀원들 수고했어 다음에도 같이 하자"],
    ],
)

if __name__ == "__main__":
    # share=True 로 설정하면 임시 공개 링크가 생성됩니다 (데모/발표용).
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
