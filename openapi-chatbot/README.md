# 한국어 챗봇 프로젝트

이 프로젝트는 Hugging Face의 Transformers 라이브러리를 사용하여 만든 간단한 한국어 챗봇입니다.

## 설치 방법

1. 필요한 패키지 설치:
```bash
pip install -r requirements.txt
```

## 사용 방법

1. 모델 학습:
```bash
python train_chatbot.py
```
이 과정은 시간이 걸릴 수 있으며, 학습된 모델은 `chatbot_model` 디렉토리에 저장됩니다.

2. 챗봇과 대화하기:
```bash
python chat.py
```

## 프로젝트 구조

- `requirements.txt`: 필요한 Python 패키지 목록
- `train_chatbot.py`: 모델 학습 스크립트
- `chat.py`: 챗봇 인터페이스
- `chatbot_model/`: 학습된 모델이 저장되는 디렉토리

## 데이터셋

현재 구현된 데이터셋은 간단한 예제 질문-답변 쌍으로 구성되어 있습니다. 실제 사용을 위해서는 더 많은 데이터를 추가하거나 custom 데이터셋으로 교체하시면 됩니다. 