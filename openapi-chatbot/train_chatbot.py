import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
import torch

# 간단한 데이터셋 생성
def create_dataset():
    data = {
        'question': [
            '안녕하세요?',
            '오늘 날씨는 어때요?',
            '인공지능이 무엇인가요?',
            '파이썬은 어떤 프로그래밍 언어인가요?',
            '머신러닝과 딥러닝의 차이는 무엇인가요?'
        ],
        'answer': [
            '안녕하세요! 무엇을 도와드릴까요?',
            '죄송하지만 실시간 날씨 정보는 제공할 수 없습니다. 날씨 앱이나 웹사이트를 확인해보세요.',
            '인공지능은 인간의 학습능력, 추론능력, 지각능력을 컴퓨터로 구현하는 기술입니다.',
            '파이썬은 읽기 쉽고 간단한 문법을 가진 고급 프로그래밍 언어입니다. 데이터 과학과 인공지능 분야에서 널리 사용됩니다.',
            '머신러닝은 데이터로부터 패턴을 학습하는 AI의 한 분야이고, 딥러닝은 머신러닝의 한 종류로 신경망을 사용하는 방식입니다.'
        ]
    }
    
    # 데이터셋 확장 - 각 대화에 대해 문맥을 더 잘 학습하도록 함
    expanded_data = {
        'text': []
    }
    
    for q, a in zip(data['question'], data['answer']):
        # 각 대화를 여러 번 반복하여 학습 데이터 증강
        for _ in range(3):  # 각 대화를 3번씩 반복
            expanded_data['text'].append(f"Question: {q}\nAnswer: {a}")
    
    return Dataset.from_dict(expanded_data)

def main():
    # 한국어 모델과 토크나이저 로드
    model_name = "skt/kogpt2-base-v2"  # 한국어 GPT-2 모델
    
    # 토크나이저 설정
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        bos_token='</s>',
        eos_token='</s>',
        unk_token='<unk>',
        pad_token='<pad>',
        mask_token='<mask>'
    )
    
    # 모델 로드
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.resize_token_embeddings(len(tokenizer))
    
    # 데이터셋 생성
    dataset = create_dataset()
    
    # 데이터 전처리 함수
    def preprocess_function(examples):
        # 토크나이징
        model_inputs = tokenizer(
            examples['text'],
            max_length=128,
            truncation=True,
            padding='max_length',
            return_tensors=None  # 배치 처리를 위해 None으로 설정
        )
        
        # 레이블 설정
        model_inputs["labels"] = model_inputs["input_ids"].copy()
        
        return model_inputs

    # 데이터셋 전처리
    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset.column_names
    )

    # 학습 파라미터 설정
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=10,
        per_device_train_batch_size=2,
        warmup_steps=100,
        save_steps=100,
        logging_dir='./logs',
        learning_rate=1e-5,
        weight_decay=0.01,
        gradient_accumulation_steps=4,
        save_total_limit=2,
        logging_steps=10,
        evaluation_strategy="no",  # 평가 데이터가 없으므로 평가 비활성화
        save_strategy="steps",
        load_best_model_at_end=False
    )

    # 트레이너 초기화 및 학습
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
    )

    # 모델 학습
    trainer.train()

    # 모델 저장
    model.save_pretrained("./chatbot_model")
    tokenizer.save_pretrained("./chatbot_model")

if __name__ == "__main__":
    main() 