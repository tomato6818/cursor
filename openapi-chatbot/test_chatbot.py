from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def test_chatbot():
    print("=== 챗봇 테스트 시작 ===\n")
    
    # 모델과 토크나이저 로드
    try:
        print("1. 모델 로딩 테스트...")
        model = AutoModelForCausalLM.from_pretrained("./chatbot_model")
        tokenizer = AutoTokenizer.from_pretrained(
            "./chatbot_model",
            bos_token='</s>',
            eos_token='</s>',
            pad_token='<pad>'
        )
        tokenizer.pad_token = tokenizer.eos_token
        print("✓ 모델 로딩 성공\n")
    except Exception as e:
        print(f"✗ 모델 로딩 실패: {str(e)}\n")
        return

    # 테스트할 질문들
    test_questions = [
        "안녕하세요?",
        "인공지능이 무엇인가요?",
        "파이썬은 어떤 언어인가요?",
        "오늘 기분이 어떠신가요?",  # 학습되지 않은 질문
    ]

    print("2. 응답 생성 테스트...")
    for question in test_questions:
        print(f"\n질문: {question}")
        try:
            # 입력 전처리
            prompt = f"Question: {question}\nAnswer:"
            inputs = tokenizer(prompt, return_tensors="pt", max_length=128, truncation=True)
            
            # 응답 생성
            with torch.no_grad():
                outputs = model.generate(
                    inputs.input_ids,
                    max_length=256,
                    num_beams=5,
                    early_stopping=True,
                    pad_token_id=tokenizer.eos_token_id,
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.95,
                    no_repeat_ngram_size=2,
                    eos_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.2
                )
            
            # 응답 디코딩 - 입력 프롬프트 이후의 텍스트만 추출
            response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            print(f"응답: {response.strip()}")
            print("✓ 응답 생성 성공")
        
        except Exception as e:
            print(f"✗ 응답 생성 실패: {str(e)}")

    print("\n=== 테스트 완료 ===")

if __name__ == "__main__":
    test_chatbot() 