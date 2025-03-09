from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def load_model_and_tokenizer():
    model = AutoModelForCausalLM.from_pretrained("./chatbot_model")
    tokenizer = AutoTokenizer.from_pretrained("./chatbot_model")
    return model, tokenizer

def generate_response(question, model, tokenizer):
    prompt = f"Question: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt", max_length=128, truncation=True)
    
    outputs = model.generate(
        inputs.input_ids,
        max_length=128,
        num_beams=5,
        early_stopping=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response.strip()

def main():
    print("모델을 로딩중입니다...")
    model, tokenizer = load_model_and_tokenizer()
    print("챗봇이 준비되었습니다! 대화를 시작하세요 (종료하려면 'quit' 입력)")
    
    while True:
        user_input = input("\n사용자: ")
        if user_input.lower() == 'quit':
            break
            
        response = generate_response(user_input, model, tokenizer)
        print(f"챗봇: {response}")

if __name__ == "__main__":
    main() 