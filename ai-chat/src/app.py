import gradio as gr
from model import ChatModel
import os
from dotenv import load_dotenv

load_dotenv()

class ChatInterface:
    def __init__(self):
        self.model = ChatModel()
        self.message_history = []

    def respond(self, message, history):
        """Process user message and generate response."""
        # Add user message to history
        self.message_history.append({
            "role": "user",
            "content": message
        })
        
        # Generate response
        response = self.model.chat(self.message_history)
        
        # Add assistant response to history
        self.message_history.append({
            "role": "assistant",
            "content": response
        })
        
        return response

def main():
    # Initialize chat interface
    chat = ChatInterface()
    
    # Create Gradio interface
    iface = gr.ChatInterface(
        chat.respond,
        title="AI Chatbot",
        description="Ask me anything!",
        theme="soft",
        examples=[
            ["안녕하세요!"],
            ["인공지능에 대해 설명해주세요."],
            ["파이썬으로 할 수 있는 것들을 알려주세요."]
        ]
    )
    
    # Launch the interface
    iface.launch(share=True)

if __name__ == "__main__":
    main() 