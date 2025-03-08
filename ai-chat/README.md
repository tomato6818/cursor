# NLP Chatbot Project

This project implements a chatbot using transformer-based language models. It supports both fine-tuning existing models and creating a chat interface.

## Features
- Transformer-based language model integration
- Fine-tuning capabilities
- Web-based chat interface using Gradio
- Support for multiple model backends

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the chat interface:
```bash
python src/app.py
```

## Project Structure
- `src/model.py`: Model loading and inference
- `src/train.py`: Fine-tuning functionality
- `src/app.py`: Gradio web interface
- `src/utils.py`: Utility functions

## Configuration
Create a `.env` file in the root directory with your configuration:
```
MODEL_NAME=facebook/opt-350m  # or other model of your choice
``` 