from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
import torch
import os
from dotenv import load_dotenv

load_dotenv()

def prepare_dataset(
    dataset_name: str,
    tokenizer,
    max_length: int = 512,
    text_column: str = "context"
):
    """Prepare dataset for training."""
    def tokenize_function(examples):
        return tokenizer(
            examples[text_column],
            truncation=True,
            max_length=max_length,
            padding="max_length"
        )

    # Load dataset
    dataset = load_dataset(dataset_name)
    
    # Tokenize dataset
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset["train"].column_names
    )
    
    return tokenized_dataset

def train_model(
    model_name: str = None,
    dataset_name: str = "databricks/databricks-dolly-15k",
    output_dir: str = "fine_tuned_model",
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 8,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 2e-5,
):
    """Fine-tune the model on a dataset."""
    # Load model and tokenizer
    model_name = model_name or os.getenv("MODEL_NAME", "facebook/opt-350m")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )

    # Prepare dataset
    dataset = prepare_dataset(dataset_name, tokenizer)
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        fp16=torch.cuda.is_available(),
        logging_steps=100,
        save_strategy="epoch",
        evaluation_strategy="epoch",
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"] if "validation" in dataset else None,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    # Train model
    trainer.train()
    
    # Save model
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    train_model() 