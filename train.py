"""
This is a skeleton script to demonstrate how you could fine-tune 
the summarization model later on your own data.
"""

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Trainer, TrainingArguments

def fine_tune_model():
    model_name = "sshleifer/distilbart-cnn-12-6"
    print(f"Loading tokenizer and model for {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    # --- 1. Load your dataset here ---
    # Example format: 
    # dataset = [
    #    {"transcript": "Hello this is a video about...", "summary": "Video about X."},
    #    ...
    # ]
    # You would typically use the Hugging Face `datasets` library here.
    print("Step 1: Load your custom dataset (transcripts and summaries).")
    
    # --- 2. Tokenize your dataset ---
    print("Step 2: Tokenize the data.")
    # def preprocess_function(examples):
    #     inputs = [doc for doc in examples["transcript"]]
    #     model_inputs = tokenizer(inputs, max_length=1024, truncation=True)
    #     labels = tokenizer(examples["summary"], max_length=150, truncation=True)
    #     model_inputs["labels"] = labels["input_ids"]
    #     return model_inputs
    # tokenized_dataset = raw_dataset.map(preprocess_function, batched=True)
    
    # --- 3. Set up Training Arguments ---
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=3,
        predict_with_generate=True,
    )
    
    # --- 4. Initialize Trainer and Train ---
    print("Step 3: Train the model.")
    # trainer = Trainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=tokenized_dataset["train"],
    #     eval_dataset=tokenized_dataset["test"],
    #     tokenizer=tokenizer,
    # )
    # trainer.train()
    
    # --- 5. Save the fine-tuned model ---
    # trainer.save_model("./my-custom-summarizer")
    print("Step 4: Save the model to a local directory.")
    print("After saving, you can update model.py to load your custom model folder!")

if __name__ == "__main__":
    print("--- Model Fine-Tuning Skeleton ---")
    print("Uncomment the code inside `train.py` after preparing your dataset.")
    # fine_tune_model()
