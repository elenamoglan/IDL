import numpy as np
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from datasets import load_dataset
import evaluate


def fine_tune_model():
    model_name = "facebook/bart-base"
    print(f"Loading tokenizer and model for {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    print("Step 1: Load the cnn_dailymail dataset from Hugging Face.")
    raw_dataset = load_dataset("cnn_dailymail", "3.0.0")

    # Select a very small subset to prevent OOM kills during testing
    small_train_dataset = raw_dataset["train"].shuffle(seed=99)
    small_eval_dataset = raw_dataset["validation"].shuffle(seed=42).select(range(5000))

    # --- 2. Tokenize your dataset ---
    print("Step 2: Tokenize the data.")

    def preprocess_function(examples):
        inputs = [doc for doc in examples["article"]]
        model_inputs = tokenizer(inputs, max_length=1024, truncation=True)
        labels = tokenizer(
            text_target=examples["highlights"], max_length=150, truncation=True
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_train = small_train_dataset.map(preprocess_function, batched=True)
    tokenized_eval = small_eval_dataset.map(preprocess_function, batched=True)

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

    # evaluation metric to see improvements
    rouge = evaluate.load("rouge")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred

        labels = np.array(labels, dtype=np.int64)
        predictions = np.array(predictions, dtype=np.int64)

        # FIX: clip both to valid token id range before decoding
        vocab_size = tokenizer.vocab_size
        predictions = np.clip(predictions, 0, vocab_size - 1)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        labels = np.clip(labels, 0, vocab_size - 1)

        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds = ["\n".join(pred.strip().split(". ")) for pred in decoded_preds]
        decoded_labels = [
            "\n".join(label.strip().split(". ")) for label in decoded_labels
        ]

        result = rouge.compute(
            predictions=decoded_preds, references=decoded_labels, use_stemmer=True
        )
        return {k: round(v, 4) for k, v in result.items()}

    # --- 3. Set up Training Arguments ---
    training_args = Seq2SeqTrainingArguments(
        output_dir="./bart-results",
        eval_strategy="epoch",
        report_to="tensorboard",
        logging_steps=50,
        learning_rate=5e-6,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=3,
        predict_with_generate=True,
        dataloader_num_workers=2,
    )

    # --- 4. Initialize
    # Trainer and Train ---
    print("Step 3: Train the model.")
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=tokenizer,
    )
    trainer.train()

    # --- 5. Save the fine-tuned model ---
    print("Step 4: Save the model to a local directory.")
    trainer.save_model("./my-custom-summarizer")
    tokenizer.save_pretrained("./my-custom-summarizer")


if __name__ == "__main__":
    print("--- Model Fine-Tuning Skeleton ---")
    fine_tune_model()
