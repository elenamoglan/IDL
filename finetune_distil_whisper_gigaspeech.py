from dataclasses import dataclass
from typing import Any, Dict, List, Union
import io
import os
from pathlib import Path

import evaluate
import librosa
import numpy as np
import soundfile as sf
import torch
from datasets import Audio, load_dataset
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

model_id = "distil-whisper/distil-large-v3"

OUTPUT_DIR = "./distil-whisper-gigaspeech-xs"
FINAL_DIR = "./distil-whisper-gigaspeech-xs-final"

TRAIN_SAMPLES = 200          
EVAL_SAMPLES = 20           
MAX_STEPS = 200               
SAVE_STEPS = 15              
EVAL_STEPS = 15             
LOGGING_STEPS = 5

DO_EVAL = False              
PREDICT_WITH_GENERATE = False 
RESUME = True               # resume automatically from latest checkpoint if found

# Conservative training settings
LEARNING_RATE = 1e-5
WARMUP_STEPS = 10
TRAIN_BATCH_SIZE = 1
EVAL_BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 8

# Reduce CPU-side pressure
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"
torch.set_num_threads(4)


print("torch.cuda.is_available():", torch.cuda.is_available())
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

processor = AutoProcessor.from_pretrained(model_id)

# Keep float32 for stability during training
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id,
    dtype=torch.float32,
)
model = model.to(device)


dataset = load_dataset("speechcolab/gigaspeech", "xs")
dataset = dataset.cast_column("audio", Audio(decode=False))

train_ds = dataset["train"].select(range(min(TRAIN_SAMPLES, len(dataset["train"]))))
eval_split_name = "validation" if "validation" in dataset else "dev"
eval_ds = dataset[eval_split_name].select(range(min(EVAL_SAMPLES, len(dataset[eval_split_name]))))


def load_audio_robust(audio_obj):
    """
    Load audio from a datasets Audio(decode=False) object.
    Tries path first with librosa, then bytes with soundfile.
    Returns mono float32 audio and sampling rate.
    """
    audio_path = audio_obj.get("path")
    audio_bytes = audio_obj.get("bytes")

    if audio_path and os.path.exists(audio_path):
        audio_array, sampling_rate = librosa.load(audio_path, sr=None, mono=True)
        return np.asarray(audio_array, dtype=np.float32), sampling_rate

    if audio_bytes is not None:
        with io.BytesIO(audio_bytes) as bio:
            audio_array, sampling_rate = sf.read(bio)
        if len(audio_array.shape) > 1:
            audio_array = audio_array.mean(axis=1)
        return np.asarray(audio_array, dtype=np.float32), sampling_rate

    raise FileNotFoundError(f"Could not load audio. path={audio_path!r}")


def prepare_batch(batch):
    audio_array, sampling_rate = load_audio_robust(batch["audio"])

    features = processor.feature_extractor(
        audio_array,
        sampling_rate=sampling_rate,
    ).input_features[0]

    batch["input_features"] = np.asarray(features, dtype=np.float32)
    batch["labels"] = processor.tokenizer(batch["text"]).input_ids
    return batch


train_ds = train_ds.map(prepare_batch, remove_columns=train_ds.column_names)
eval_ds = eval_ds.map(prepare_batch, remove_columns=eval_ds.column_names)


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # Force float32 to avoid FP16 scaler issues
        batch["input_features"] = batch["input_features"].to(dtype=torch.float32)

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


wer_metric = evaluate.load("wer")


def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    if isinstance(pred_ids, tuple):
        pred_ids = pred_ids[0]

    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * wer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}


training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=EVAL_BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,
    learning_rate=LEARNING_RATE,
    warmup_steps=WARMUP_STEPS,
    max_steps=MAX_STEPS,
    eval_strategy="steps" if DO_EVAL else "no",
    eval_steps=EVAL_STEPS,
    save_steps=SAVE_STEPS,
    logging_steps=LOGGING_STEPS,
    predict_with_generate=PREDICT_WITH_GENERATE,
    generation_max_length=225,
    fp16=False,
    bf16=False,
    dataloader_num_workers=0,
    report_to="none",
    save_total_limit=3,
    load_best_model_at_end=False,
)


trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds if DO_EVAL else None,
    data_collator=DataCollatorSpeechSeq2SeqWithPadding(processor),
    processing_class=processor.feature_extractor,
    compute_metrics=compute_metrics if DO_EVAL and PREDICT_WITH_GENERATE else None,
)


def find_latest_checkpoint(output_dir: str):
    path = Path(output_dir)
    if not path.exists():
        return None

    checkpoints = []
    for p in path.glob("checkpoint-*"):
        try:
            step = int(p.name.split("-")[-1])
            checkpoints.append((step, str(p)))
        except ValueError:
            continue

    if not checkpoints:
        return None

    checkpoints.sort(key=lambda x: x[0])
    return checkpoints[-1][1]


resume_path = find_latest_checkpoint(OUTPUT_DIR) if RESUME else None

if resume_path:
    print(f"Resuming from checkpoint: {resume_path}")
    trainer.train(resume_from_checkpoint=resume_path)
else:
    print("Starting fresh training run.")
    trainer.train()


trainer.save_model(FINAL_DIR)
processor.save_pretrained(FINAL_DIR)

print(f"Saved final model to: {FINAL_DIR}")