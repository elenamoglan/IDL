import argparse
import json
import os
import sys
import subprocess
from pathlib import Path
from typing import Any

import soundfile as sf
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


FFMPEG_EXE = r"C:\Users\katiu\Downloads\ffmpeg-N-123477-g5640bd3a4f-win64-gpl-shared\ffmpeg-N-123477-g5640bd3a4f-win64-gpl-shared\bin\ffmpeg.exe"

DEFAULT_MODEL_ID = "catnip11/finetuned-distil-whisper"
BASE_MODEL_ID = "distil-whisper/distil-large-v3"


def extract_audio(video_path: str, audio_path: str) -> None:
    cmd = [
        FFMPEG_EXE,
        "-y",
        "-i",
        video_path,
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        audio_path,
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{result.stderr}")


def make_json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [make_json_safe(v) for v in obj]

    if isinstance(obj, tuple):
        return [make_json_safe(v) for v in obj]

    return obj


def load_model_and_processor(model_id: str, dtype: torch.dtype):
    """
    Loads either:
    1. a full Hugging Face model checkpoint, or
    2. a PEFT/LoRA adapter repo on top of BASE_MODEL_ID.

    If your repo contains only adapter files, install:
        pip install peft
    """
    hf_token = os.environ.get("HF_TOKEN")

    model_kwargs = {
        "dtype": dtype,
        "low_cpu_mem_usage": True,
    }

    processor_kwargs = {}

    if hf_token:
        model_kwargs["token"] = hf_token
        processor_kwargs["token"] = hf_token

    print(f"Trying to load full model from: {model_id}")

    try:
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            **model_kwargs,
        )

        processor = AutoProcessor.from_pretrained(
            model_id,
            **processor_kwargs,
        )

        print("Loaded as full model checkpoint.")
        return model, processor

    except Exception as full_model_error:
        print("Full model load failed.")
        print(str(full_model_error))

    print("Trying to load as PEFT/LoRA adapter...")

    try:
        from peft import PeftModel, PeftConfig
    except ImportError as exc:
        raise RuntimeError(
            "The Hugging Face repo did not load as a full model. "
            "It might be a LoRA/PEFT adapter repo. Install PEFT first:\n\n"
            "pip install peft\n"
        ) from exc

    try:
        peft_kwargs = {}
        if hf_token:
            peft_kwargs["token"] = hf_token

        peft_config = PeftConfig.from_pretrained(model_id, **peft_kwargs)
        base_model_id = peft_config.base_model_name_or_path or BASE_MODEL_ID

        print(f"Adapter detected. Loading base model: {base_model_id}")

        base_model = AutoModelForSpeechSeq2Seq.from_pretrained(
            base_model_id,
            **model_kwargs,
        )

        model = PeftModel.from_pretrained(
            base_model,
            model_id,
            **peft_kwargs,
        )

        try:
            model = model.merge_and_unload()
            print("Merged PEFT adapter into base model.")
        except Exception:
            print("Using PEFT adapter without merging.")

        try:
            processor = AutoProcessor.from_pretrained(
                model_id,
                **processor_kwargs,
            )
        except Exception:
            processor = AutoProcessor.from_pretrained(
                base_model_id,
                **processor_kwargs,
            )

        print("Loaded as PEFT/LoRA adapter.")
        return model, processor

    except Exception as adapter_error:
        raise RuntimeError(
            f"Could not load '{model_id}' as a full model or PEFT adapter.\n\n"
            "Most likely problem: your Hugging Face repo does not contain model weights.\n\n"
            "A full model repo should contain one of these:\n"
            "- model.safetensors\n"
            "- pytorch_model.bin\n"
            "- sharded model files plus an index JSON\n\n"
            "A PEFT adapter repo should contain:\n"
            "- adapter_config.json\n"
            "- adapter_model.safetensors or adapter_model.bin\n\n"
            f"Original adapter error:\n{adapter_error}"
        ) from adapter_error


def transcribe_audio(
    audio_path: str,
    model_id: str = DEFAULT_MODEL_ID,
    chunk_length_s: int = 25,
    batch_size: int = 8,
) -> dict:
    has_cuda = torch.cuda.is_available()

    device_name = "cuda:0" if has_cuda else "cpu"
    pipeline_device = 0 if has_cuda else -1
    dtype = torch.float16 if has_cuda else torch.float32

    print(f"Model ID: {model_id}")
    print(f"Device: {device_name}")
    print(f"Dtype: {dtype}")

    model, processor = load_model_and_processor(model_id, dtype)

    model.to(device_name)
    model.eval()

    asr_pipe = pipeline(
        task="automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        chunk_length_s=chunk_length_s,
        batch_size=batch_size,
        dtype=dtype,
        device=pipeline_device,
    )

    audio, sr = sf.read(audio_path)

    if sr != 16000:
        raise ValueError(f"Expected 16kHz WAV audio, but got {sr} Hz")

    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)

    result = asr_pipe(
        {
            "array": audio,
            "sampling_rate": sr,
        },
        return_timestamps=True,
        generate_kwargs={
            "language": "english",
            "task": "transcribe",
        },
    )

    return make_json_safe(result)


def build_output(
    video_path: str,
    audio_path: str,
    result: dict,
    model_id: str,
) -> dict:
    chunks = result.get("chunks", [])

    segment_count = len(chunks)
    duration_estimate = None

    if chunks:
        last_ts = chunks[-1].get("timestamp")
        if isinstance(last_ts, (list, tuple)) and len(last_ts) == 2:
            duration_estimate = last_ts[1]

    return {
        "source_video": str(Path(video_path).resolve()),
        "extracted_audio": str(Path(audio_path).resolve()),
        "asr_model": model_id,
        "full_text": result.get("text", "").strip(),
        "segments": [
            {
                "start": seg.get("timestamp", [None, None])[0],
                "end": seg.get("timestamp", [None, None])[1],
                "text": seg.get("text", "").strip(),
            }
            for seg in chunks
        ],
        "metadata": {
            "segment_count": segment_count,
            "estimated_duration_seconds": duration_estimate,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transcribe a video file into transcript JSON."
    )

    parser.add_argument("video", help="Path to input video file")

    parser.add_argument(
        "--output",
        default="transcript.json",
        help="Path to output JSON file",
    )

    parser.add_argument(
        "--audio-temp",
        default="temp_audio.wav",
        help="Temporary extracted audio path",
    )

    parser.add_argument(
        "--model-id",
        default=None,
        help="Hugging Face model repo or local model path",
    )

    parser.add_argument(
        "--model",
        default=None,
        help="Backward-compatible alias for --model-id",
    )

    parser.add_argument(
        "--chunk-length",
        type=int,
        default=25,
        help="Chunk length in seconds for long-form transcription",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for ASR pipeline",
    )

    parser.add_argument(
        "--keep-audio",
        action="store_true",
        help="Keep extracted WAV file instead of deleting it",
    )

    args = parser.parse_args()

    video_path = args.video
    output_path = args.output
    audio_path = args.audio_temp
    model_id = args.model_id or args.model or DEFAULT_MODEL_ID

    if not os.path.exists(video_path):
        print(f"Input video not found: {video_path}", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(FFMPEG_EXE):
        print(f"ffmpeg executable not found: {FFMPEG_EXE}", file=sys.stderr)
        sys.exit(1)

    try:
        print("1/3 Extracting audio...")
        extract_audio(video_path, audio_path)

        print("2/3 Transcribing audio...")
        result = transcribe_audio(
            audio_path=audio_path,
            model_id=model_id,
            chunk_length_s=args.chunk_length,
            batch_size=args.batch_size,
        )

        print("3/3 Saving JSON...")
        output = build_output(
            video_path=video_path,
            audio_path=audio_path,
            result=result,
            model_id=model_id,
        )

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        print(f"Done. Transcript JSON saved to: {output_path}")

    finally:
        if not args.keep_audio and os.path.exists(audio_path):
            try:
                os.remove(audio_path)
            except OSError:
                pass


if __name__ == "__main__":
    main()