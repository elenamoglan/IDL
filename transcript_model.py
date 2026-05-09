import re
import os
import yt_dlp
import warnings
import torch
import soundfile as sf
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

warnings.filterwarnings("ignore")

WHISPER_MODEL_ID = "catnip11/finetuned-distil-whisper"
BASE_MODEL_ID = "distil-whisper/distil-large-v3"


class TranscriptExtractor:
    whisper_model = None

    @classmethod
    def load_model(cls):
        if cls.whisper_model is None:
            print(f"Loading Whisper model: {WHISPER_MODEL_ID}...")

            has_cuda = torch.cuda.is_available()
            dtype = torch.float16 if has_cuda else torch.float32
            device = "cuda:0" if has_cuda else "cpu"
            pipeline_device = 0 if has_cuda else -1

            # Try loading as full model first, fall back to PEFT/LoRA adapter
            try:
                model = AutoModelForSpeechSeq2Seq.from_pretrained(
                    WHISPER_MODEL_ID,
                    dtype=dtype,
                    low_cpu_mem_usage=True,
                )
                processor = AutoProcessor.from_pretrained(WHISPER_MODEL_ID)
                print("Loaded as full model checkpoint.")

            except Exception:
                print("Full model load failed, trying PEFT/LoRA adapter...")
                try:
                    from peft import PeftModel, PeftConfig
                    peft_config = PeftConfig.from_pretrained(WHISPER_MODEL_ID)
                    base_id = peft_config.base_model_name_or_path or BASE_MODEL_ID

                    base_model = AutoModelForSpeechSeq2Seq.from_pretrained(
                        base_id, dtype=dtype, low_cpu_mem_usage=True
                    )
                    model = PeftModel.from_pretrained(base_model, WHISPER_MODEL_ID)
                    try:
                        model = model.merge_and_unload()
                    except Exception:
                        pass

                    try:
                        processor = AutoProcessor.from_pretrained(WHISPER_MODEL_ID)
                    except Exception:
                        processor = AutoProcessor.from_pretrained(base_id)

                    print("Loaded as PEFT/LoRA adapter.")
                except ImportError:
                    raise RuntimeError("Model is a PEFT adapter — run: pip install peft")

            model.to(device)
            model.eval()

            cls.whisper_model = pipeline(
                task="automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                max_new_tokens=128,
                chunk_length_s=25,
                batch_size=8,
                dtype=dtype,
                device=pipeline_device,
            )
            print("Whisper pipeline ready.")

    @staticmethod
    def extract_video_id(url):
        """
        Extracts the YouTube video ID from various YouTube URL formats.
        """
        patterns = [
            r"(?:v=|\/)([0-9A-Za-z_-]{11}).*",
            r"(?:youtu\.be\/)([0-9A-Za-z_-]{11})",
            r"(?:shorts\/)([0-9A-Za-z_-]{11})",
        ]
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return None

    @classmethod
    def get_transcript(cls, video_id):
        """
        Downloads audio from YouTube and transcribes it using
        the fine-tuned distil-whisper model.
        """
        cls.load_model()
        url = f"https://www.youtube.com/watch?v={video_id}"

        # Download as WAV at 16kHz, required by distil-whisper
        audio_file = f"audio_{video_id}.wav"

        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": f"audio_{video_id}.%(ext)s",
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "wav",
                }
            ],
            "postprocessor_args": ["-ar", "16000", "-ac", "1"],
            "quiet": True,
            "no_warnings": True,
        }

        try:
            print(f"Downloading audio for {video_id}...")
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])

            print(f"Transcribing audio for {video_id}...")
            audio, sr = sf.read(audio_file)

            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)

            result = cls.whisper_model(
                {"array": audio, "sampling_rate": sr},
                return_timestamps=False,
                generate_kwargs={"language": "english", "task": "transcribe"},
            )

            text_transcript = result["text"]
            text_transcript = " ".join(text_transcript.split())

            if os.path.exists(audio_file):
                os.remove(audio_file)

            return text_transcript

        except Exception as e:
            if os.path.exists(audio_file):
                os.remove(audio_file)
            return f"Error fetching transcript: {str(e)}"

