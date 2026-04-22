import re
import os
import yt_dlp
import whisper
import warnings

warnings.filterwarnings("ignore")


class TranscriptExtractor:
    # Load whisper model once when class is used
    whisper_model = None

    @classmethod
    def load_model(cls):
        if cls.whisper_model is None:
            print("Loading Whisper model (tiny)...")
            cls.whisper_model = whisper.load_model("tiny")

    @staticmethod
    def extract_video_id(url):
        """
        Extracts the YouTube video ID from various YouTube URL formats.
        """
        # Handle regular youtube.com/watch?v=, youtu.be/, and youtube.com/shorts/
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
        Fetches the transcript for the given video ID using yt-dlp and whisper.
        Returns the text as a single string.
        """
        cls.load_model()
        url = f"https://www.youtube.com/watch?v={video_id}"
        audio_file = f"audio_{video_id}.mp3"

        ydl_opts = {
            "format": "best",
            "outtmpl": f"audio_{video_id}.%(ext)s",
            # "cookiefile": "cookies.txt",
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "mp3",
                    "preferredquality": "192",
                }
            ],
            "quiet": True,
            "no_warnings": True,
        }

        try:
            print(f"Downloading audio for {video_id}...")
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])

            print(f"Transcribing audio for {video_id} using Whisper...")
            result = cls.whisper_model.transcribe(audio_file)
            text_transcript = result["text"]

            # Clean up newlines
            text_transcript = " ".join(text_transcript.split())

            # Clean up audio file
            if os.path.exists(audio_file):
                os.remove(audio_file)

            return text_transcript
        except Exception as e:
            if os.path.exists(audio_file):
                os.remove(audio_file)
            return f"Error fetching transcript: {str(e)}"


# Quick test when running the script directly
if __name__ == "__main__":
    test_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # Rickroll
    video_id = TranscriptExtractor.extract_video_id(test_url)
    print(f"Extracted ID: {video_id}")
    if video_id:
        print("Transcript Preview:")
        print(TranscriptExtractor.get_transcript(video_id)[:500])
