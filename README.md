# YouTube Video Summarizer

This application transcribes YouTube videos by extracting their audio and passing it through a local Whisper AI model. The resulting text is then summarized using a custom, open-source AI model (by default: `sshleifer/distilbart-cnn-12-6`). It uses Gradio for a simple, interactive user interface.

## Why this app?
Unlike apps built on closed APIs (like ChatGPT or GPT-4), this application uses an open-source Hugging Face model running entirely locally. This means:
1. **Privacy**: Your data doesn't leave your machine/server.
2. **Customizability**: You can train and fine-tune this model on your own specific data (e.g., specific types of YouTube videos) to improve its performance.

## Installation

1. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   *(Note: This app relies on Whisper and yt-dlp which may require having `ffmpeg` installed on your system. To install `ffmpeg` on Ubuntu/Debian, run `sudo apt-get install ffmpeg`.)*

## Running the App

Run the following command to start the Gradio application:

```bash
python app.py
```

Then, open the provided URL in your web browser (usually `http://127.0.0.1:7860`).

## Fine-Tuning the Model Later

Because this app uses an open-source model from Hugging Face (`transformers` library), you can train it further on your own pairs of `(video_transcript, ideal_summary)`.

A basic skeleton for how you might fine-tune the model is provided in `train.py`.

### Steps to fine-tune:
1. Gather a dataset of transcripts and their corresponding summaries.
2. Format them into a Hugging Face Dataset format.
3. Update `train.py` with your dataset loading logic.
4. Run `python train.py`.
5. Update `model.py` to point to your new locally saved model directory instead of the default model name.
