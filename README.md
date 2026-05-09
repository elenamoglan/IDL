# YouTube Video Summarizer

This application transcribes YouTube videos by extracting their audio and passing it through a fine-tuned Whisper AI model. The resulting text is then summarized using a custom, fine-tuned AI model BART-Base. It uses Gradio for a simple, interactive user interface.

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

## Running the App

Run the following command to start the Gradio application:

```bash
python app.py
```

Then, open the provided URL in your web browser (usually `http://127.0.0.1:7860`).
