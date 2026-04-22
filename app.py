import gradio as gr
from model import Summarizer
from extractor import TranscriptExtractor

# Initialize the model at app startup
print("Initializing summarizer model...")
summarizer = Summarizer()
print("Model initialized!")

def process_video(url, max_length, min_length):
    """
    Main function to process the user request from Gradio.
    Extracts transcript from the YouTube URL and summarizes it.
    """
    if not url:
        return "Please enter a valid YouTube URL.", ""
        
    video_id = TranscriptExtractor.extract_video_id(url)
    if not video_id:
        return "Could not extract video ID from the provided URL. Please check the URL format.", ""
        
    transcript = TranscriptExtractor.get_transcript(video_id)
    if transcript.startswith("Error"):
        return f"Failed to get transcript. {transcript}", ""
        
    if not transcript.strip():
        return "Found a transcript, but it was empty.", ""
        
    summary = summarizer.summarize(transcript, max_length=max_length, min_length=min_length)
    
    return summary, transcript

# Gradio Interface
with gr.Blocks(title="YouTube Video Summarizer (Custom AI)") as demo:
    gr.Markdown("# 📺 YouTube Video Summarizer")
    gr.Markdown("This app uses a local Whisper model to transcribe YouTube videos and a custom open-source AI model (`sshleifer/distilbart-cnn-12-6`) to summarize the transcript. You can fine-tune this summarizer model locally later on your own data.")
    
    with gr.Row():
        with gr.Column(scale=2):
            url_input = gr.Textbox(label="YouTube Video URL", placeholder="https://www.youtube.com/watch?v=...", lines=1)
            
            with gr.Row():
                max_length_slider = gr.Slider(minimum=50, maximum=300, value=150, step=10, label="Max Summary Length (tokens)")
                min_length_slider = gr.Slider(minimum=20, maximum=150, value=40, step=10, label="Min Summary Length (tokens)")
            
            submit_btn = gr.Button("Process Video", variant="primary")
            
        with gr.Column(scale=3):
            summary_output = gr.Textbox(label="Summary", lines=6, interactive=False)
            transcript_output = gr.Textbox(label="Full Transcript", lines=10, interactive=False)
            
    submit_btn.click(
        fn=process_video, 
        inputs=[url_input, max_length_slider, min_length_slider], 
        outputs=[summary_output, transcript_output]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)