from pathlib import Path

import gradio as gr

from src.predict import predict_from_audio

# Update this path to wherever you store the pretrained checkpoint locally
MODEL_PATH = "models/model.pth"


def run_demo(audio_input):
    if audio_input is None:
        return "No transcript available.", "No prediction."

    audio_path = audio_input if isinstance(audio_input, str) else str(audio_input)

    if not Path(MODEL_PATH).exists():
        return "Model checkpoint not found.", f"Please update MODEL_PATH in demo/app.py: {MODEL_PATH}"

    result = predict_from_audio(audio_path=audio_path, model_path=MODEL_PATH)
    return result["transcript"], result["label"]


demo = gr.Interface(
    fn=run_demo,
    inputs=gr.Audio(type="filepath", label="Upload WAV audio"),
    outputs=[
        gr.Textbox(label="Transcript"),
        gr.Textbox(label="Sarcasm Prediction"),
    ],
    title="SarcEmotiq Demo",
    description="Upload a .wav audio clip to run sarcasm prediction.",
)


if __name__ == "__main__":
    demo.launch()