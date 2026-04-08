import argparse

from src.inference import SarcasmRecognize


def predict_from_audio(audio_path: str, model_path: str) -> dict:
    recognizer = SarcasmRecognize(model_path=model_path)
    transcript, label = recognizer.predict_sarcasm(audio_path)
    return {
        "transcript": transcript,
        "label": label,
    }


def main():
    parser = argparse.ArgumentParser(description="Run sarcasm prediction on a single audio file.")
    parser.add_argument("--input", type=str, required=True, help="Path to the input audio file (.wav)")
    parser.add_argument("--model", type=str, required=True, help="Path to the pretrained model (.pth)")
    args = parser.parse_args()

    result = predict_from_audio(audio_path=args.input, model_path=args.model)

    print("\n===============================")
    print(f"📝 Transcribed Text: {result['transcript']}")
    print(f"🎯 Sarcasm Prediction: {result['label']}")
    print("===============================")


if __name__ == "__main__":
    main()