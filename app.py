from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import cv2
import requests
from ultralytics import YOLO
from transformers import pipeline

app = Flask(__name__)
CORS(app)  # 👈 Enables CORS for all domains

# Load models
yolo_model = YOLO("yolov8n.pt")
gpt_pipeline = pipeline("text-classification", model="openai-community/gpt2")

# Transcription via OpenAI Whisper API
def transcribe_with_openai(video_path):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "[Missing API key]"
    try:
        with open(video_path, "rb") as audio_file:
            response = requests.post(
                "https://api.openai.com/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {api_key}"},
                files={"file": audio_file},
                data={"model": "whisper-1"},
            )
            response.raise_for_status()
            return response.json().get("text", "[No transcript found]")
    except Exception as e:
        return f"[Transcription error: {str(e)}]"

@app.route("/analyze", methods=["POST"])
def analyze():
    file = request.files.get("video")
    if not file:
        return jsonify({"error": "No video uploaded"}), 400

    temp_path = f"/tmp/{file.filename}"
    file.save(temp_path)

    # Transcribe audio
    transcript = transcribe_with_openai(temp_path)
    keywords = gpt_pipeline(transcript[:512]) if transcript else []

    # Frame extraction
    cap = cv2.VideoCapture(temp_path)
    frames = []
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % 30 == 0:
            frame_path = f"/tmp/frame_{count}.jpg"
            cv2.imwrite(frame_path, frame)
            frames.append(frame_path)
        count += 1
    cap.release()

    # Object detection
    detected_objects = {}
    for path in frames:
        result = yolo_model(path)
        labels = [yolo_model.names[int(box.cls)] for box in result[0].boxes] if result[0].boxes else []
        detected_objects[path] = labels

    return jsonify({
        "transcript": transcript,
        "keywords": keywords,
        "detectedObjects": detected_objects
    })

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
