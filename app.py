from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import cv2
import requests
from ultralytics import YOLO
from transformers import pipeline

app = Flask(__name__)
CORS(app)  # ✅ Enables CORS for all origins

# Load models
yolo_model = YOLO("yolov8n.pt")
gpt_pipeline = pipeline("text-classification", model="openai-community/gpt2")

def transcribe_with_openai(video_path):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "[Missing OpenAI API key]"

    try:
        with open(video_path, "rb") as audio_file:
            response = requests.post(
                "https://api.openai.com/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {api_key}"},
                files={"file": audio_file},
                data={"model": "whisper-1"}
            )
            response.raise_for_status()
            return response.json().get("text", "")
    except Exception as e:
        return f"[Transcription failed: {str(e)}]"

@app.route("/analyze", methods=["POST"])
def analyze():
    file = request.files.get("video")
    if not file:
        return jsonify({"error": "No video uploaded"}), 400

    path = f"/tmp/{file.filename}"
    file.save(path)

    transcript = transcribe_with_openai(path)
    keywords = gpt_pipeline(transcript[:512]) if transcript else []

    frames_dir = "/tmp/frames/"
    os.makedirs(frames_dir, exist_ok=True)

    cap = cv2.VideoCapture(path)
    frame_paths = []
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % 30 == 0:
            frame_path = os.path.join(frames_dir, f"frame_{count}.jpg")
            cv2.imwrite(frame_path, frame)
            frame_paths.append(frame_path)
        count += 1
    cap.release()

    detected = {}
    for frame in frame_paths:
        detections = yolo_model(frame)
        labels = [yolo_model.names[int(box.cls)] for box in detections[0].boxes] if detections[0].boxes else []
        detected[os.path.basename(frame)] = labels

    return jsonify({
        "transcript": transcript,
        "keywords": keywords,
        "detectedObjects": detected
    })

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
