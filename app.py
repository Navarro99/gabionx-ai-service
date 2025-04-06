from flask import Flask, request, jsonify
import os
import cv2
import requests
from ultralytics import YOLO
from transformers import pipeline
from moviepy.editor import VideoFileClip

app = Flask(__name__)

# Initialize models once
yolo_model = YOLO("yolov8n.pt")
gpt_pipeline = pipeline("text-classification", model="openai-community/gpt2")  # Placeholder

# Real transcription using OpenAI Whisper API (requires OPENAI_API_KEY env variable)
def transcribe_with_openai(video_path):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "[No OpenAI API key provided]"

    try:
        with open(video_path, "rb") as audio_file:
            response = requests.post(
                "https://api.openai.com/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {api_key}"},
                files={"file": audio_file},
                data={"model": "whisper-1"}
            )
            response.raise_for_status()
            return response.json().get("text", "[No transcript found]")
    except Exception as e:
        return f"[Transcription error: {str(e)}]"

@app.route('/analyze', methods=['POST'])
def analyze():
    file = request.files['video']
    temp_path = f"/tmp/{file.filename}"
    file.save(temp_path)

    # Transcribe audio with OpenAI fallback
    transcript = transcribe_with_openai(temp_path)
    keywords = gpt_pipeline(transcript[:512]) if transcript else []

    # Extract frames
    frames_dir = "/tmp/frames/"
    os.makedirs(frames_dir, exist_ok=True)
    cap = cv2.VideoCapture(temp_path)
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

    # Detect objects
    detected_objects = {}
    for path in frame_paths:
        detections = yolo_model(path)
        labels = [yolo_model.names[int(box.cls)] for box in detections[0].boxes] if detections[0].boxes else []
        detected_objects[path] = labels

    return jsonify({
        "transcript": transcript,
        "keywords": keywords,
        "detectedObjects": detected_objects
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
