from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import cv2
import requests
from ultralytics import YOLO
from transformers import pipeline

app = Flask(__name__)
CORS(app)

# Load models
yolo_model = YOLO("yolov8n.pt")
gpt_pipeline = pipeline("text-classification", model="openai-community/gpt2")

# AssemblyAI configuration
ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")
ASSEMBLYAI_UPLOAD_URL = "https://api.assemblyai.com/v2/upload"
ASSEMBLYAI_TRANSCRIBE_URL = "https://api.assemblyai.com/v2/transcript"

headers = {
    "authorization": ASSEMBLYAI_API_KEY,
    "content-type": "application/json"
}

def transcribe_with_assemblyai(video_path):
    if not ASSEMBLYAI_API_KEY:
        return "[Missing AssemblyAI API key]"

    try:
        # Step 1: Upload the video file
        with open(video_path, 'rb') as f:
            upload_res = requests.post(
                ASSEMBLYAI_UPLOAD_URL,
                headers={"authorization": ASSEMBLYAI_API_KEY},
                files={"file": f}
            )
        upload_url = upload_res.json()["upload_url"]

        # Step 2: Start the transcription job
        transcribe_res = requests.post(
            ASSEMBLYAI_TRANSCRIBE_URL,
            headers=headers,
            json={"audio_url": upload_url}
        )
        transcript_id = transcribe_res.json()["id"]

        # Step 3: Poll for completion
        polling_endpoint = f"{ASSEMBLYAI_TRANSCRIBE_URL}/{transcript_id}"
        while True:
            poll_res = requests.get(polling_endpoint, headers=headers).json()
            if poll_res["status"] == "completed":
                return poll_res["text"]
            elif poll_res["status"] == "error":
                return f"[AssemblyAI Error: {poll_res['error']}]"
    except Exception as e:
        return f"[Transcription failed: {str(e)}]"

@app.route("/analyze", methods=["POST"])
def analyze():
    file = request.files.get("video")
    if not file:
        return jsonify({"error": "No video uploaded"}), 400

    path = f"/tmp/{file.filename}"
    file.save(path)

    transcript = transcribe_with_assemblyai(path)
    keywords = gpt_pipeline(transcript[:512]) if transcript and not transcript.startswith("[") else []

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
