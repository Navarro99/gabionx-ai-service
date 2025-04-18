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

ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")

def transcribe_with_assemblyai(file_path):
    headers = {
        "authorization": ASSEMBLYAI_API_KEY,
    }

    # Upload the file to AssemblyAI
    with open(file_path, "rb") as f:
        upload_response = requests.post(
            "https://api.assemblyai.com/v2/upload",
            headers=headers,
            files={"file": f}
        )
    upload_url = upload_response.json().get("upload_url")
    if not upload_url:
        raise Exception("Failed to obtain upload URL from AssemblyAI.")

    # Start the transcription job
    json_data = {"audio_url": upload_url}
    transcript_response = requests.post(
        "https://api.assemblyai.com/v2/transcript",
        json=json_data,
        headers=headers
    )
    transcript_id = transcript_response.json().get("id")
    if not transcript_id:
        raise Exception("Failed to start transcription.")

    # Poll for result
    while True:
        poll_response = requests.get(
            f"https://api.assemblyai.com/v2/transcript/{transcript_id}",
            headers=headers
        )
        status = poll_response.json().get("status")
        if status == "completed":
            return poll_response.json().get("text")
        elif status == "error":
            raise Exception(f"Transcription failed: {poll_response.json().get('error')}")
        # You might want to add a delay here to prevent too frequent polling:
        # time.sleep(5)

@app.route("/analyze", methods=["POST"])
def analyze():
    file = request.files.get("video")
    if not file:
        return jsonify({"error": "No video uploaded"}), 400

    # Save the uploaded file to a temporary location
    path = f"/tmp/{file.filename}"
    file.save(path)

    try:
        transcript = transcribe_with_assemblyai(path)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    # Use a truncated transcript if it's too long for analysis
    keywords = gpt_pipeline(transcript[:512]) if transcript else []

    # Extract frames from the video for object detection
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
        # Check if boxes exist
        if detections and detections[0].boxes is not None:
            labels = [yolo_model.names[int(box.cls)] for box in detections[0].boxes]
        else:
            labels = []
        detected[os.path.basename(frame)] = labels

    return jsonify({
        "transcript": transcript,
        "keywords": keywords,
        "detectedObjects": detected
    })

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
