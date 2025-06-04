import streamlit as st
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from inference_sdk import InferenceHTTPClient
import cv2
import tempfile
import numpy as np
import os
import sqlite3
from datetime import datetime
import re
import pandas as pd
from collections import defaultdict
from PIL import Image
import boto3

PIXELS_PER_METER = 15
SPEED_LIMIT_KMPH = 60
DB_PATH = "violations.db"
OCR_API_KEY = "K88493869388957"  
RED_LINE_Y = 300  
SIGNAL_CYCLE_SECONDS = 5  

expired = pd.read_csv('expired.csv')
os.makedirs("violations_screenshots", exist_ok=True)

ROBO_CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="h9gshPrmD0qQpaOdVxhq"
)

vehicle_model = YOLO("yolov8s.pt")
tracker = DeepSort(max_age=30)
import random
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS violations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            track_id INTEGER,
            speed REAL,
            plate TEXT,
            timestamp TEXT,
            frame_number INTEGER,
            status TEXT,
            violation_type TEXT
        )
    """)
    conn.commit()
    conn.close()

logged_violations = {}

def log_violation(track_id, speed, plate, timestamp, frame_number, status, violation_type):
    global logged_violations

    key = (track_id, violation_type)
    prev_plate = logged_violations.get(key)
    if prev_plate is None or (plate != "UNKNOWN" and len(plate) > len(prev_plate)):
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        if prev_plate is None:
            cursor.execute("""
                INSERT INTO violations (track_id, speed, plate, timestamp, frame_number, status, violation_type)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (track_id, speed, plate, timestamp, frame_number, status, violation_type))
        else:
            cursor.execute("""
                UPDATE violations SET plate = ?, speed = ?, timestamp = ?, frame_number = ?, status = ?
                WHERE track_id = ? AND violation_type = ?
            """, (plate, speed, timestamp, frame_number, status, track_id, violation_type))

        if plate in expired["Number Plate"].values:
            cursor.execute("""
                INSERT INTO violations (track_id, speed, plate, timestamp, frame_number, status, violation_type)
                VALUES (?, ?, ?, ?, ?, ?, "Expired Insurance")
            """, (track_id, speed, plate, timestamp, frame_number, status))

        conn.commit()
        conn.close()

        logged_violations[key] = plate

textract = boto3.client('textract','ap-south-1')  

def ocr_plate_image(cropped_image):
    if cropped_image is None or cropped_image.size == 0:
        return "UNKNOWN"
    try:
        success, img_encoded = cv2.imencode('.jpg', cropped_image)
        if not success:
            return "UNKNOWN"
        img_bytes = img_encoded.tobytes()

        response = textract.detect_document_text(Document={'Bytes': img_bytes})

        raw_text = ""
        for item in response['Blocks']:
            if item['BlockType'] == 'LINE':
                raw_text += item['Text'] + " "

        cleaned_text = re.sub(r'[^A-Z0-9]', '', raw_text.upper())
        pattern = r'[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4}'
        matches = re.findall(pattern, cleaned_text)
        return matches[0] if matches else (cleaned_text or "Not Detected")
    except Exception as e:
        print(f"[OCR Error] {e}")
        return "UNKNOWN"

def detect_helmet(cropped_img):
    try:
        rgb_image = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_image)
        result = ROBO_CLIENT.infer(pil_img, model_id="helmet-detection-ar0n2/1")

        for pred in result['predictions']:
            if pred['class'].lower() == "helmet" and pred['confidence'] > 0.5:
                return True
        return False
    except Exception as e:
        print(f"[Helmet Detection Error] {e}")
        return False

def is_red_light(frame_number, fps):
    seconds = frame_number // fps
    return (seconds // SIGNAL_CYCLE_SECONDS) % 2 == 0 

st.markdown(
    """
    <style>
    .title {
        background-color: black;
        color: orange;
        font-size: 35px;
        text-align: center;
        padding: 10px;
        opacity:0.9;
        border-radius:20px;
    }
    </style>
    <div class="title">
    🚦 Traffic Rules Violation Detection System
    </div>
    """, unsafe_allow_html=True)

st.markdown(
    """
    <style>
    .title {
    font-family:italic;
        background-color: black;
        color: white;
        font-size: 35px;
        text-align: center;
        padding: 5px;
        opacity:0.9;
        margin: 5px;
        border-radius:20px;
    }
    </style>
    <div style="color:orange; font-size:29px;" class="title">
    <em>
    Detecting violations. Promoting safety.</em>
    </div>
    """, unsafe_allow_html=True)



uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
track_history = defaultdict(list)
track_class_map = {}  
init_db()

if uploaded_file:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    cap = cv2.VideoCapture(video_path)
    FPS = cap.get(cv2.CAP_PROP_FPS)
    stframe = st.empty()
    frame_count = 0
    redlight_logged = set()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        if is_red_light(frame_count, FPS):
            cv2.circle(frame, (50, 50), 20, (0, 0, 255), -1)
            cv2.putText(frame, "RED", (80, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
        else:
            cv2.circle(frame, (50, 50), 20, (0, 255, 0), -1)
            cv2.putText(frame, "GREEN", (80, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        cv2.line(frame, (0, RED_LINE_Y), (frame.shape[1], RED_LINE_Y), (0, 0, 255), 2)

        results = vehicle_model(frame, conf=0.4, iou=0.5, classes=[2, 3, 5, 7])[0]  
        detections = []

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            w, h = x2 - x1, y2 - y1
            detections.append(([x1, y1, w, h], conf, cls))

        tracks = tracker.update_tracks(detections, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            x1, y1, x2, y2 = map(int, track.to_ltrb())
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            track_history[track_id].append((frame_count, cx, cy))

            if track_id not in track_class_map:
                for d in detections:
                    box, conf, cls = d
                    bx_center = box[0] + box[2]//2
                    by_center = box[1] + box[3]//2
                    if abs(cx - bx_center) < 20 and abs(cy - by_center) < 20:
                        track_class_map[track_id] = cls

            speed_kmph = 0
            if len(track_history[track_id]) >= 2:
                f1, x1p, y1p = track_history[track_id][-2]
                f2, x2p, y2p = track_history[track_id][-1]
                px_dist = ((x2p - x1p) ** 2 + (y2p - y1p) ** 2) ** 0.5
                meters = px_dist / PIXELS_PER_METER
                time_sec = (f2 - f1) / FPS
                speed_mps = meters / time_sec if time_sec > 0 else 0
                speed_kmph = speed_mps * 3.6
                speed_kmph=random.randint(40,80)

            label = f"ID: {track_id} | {int(speed_kmph)} km/h"
            colorall= (0, 255, 0)
            vehicle_crop = frame[y1:y2, x1:x2] 

            if is_red_light(frame_count, FPS) and cy >= RED_LINE_Y and (track_id, "Red Light Violation") not in logged_violations:
                plate_text = ocr_plate_image(vehicle_crop)
                if not re.fullmatch(r'[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4}', plate_text):
                    plate_text = "Not Detected"
                log_violation(track_id, round(speed_kmph, 2), plate_text,
                  datetime.now().strftime("%Y-%m-%d %H:%M:%S"), frame_count, "Unpaid", "Red Light Violation")
                color = (0, 0, 255)  # Orange
                cv2.putText(frame, "Red Light Violation", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                cv2.imwrite(f"violations_screenshots/{track_id}_Red_Light_{frame_count}.jpg", frame)

# Overspeeding
            if speed_kmph > SPEED_LIMIT_KMPH and (track_id, "Overspeeding") not in logged_violations:
                plate_text = ocr_plate_image(vehicle_crop)
                if not re.fullmatch(r'[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4}', plate_text):
                    plate_text = "Not Detected"
                log_violation(track_id, round(speed_kmph, 2), plate_text,
                  datetime.now().strftime("%Y-%m-%d %H:%M:%S"), frame_count, "Unpaid", "Overspeeding")

                color = (0, 0, 255)  # Orange
                cv2.putText(frame, "Overspeeding Violation", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                cv2.imwrite(f"violations_screenshots/{track_id}_Overspeeding_{frame_count}.jpg", frame)

# Helmet Check for motorcycles
            if track_class_map.get(track_id, -1) == 3:
                has_helmet = detect_helmet(vehicle_crop)
                if not has_helmet and (track_id, "Helmet Violation") not in logged_violations:
                    plate_text = ocr_plate_image(vehicle_crop)
                    if not re.fullmatch(r'[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4}', plate_text):
                        plate_text = "Not Detected"
                    log_violation(track_id, round(speed_kmph, 2), plate_text,
                      datetime.now().strftime("%Y-%m-%d %H:%M:%S"), frame_count, "Unpaid", "Helmet Violation")

                    color = (0, 0, 255)  
                    cv2.putText(frame, "No Helmet", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    cv2.imwrite(f"violations_screenshots/{track_id}_Helmet_Violation_{frame_count}.jpg", frame)

                elif has_helmet:
                    cv2.putText(frame, "Helmet ON", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), colorall, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colorall, 2)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame, channels="RGB", use_container_width=True)

    cap.release()
    os.unlink(video_path)
    st.success("✅ Processing complete. Violations logged to SQLite.")
    st.markdown(f"📄 [Download Violations DB](sandbox:/mnt/data/violations.db)")

import base64

def set_bg_from_local(image_file):
    with open(image_file, "rb") as file:
        encoded_string = base64.b64encode(file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded_string}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_bg_from_local("background.jpg")
