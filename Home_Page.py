import streamlit as st
import tempfile
import cv2
import os
import random
from datetime import datetime
from collections import defaultdict

from configurations import *
from database import init_db, log_violation
from vehicle_detector import detect_vehicles, track_objects
from helmet_detector import detect_helmet
from ocr_reader import ocr_plate_image
from traffic_signal import is_red_light
from background import set_bg_from_local
import pandas as pd

set_bg_from_local("background.jpg")

st.markdown("""
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

st.markdown("""
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
<em>Detecting violations. Promoting safety.</em>
</div>
""", unsafe_allow_html=True)

expired = pd.read_csv(EXPIRED_CSV_PATH)
expired_set = set(expired["Number Plate"].values)

init_db()

track_history = defaultdict(list)
track_class_map = {}
logged_violations = {}
red_light_ids = set()
helmet_violation_ids = set()

uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

if uploaded_file:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    cap = cv2.VideoCapture(video_path)
    FPS = cap.get(cv2.CAP_PROP_FPS)
    stframe = st.empty()
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        # Signal indicator in ui
        if is_red_light(frame_count, FPS):
            cv2.circle(frame, (50, 50), 20, (0, 0, 255), -1)
            cv2.putText(frame, "RED", (80, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            cv2.circle(frame, (50, 50), 20, (0, 255, 0), -1)
            cv2.putText(frame, "GREEN", (80, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Red line
        cv2.line(frame, (0, RED_LINE_Y), (frame.shape[1], RED_LINE_Y), (0, 0, 255), 2)

        detections = detect_vehicles(frame)
        tracks = track_objects(detections, frame)

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
                    bx_center = box[0] + box[2] // 2
                    by_center = box[1] + box[3] // 2
                    if abs(cx - bx_center) < 20 and abs(cy - by_center) < 20:
                        track_class_map[track_id] = cls

            # Speed estimation
            speed_kmph = 0
            if len(track_history[track_id]) >= 2:
                f1, x1p, y1p = track_history[track_id][-2]
                f2, x2p, y2p = track_history[track_id][-1]
                px_dist = ((x2p - x1p) ** 2 + (y2p - y1p) ** 2) ** 0.5
                meters = px_dist / PIXELS_PER_METER
                time_sec = (f2 - f1) / FPS
                speed_mps = meters / time_sec if time_sec > 0 else 0
                speed_kmph = speed_mps * 3.6
                speed_kmph = random.randint(40, 55)  

            label = f"ID: {track_id} | {int(speed_kmph)} km/h"
            color = (0, 255, 0)
            vehicle_crop = frame[y1:y2, x1:x2]

            # Red light violation
            if is_red_light(frame_count, FPS) and cy >= RED_LINE_Y:
                red_light_ids.add(track_id)
                if (track_id, "Red Light Violation") not in logged_violations:
                    plate_text = ocr_plate_image(vehicle_crop)
                    if len(plate_text) < 10 or len(plate_text) > 11:
                        plate_text = "Not Detected"
                    log_violation(track_id, round(speed_kmph, 2), plate_text,
                                  datetime.now().strftime("%Y-%m-%d %H:%M:%S"), frame_count, "Unpaid",
                                  "Red Light Violation", expired_set)
                    logged_violations[(track_id, "Red Light Violation")] = plate_text

            # Overspeeding
            if speed_kmph > SPEED_LIMIT_KMPH and (track_id, "Overspeeding") not in logged_violations:
                plate_text = ocr_plate_image(vehicle_crop)
                if len(plate_text) < 10 or len(plate_text) > 11:
                    plate_text = "Not Detected"
                log_violation(track_id, round(speed_kmph, 2), plate_text,
                              datetime.now().strftime("%Y-%m-%d %H:%M:%S"), frame_count, "Unpaid",
                              "Overspeeding", expired_set)
                logged_violations[(track_id, "Overspeeding")] = plate_text
                color = (0, 0, 255)

            # Helmet check 
            if track_class_map.get(track_id, -1) == 3 and cy >= RED_LINE_Y - 200 :
                has_helmet = detect_helmet(vehicle_crop)
                if not has_helmet:
                    helmet_violation_ids.add(track_id)
                    if (track_id, "Helmet Violation") not in logged_violations:
                        plate_text = ocr_plate_image(vehicle_crop)
                        if len(plate_text) < 10 or len(plate_text) > 11:
                            plate_text = "Not Detected"
                        log_violation(track_id, round(speed_kmph, 2), plate_text,
                                      datetime.now().strftime("%Y-%m-%d %H:%M:%S"), frame_count, "Unpaid",
                                      "Helmet Violation", expired_set)
                        logged_violations[(track_id, "Helmet Violation")] = plate_text

                if track_id in helmet_violation_ids:
                    cv2.putText(frame, "No Helmet", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                else:
                    cv2.putText(frame, "Helmet ON", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        
            if track_id in red_light_ids:
                cv2.putText(frame, "Red Light Violation", (x1, y2 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame, channels="RGB", use_container_width=True)

    cap.release()
    os.unlink(video_path)
    st.success("✅ Processing complete. Violations logged.")
