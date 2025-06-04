from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

vehicle_model = YOLO("yolov8s.pt")
tracker = DeepSort(max_age=30)

def detect_vehicles(frame):
    results = vehicle_model(frame, conf=0.4, iou=0.5, classes=[2, 3, 5, 7])[0]
    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        w, h = x2 - x1, y2 - y1
        detections.append(([x1, y1, w, h], conf, cls))
    return detections

def track_objects(detections, frame):
    return tracker.update_tracks(detections, frame=frame)
