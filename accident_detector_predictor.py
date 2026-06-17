import os
from collections import deque
from typing import Callable, Dict, Optional

os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices=false"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import cv2
import numpy as np

from models.cnn_model import load_cnnmodel
from models.yolo_model import load_yolomodel

DETECTION_MODES = {"cnn", "yolo", "hybrid"}

CNN_THRESHOLD = 0.7
YOLO_CONFIDENCE = 0.4
IOU_COLLISION_THRESHOLD = 0.15
PROXIMITY_DISTANCE = 80
FRAME_HISTORY = 10
CNN_FRAME_SKIP = 3

PREDICTION_ENABLED = True
TRAJECTORY_HISTORY = 15
COLLISION_COURSE_ANGLE = 30
# COLLISION_COURSE_ANGLE = 15
DANGEROUS_APPROACH_SPEED = 50
# DANGEROUS_APPROACH_SPEED = 20
PREDICTION_HORIZON = 20
# PREDICTION_HORIZON = 40
CNN_CONSECUTIVE_THRESHOLD = 3
TRACK_MAX_MISSED_FRAMES = 5

_cnn_model = None
_yolo_model = None
_input_shape = None


def _get_models():
    global _cnn_model, _yolo_model, _input_shape
    if _cnn_model is None:
        _cnn_model = load_cnnmodel()
        _input_shape = _cnn_model.input_shape
    if _yolo_model is None:
        _yolo_model = load_yolomodel()
        try:
            _yolo_model.to("cuda")
        except Exception:
            # Safe fallback to CPU if CUDA is unavailable.
            pass
    return _cnn_model, _yolo_model, _input_shape

#calculate Intersection over Union. This helps in measuring how much two boxes overlap
def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    if x2 < x1 or y2 < y1:
        return 0.0
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0.0


def detect_vehicle_collisions(boxes, iou_threshold=0.15, distance_threshold=80):
    if len(boxes) < 2:
        return False, []
    collisions = []
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            box1 = boxes[i][:4]
            box2 = boxes[j][:4]
            iou = calculate_iou(box1, box2)
            center1 = ((box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2)
            center2 = ((box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2)
            distance = np.sqrt((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2)
            if iou > iou_threshold or distance < distance_threshold:
                collisions.append(
                    {
                        "box1": box1,
                        "box2": box2,
                        "iou": iou,
                        "distance": distance,
                        "type": "overlap" if iou > iou_threshold else "proximity",
                    }
                )
    return len(collisions) > 0, collisions


def calculate_velocity(track_history):
    if len(track_history) < 2:
        return (0, 0), 0
    recent_points = list(track_history)[-5:]
    if len(recent_points) < 2:
        return (0, 0), 0
    velocities = []
    for i in range(1, len(recent_points)):
        p1 = recent_points[i - 1]
        p2 = recent_points[i]
        velocities.append((p2[0] - p1[0], p2[1] - p1[1]))
    avg_vx = np.mean([v[0] for v in velocities])
    avg_vy = np.mean([v[1] for v in velocities])
    speed = np.sqrt(avg_vx**2 + avg_vy**2)
    return (avg_vx, avg_vy), speed


def predict_collision_course(box1, track1, box2, track2, angle_threshold=30, speed_threshold=50):
    center1 = np.array([(box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2])
    center2 = np.array([(box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2])
    vel1, speed1 = calculate_velocity(track1)
    vel2, speed2 = calculate_velocity(track2)

    # If both cars are barely moving, they won't collide
    if speed1 < 5 and speed2 < 5:
        return False, float("inf"), 0.0

    relative_pos = center2 - center1
    future_center1 = center1 + np.array(vel1) * PREDICTION_HORIZON
    future_center2 = center2 + np.array(vel2) * PREDICTION_HORIZON
    future_distance = np.linalg.norm(future_center1 - future_center2)
    current_distance = np.linalg.norm(relative_pos)
    is_approaching = future_distance < current_distance
    approach_speed = (current_distance - future_distance) / PREDICTION_HORIZON

    heading_towards = False
    if speed1 > 5 and speed2 > 5:
        dot_product = vel1[0] * vel2[0] + vel1[1] * vel2[1] # measures how aligned their directions are
        mag_product = speed1 * speed2
        if mag_product > 0:
            cos_angle = np.clip(dot_product / mag_product, -1, 1)
            angle = np.arccos(cos_angle) * 180 / np.pi
            heading_towards = angle > (180 - angle_threshold)

    time_to_collision = current_distance / approach_speed if approach_speed > 0 else float("inf")
    confidence = 0.0
    if is_approaching:
        confidence += 0.3
    if approach_speed > speed_threshold:
        confidence += 0.3
    if heading_towards:
        confidence += 0.4
    if current_distance < 200:
        confidence += 0.4
    confidence = min(1.0, confidence)
    will_collide = (confidence > 0.5) and (time_to_collision < 30)
    return will_collide, time_to_collision, confidence


def match_vehicles_to_tracks(current_boxes, tracks, max_distance=100):
    if not tracks:
        return {}, list(range(len(current_boxes)))
    matches = {}
    unmatched_detections = list(range(len(current_boxes)))
    current_centers = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in current_boxes]
    for track_id, track_data in tracks.items():
        history = track_data["history"]
        if not history:
            continue
        last_center = history[-1][:2]
        min_dist = max_distance
        best_match = None
        for i, center in enumerate(current_centers):
            if i not in matches.values():
                dist = np.sqrt((center[0] - last_center[0]) ** 2 + (center[1] - last_center[1]) ** 2)
                if dist < min_dist:
                    min_dist = dist
                    best_match = i
        if best_match is not None:
            matches[track_id] = best_match
            unmatched_detections.remove(best_match)
    return matches, unmatched_detections


def _compute_accident(mode: str, cnn_confirmed: bool, yolo_collision_smoothed: bool) -> bool:
    if mode == "cnn":
        return cnn_confirmed
    if mode == "yolo":
        return yolo_collision_smoothed
    return cnn_confirmed and yolo_collision_smoothed


def _mode_banner(mode: str) -> str:
    if mode == "cnn":
        return "Accident DETECTION: CNN Only (YOLO collision validation OFF)"
    if mode == "yolo":
        return "Accident DETECTION: YOLO Collision Only (CNN OFF)"
    return "Accident DETECTION: CNN + YOLO Consensus"


def run_detector(
    video_path: str,
    output_path: str,
    detection_mode: str = "hybrid",
    metrics_callback: Optional[Callable[[Dict[str, object]], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
):
    mode = detection_mode.strip().lower()
    if mode not in DETECTION_MODES:
        raise ValueError(f"Unsupported detection_mode: {detection_mode}")

    cnn_model, yolo_model, input_shape = _get_models()
    _, img_h, img_w, channels = input_shape

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open video.")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v") # codec used for output video
    fps = int(cap.get(cv2.CAP_PROP_FPS)) # get video fps
    fps = fps if fps > 0 else 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height)) # create output video writer

    print("[INFO] Processing video...")
    print(f"[INFO] {_mode_banner(mode)}")
    print("[INFO] Accident PREDICTION: YOLO + Trajectory Analysis")

    frame_count = 0 # counts total processed frames
    cnn_history = []
    yolo_collision_history = []
    last_cnn_prediction = 0.0
    cnn_accident_frames = 0

    vehicle_tracks = {}
    next_vehicle_id = 0

    try:
        while True:
            if should_stop and should_stop():
                raise RuntimeError("Processing was cancelled by user.")
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1

            cnn_smoothed = None
            cnn_confirmed = False
            if mode in {"cnn", "hybrid"}:
                if frame_count % CNN_FRAME_SKIP == 0:
                    resized = cv2.resize(frame, (img_w, img_h)) # resize frame to cnn input size
                    if channels == 1:
                        processed = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
                        processed = processed / 255.0
                        processed = np.expand_dims(processed, axis=-1)
                    else:
                        processed = resized / 255.0
                    cnn_input = np.expand_dims(processed, axis=0)
                    last_cnn_prediction = cnn_model.predict(cnn_input, verbose=0)[0][0]

                cnn_history.append(last_cnn_prediction)
                if len(cnn_history) > FRAME_HISTORY:
                    cnn_history.pop(0)
                cnn_smoothed = float(np.mean(cnn_history))
                cnn_current = cnn_smoothed > CNN_THRESHOLD
                cnn_accident_frames = cnn_accident_frames + 1 if cnn_current else 0
                cnn_confirmed = cnn_accident_frames >= CNN_CONSECUTIVE_THRESHOLD

            results = yolo_model(frame, verbose=False)
            vehicle_classes = [2, 3, 5, 7]
            vehicle_boxes = []
            for result in results:
                boxes = result.boxes
                if boxes is None:
                    continue
                for box in boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    if cls in vehicle_classes and conf > YOLO_CONFIDENCE:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        vehicle_boxes.append((x1, y1, x2, y2, conf, cls))

            prediction_warnings = []
            if PREDICTION_ENABLED and len(vehicle_boxes) >= 1:
                matches, unmatched = match_vehicles_to_tracks(vehicle_boxes, vehicle_tracks)
                updated_tracks = {}
                for track_id, box_idx in matches.items():
                    box = vehicle_boxes[box_idx]
                    center = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
                    if track_id in vehicle_tracks:
                        history = vehicle_tracks[track_id]["history"]
                        history.append((center[0], center[1], frame_count))
                        if len(history) > TRAJECTORY_HISTORY:
                            history.popleft()
                        updated_tracks[track_id] = {"history": history, "missed": 0}
                    else:
                        updated_tracks[track_id] = {
                            "history": deque([(center[0], center[1], frame_count)], maxlen=TRAJECTORY_HISTORY),
                            "missed": 0,
                        }

                for box_idx in unmatched:
                    box = vehicle_boxes[box_idx]
                    center = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
                    updated_tracks[next_vehicle_id] = {
                        "history": deque([(center[0], center[1], frame_count)], maxlen=TRAJECTORY_HISTORY),
                        "missed": 0,
                    }
                    next_vehicle_id += 1

                # Keep recently-lost tracks to avoid trajectory resets on short YOLO misses.
                for track_id, track_data in vehicle_tracks.items():
                    if track_id in updated_tracks:
                        continue
                    missed = track_data["missed"] + 1
                    if missed <= TRACK_MAX_MISSED_FRAMES:
                        updated_tracks[track_id] = {"history": track_data["history"], "missed": missed}

                vehicle_tracks = updated_tracks

                if len(vehicle_tracks) >= 2:
                    track_items = list(vehicle_tracks.items())
                    for i in range(len(track_items)):
                        for j in range(i + 1, len(track_items)):
                            track_id1, track_data1 = track_items[i]
                            track_id2, track_data2 = track_items[j]
                            if track_id1 not in matches or track_id2 not in matches:
                                continue
                            box1 = vehicle_boxes[matches[track_id1]]
                            box2 = vehicle_boxes[matches[track_id2]]
                            will_collide, ttc, confidence = predict_collision_course(
                                box1,
                                track_data1["history"],
                                box2,
                                track_data2["history"],
                                angle_threshold=COLLISION_COURSE_ANGLE,
                                speed_threshold=DANGEROUS_APPROACH_SPEED,
                            )
                            if will_collide:
                                prediction_warnings.append(
                                    {"box1": box1, "box2": box2, "ttc": ttc, "confidence": confidence}
                                )

            yolo_collision, collision_details = detect_vehicle_collisions(
                vehicle_boxes,
                iou_threshold=IOU_COLLISION_THRESHOLD,
                distance_threshold=PROXIMITY_DISTANCE,
            )
            yolo_collision_history.append(1 if yolo_collision else 0)
            if len(yolo_collision_history) > FRAME_HISTORY:
                yolo_collision_history.pop(0)
            yolo_collision_smoothed = bool(np.mean(yolo_collision_history) > 0.5)

            accident_confirmed = _compute_accident(mode, cnn_confirmed, yolo_collision_smoothed)
            prediction_active = len(prediction_warnings) > 0

            for box in vehicle_boxes:
                x1, y1, x2, y2, _, _ = box
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)

            # Keep motion visualization on the video (trajectory + direction arrow),
            # while metrics/text stay only in the frontend UI.
            if PREDICTION_ENABLED:
                for track_data in vehicle_tracks.values():
                    track = track_data["history"]
                    if len(track) > 1:
                        points = [(int(p[0]), int(p[1])) for p in track]
                        for idx in range(1, len(points)):
                            cv2.line(frame, points[idx - 1], points[idx], (255, 255, 0), 2)

                        if len(track) >= 3:
                            velocity, speed = calculate_velocity(track)
                            if speed > 5:
                                last_point = points[-1]
                                future_x = int(last_point[0] + velocity[0] * PREDICTION_HORIZON)
                                future_y = int(last_point[1] + velocity[1] * PREDICTION_HORIZON)
                                cv2.arrowedLine(
                                    frame,
                                    last_point,
                                    (future_x, future_y),
                                    (0, 255, 255),
                                    2,
                                    tipLength=0.3,
                                )

            for collision in collision_details:
                box1 = collision["box1"]
                box2 = collision["box2"]
                cv2.rectangle(frame, (int(box1[0]), int(box1[1])), (int(box1[2]), int(box1[3])), (0, 0, 255), 3)
                cv2.rectangle(frame, (int(box2[0]), int(box2[1])), (int(box2[2]), int(box2[3])), (0, 0, 255), 3)

            for warning in prediction_warnings:
                box1 = warning["box1"]
                box2 = warning["box2"]
                cv2.rectangle(frame, (int(box1[0]), int(box1[1])), (int(box1[2]), int(box1[3])), (0, 165, 255), 3)
                cv2.rectangle(frame, (int(box2[0]), int(box2[1])), (int(box2[2]), int(box2[3])), (0, 165, 255), 3)

            out.write(frame)

            if metrics_callback:
                yolo_score = yolo_collision_history.count(1) / max(1, len(yolo_collision_history))
                state = "accident" if accident_confirmed else ("warning" if prediction_active else "normal")
                metrics_callback(
                    {
                        "frame": int(frame_count),
                        "mode": mode,
                        "fps": float(fps),
                        "time_sec": float(frame_count / max(1, fps)),
                        "cnn_score": None if cnn_smoothed is None else float(cnn_smoothed),
                        "cnn_accident": None if mode == "yolo" else bool(cnn_confirmed),
                        "yolo_score": float(yolo_score),
                        "yolo_collision": bool(yolo_collision_smoothed),
                        "vehicles": int(len(vehicle_boxes)),
                        "prediction_pairs": int(len(prediction_warnings)),
                        "prediction_warning": bool(prediction_active),
                        "accident_detected": bool(accident_confirmed),
                        "state": state,
                    }
                )

            cv2.imshow("Accident Detection + Prediction", frame)
            try:
                if cv2.getWindowProperty("Accident Detection + Prediction", cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break
            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()
    print(f"[INFO] Done. Saved to: {output_path}")
    print(f"[INFO] Total frames processed: {frame_count}")
    return output_path


def run_cnn(video_path, output_path, metrics_callback=None, should_stop=None):
    return run_detector(
        video_path,
        output_path,
        detection_mode="cnn",
        metrics_callback=metrics_callback,
        should_stop=should_stop,
    )


def run_yolo(video_path, output_path, metrics_callback=None, should_stop=None):
    return run_detector(
        video_path,
        output_path,
        detection_mode="yolo",
        metrics_callback=metrics_callback,
        should_stop=should_stop,
    )


def run_cnn_yolo(video_path, output_path, metrics_callback=None, should_stop=None):
    return run_detector(
        video_path,
        output_path,
        detection_mode="hybrid",
        metrics_callback=metrics_callback,
        should_stop=should_stop,
    )
