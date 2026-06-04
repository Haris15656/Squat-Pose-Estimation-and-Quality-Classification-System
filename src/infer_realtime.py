import cv2
import mediapipe as mp
import joblib
import numpy as np
import pandas as pd
from collections import deque
from pathlib import Path
import argparse

from rep_dataset import summarize_rep_frames, rep_model_feature_columns

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REP_LOG_PATH = PROJECT_ROOT / "data" / "training" / "rep_inference_log.csv"
REP_MODEL_PATH = PROJECT_ROOT / "models" / "squat_rep_classifier.pkl"
REALTIME_MODEL_PATH = PROJECT_ROOT / "models" / "squat_classifier_realtime.pkl"
MODEL_PATH = PROJECT_ROOT / "models" / "squat_classifier.pkl"
POSE_MODEL_PATH = PROJECT_ROOT / "models" / "pose_landmarker.task"

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

if REP_MODEL_PATH.exists():
    rep_model = joblib.load(REP_MODEL_PATH)
    active_rep_model_path = REP_MODEL_PATH
else:
    rep_model = None
    active_rep_model_path = None

if REALTIME_MODEL_PATH.exists():
    model = joblib.load(REALTIME_MODEL_PATH)
    active_model_path = REALTIME_MODEL_PATH
else:
    model = joblib.load(MODEL_PATH)
    active_model_path = MODEL_PATH

FEATURE_KEYS = [
    "left_knee_angle",
    "right_knee_angle",
    "avg_knee_angle",
    "left_hip_angle",
    "right_hip_angle",
    "avg_hip_angle",
    "torso_angle_left",
    "torso_angle_right",
    "avg_torso_angle",
    "hip_depth",
    "avg_knee_angle_smooth",
    "avg_hip_angle_smooth",
    "avg_torso_angle_smooth",
    "hip_depth_smooth",
]

POSE_CONNECTIONS = [
    (11, 12), (11, 23), (12, 24),
    (23, 24), (23, 25), (24, 26),
    (25, 27), (26, 28), (11, 25), (12, 26),
]

# Runtime thresholds and gating
CONF_THRESHOLD = 0.35
MIN_DOWN_TIME_MS = 150
MIN_REP_INTERVAL_MS = 500

def angle_2d(a, b, c):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    c = np.array(c, dtype=float)
    ba = a - b
    bc = c - b
    n1 = np.linalg.norm(ba)
    n2 = np.linalg.norm(bc)
    if n1 == 0 or n2 == 0:
        return np.nan
    cosine = np.dot(ba, bc) / (n1 * n2)
    cosine = np.clip(cosine, -1.0, 1.0)
    return np.degrees(np.arccos(cosine))

def angle_to_vertical(top, bottom):
    top = np.array(top, dtype=float)
    bottom = np.array(bottom, dtype=float)
    vec = top - bottom
    vertical = np.array([0.0, -1.0])
    n = np.linalg.norm(vec)
    if n == 0:
        return np.nan
    cosine = np.dot(vec, vertical) / n
    cosine = np.clip(cosine, -1.0, 1.0)
    return np.degrees(np.arccos(cosine))

def midpoint(p1, p2):
    return ((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0)

def get_xy(landmarks, idx):
    return (landmarks[idx].x, landmarks[idx].y)

def visibility_ok(landmarks, indices, thresh=0.5):
    return all(getattr(landmarks[i], "visibility", 0.0) > thresh for i in indices)


def classify_rep_frames(rep_frames, rep_model):
    rep_summary = summarize_rep_frames(rep_frames)
    rep_features = rep_model_feature_columns(list(rep_model.feature_names_in_))
    rep_row = {feature_name: rep_summary.get(feature_name, 0.0) for feature_name in rep_features}
    rep_X = pd.DataFrame([rep_row], columns=rep_features).fillna(0)
    rep_pred = rep_model.predict(rep_X)[0]
    rep_conf = float(rep_model.predict_proba(rep_X).max()) if hasattr(rep_model, "predict_proba") else 0.0
    return rep_pred, rep_conf, rep_summary, rep_row


def append_rep_log(rep_row: dict) -> None:
    REP_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    log_exists = REP_LOG_PATH.exists()
    pd.DataFrame([rep_row]).to_csv(REP_LOG_PATH, mode="a", header=not log_exists, index=False)

def compute_feature_dict(landmarks):
    left_shoulder = get_xy(landmarks, 11)
    right_shoulder = get_xy(landmarks, 12)
    left_hip = get_xy(landmarks, 23)
    right_hip = get_xy(landmarks, 24)
    left_knee = get_xy(landmarks, 25)
    right_knee = get_xy(landmarks, 26)
    left_ankle = get_xy(landmarks, 27)
    right_ankle = get_xy(landmarks, 28)

    left_knee_angle = angle_2d(left_hip, left_knee, left_ankle)
    right_knee_angle = angle_2d(right_hip, right_knee, right_ankle)
    left_hip_angle = angle_2d(left_shoulder, left_hip, left_knee)
    right_hip_angle = angle_2d(right_shoulder, right_hip, right_knee)
    torso_angle_left = angle_to_vertical(left_shoulder, left_hip)
    torso_angle_right = angle_to_vertical(right_shoulder, right_hip)

    hip_mid = midpoint(left_hip, right_hip)
    knee_mid = midpoint(left_knee, right_knee)
    ankle_mid = midpoint(left_ankle, right_ankle)
    hip_depth = (ankle_mid[1] - hip_mid[1]) / max(abs(ankle_mid[1] - knee_mid[1]), 1e-6)

    return {
        "left_knee_angle": left_knee_angle,
        "right_knee_angle": right_knee_angle,
        "avg_knee_angle": np.nanmean([left_knee_angle, right_knee_angle]),
        "left_hip_angle": left_hip_angle,
        "right_hip_angle": right_hip_angle,
        "avg_hip_angle": np.nanmean([left_hip_angle, right_hip_angle]),
        "torso_angle_left": torso_angle_left,
        "torso_angle_right": torso_angle_right,
        "avg_torso_angle": np.nanmean([torso_angle_left, torso_angle_right]),
        "hip_depth": hip_depth,
    }

pose_options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=str(POSE_MODEL_PATH)),
    running_mode=VisionRunningMode.VIDEO,
    num_poses=1,
    min_pose_detection_confidence=0.3,
    min_pose_presence_confidence=0.3,
    min_tracking_confidence=0.3,
    output_segmentation_masks=False,
)

# CLI: allow running on a video file instead of webcam
parser = argparse.ArgumentParser(description="Run squat pose model on webcam or video file")
parser.add_argument("--video", "-v", type=str, default=None, help="Path to input video file. If omitted, uses webcam.")
args = parser.parse_args()

input_source = args.video or "webcam"

if args.video:
    cap = cv2.VideoCapture(str(args.video))
    is_webcam = False
else:
    cap = cv2.VideoCapture(0)
    is_webcam = True

if not cap.isOpened():
    raise SystemExit(f"Could not open capture source: {args.video or 'webcam'}")

# Determine FPS for timestamp calculations
fps = cap.get(cv2.CAP_PROP_FPS)
if not fps or fps <= 0:
    fps = 30.0

frame_idx = 0

rep_count = 0
correct_rep_count = 0
incorrect_rep_count = 0
shallow_rep_count = 0
forward_lean_rep_count = 0
state = "up"
status = "No body detected"
feedback = "Step into frame"

knee_hist = deque(maxlen=5)
hip_hist = deque(maxlen=5)
torso_hist = deque(maxlen=5)
depth_hist = deque(maxlen=5)

down_since_ms = None
last_counted_ms = -999999
# Track extrema while in the down state for correct/incorrect evaluation
down_min_knee = None
down_max_torso = None
current_rep_frames = []
current_rep_start_frame = None
current_rep_start_time_ms = None


def draw_body_markers(frame, landmarks, thickness=2):
    height, width = frame.shape[:2]

    points = {}
    for idx, lm in enumerate(landmarks):
        x = int(lm.x * width)
        y = int(lm.y * height)
        points[idx] = (x, y)

        if 0 <= x < width and 0 <= y < height:
            visibility = getattr(lm, "visibility", 1.0)
            color = (0, 255, 0) if visibility >= 0.5 else (0, 165, 255)
            cv2.circle(frame, (x, y), 4, color, -1)

    for start_idx, end_idx in POSE_CONNECTIONS:
        if start_idx in points and end_idx in points:
            cv2.line(frame, points[start_idx], points[end_idx], (255, 255, 255), thickness)

    return frame

with PoseLandmarker.create_from_options(pose_options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if is_webcam:
            frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        timestamp_ms = int((frame_idx / fps) * 1000)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)
        frame_idx += 1

        if not result.pose_landmarks:
            status = "No body detected"
            feedback = "Step into frame"
            down_since_ms = None
            current_rep_frames = []
            current_rep_start_frame = None
            current_rep_start_time_ms = None
            cv2.putText(frame, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            cv2.putText(frame, feedback, (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            landmarks = result.pose_landmarks[0]
            draw_body_markers(frame, landmarks)

            left_side = [11, 23, 25, 27]
            right_side = [12, 24, 26, 28]
            use_left = visibility_ok(landmarks, left_side)
            use_right = visibility_ok(landmarks, right_side)

            if not use_left and not use_right:
                status = "Pose unreliable"
                feedback = "Align body in frame"
                down_since_ms = None
                current_rep_frames = []
                current_rep_start_frame = None
                current_rep_start_time_ms = None
                cv2.putText(frame, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 2)
                cv2.putText(frame, feedback, (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
            else:
                feats = compute_feature_dict(landmarks)

                knee_hist.append(feats["avg_knee_angle"])
                hip_hist.append(feats["avg_hip_angle"])
                torso_hist.append(feats["avg_torso_angle"])
                depth_hist.append(feats["hip_depth"])

                feats["avg_knee_angle_smooth"] = float(np.nanmean(knee_hist))
                feats["avg_hip_angle_smooth"] = float(np.nanmean(hip_hist))
                feats["avg_torso_angle_smooth"] = float(np.nanmean(torso_hist))
                feats["hip_depth_smooth"] = float(np.nanmean(depth_hist))

                # Build input matching the model's training feature names
                model_features = list(getattr(model, "feature_names_in_", FEATURE_KEYS))
                row = {}
                for fn in model_features:
                    if fn in feats:
                        row[fn] = feats[fn]
                    elif fn == "pose_detected":
                        row[fn] = 1
                    else:
                        row[fn] = 0

                X = pd.DataFrame([row], columns=model_features).fillna(0)
                pred = model.predict(X)[0]
                conf = float(model.predict_proba(X).max()) if hasattr(model, "predict_proba") else 0.0

                knee = feats["avg_knee_angle_smooth"]
                torso = feats["avg_torso_angle_smooth"]
                depth = feats["hip_depth_smooth"]

                UP_THRESHOLD = 150
                DOWN_THRESHOLD = 120
                # Smaller hip_depth means the hips dropped lower relative to the knees/ankles.
                GOOD_DEPTH_MAX_RATIO = 1.5
                GOOD_TORSO_MIN = 55
                SHALLOW_DEPTH_MIN_RATIO = 1.5

                if knee > UP_THRESHOLD:
                    if state == "down":
                        down_time = (timestamp_ms - down_since_ms) if down_since_ms is not None else 0
                        since_last = timestamp_ms - last_counted_ms
                        if down_since_ms is not None and down_time >= MIN_DOWN_TIME_MS and since_last >= MIN_REP_INTERVAL_MS and conf >= CONF_THRESHOLD:
                            rep_count += 1
                            last_counted_ms = timestamp_ms
                            if rep_model is not None and current_rep_frames:
                                rep_pred, rep_conf, rep_summary, rep_features = classify_rep_frames(current_rep_frames, rep_model)
                                rep_log_row = {
                                    "source": input_source,
                                    "model_path": str(active_rep_model_path or active_model_path),
                                    "verdict_source": "rep_model",
                                    "rep_index": rep_count,
                                    "start_frame": int(current_rep_start_frame if current_rep_start_frame is not None else current_rep_frames[0].get("frame", 0)),
                                    "end_frame": int(current_rep_frames[-1].get("frame", 0)),
                                    "start_time_sec": float(current_rep_start_time_ms / 1000.0 if current_rep_start_time_ms is not None else current_rep_frames[0].get("timestamp_sec", 0.0)),
                                    "end_time_sec": float(current_rep_frames[-1].get("timestamp_sec", 0.0)),
                                    "duration_sec": float(rep_summary.get("duration_sec", 0.0)),
                                    "n_frames": int(rep_summary.get("n_frames", len(current_rep_frames))),
                                    "predicted_label": rep_pred,
                                    "predicted_confidence": rep_conf,
                                    "manual_label": "",
                                }
                                rep_log_row.update(rep_features)
                                append_rep_log(rep_log_row)
                                rep_depth_mean = float(rep_features.get("hip_depth_smooth_mean", rep_summary.get("hip_depth_smooth_mean", rep_summary.get("hip_depth_mean", 0.0))))
                                if rep_pred == "correct" and rep_depth_mean <= SHALLOW_DEPTH_MIN_RATIO:
                                    correct_rep_count += 1
                                    feedback = "✓ Correct form"
                                    print(f"[REP_MODEL] frames={len(current_rep_frames)} pred={rep_pred} conf={rep_conf:.2f} -> CORRECT")
                                else:
                                    incorrect_rep_count += 1
                                    if rep_pred == "shallow" or rep_depth_mean > SHALLOW_DEPTH_MIN_RATIO:
                                        shallow_rep_count += 1
                                        feedback = "⚠ Incorrect form: shallow depth"
                                    elif rep_pred == "forward_lean":
                                        forward_lean_rep_count += 1
                                        feedback = "⚠ Incorrect form: forward lean"
                                    else:
                                        feedback = f"⚠ Incorrect form: {rep_pred}"
                                    print(f"[REP_MODEL] frames={len(current_rep_frames)} pred={rep_pred} conf={rep_conf:.2f} -> INCORRECT")
                            else:
                                # Use extrema recorded during the down phase as a fallback.
                                min_knee = down_min_knee if down_min_knee is not None else knee
                                min_depth = down_min_depth if down_min_depth is not None else depth
                                max_torso = down_max_torso if down_max_torso is not None else torso
                                good_depth = (min_depth <= GOOD_DEPTH_MAX_RATIO)
                                good_torso = (max_torso >= GOOD_TORSO_MIN)
                                if good_depth and good_torso:
                                    correct_rep_count += 1
                                    feedback = "Correct rep"
                                    print(f"[REP] time={timestamp_ms}ms frame={frame_idx} min_knee={min_knee:.1f} min_depth={min_depth:.2f} max_torso={max_torso:.1f} depth={depth:.2f} pred={pred} conf={conf:.2f} -> CORRECT")
                                else:
                                    incorrect_rep_count += 1
                                    if not good_depth and good_torso:
                                        shallow_rep_count += 1
                                        feedback = "⚠ Incorrect form: shallow depth"
                                    elif good_depth and not good_torso:
                                        forward_lean_rep_count += 1
                                        feedback = "⚠ Incorrect form: forward lean"
                                    else:
                                        feedback = "⚠ Incorrect form"
                                    print(f"[REP] time={timestamp_ms}ms frame={frame_idx} min_knee={min_knee:.1f} min_depth={min_depth:.2f} max_torso={max_torso:.1f} depth={depth:.2f} pred={pred} conf={conf:.2f} -> INCORRECT")
                            # reset down extrema after counting
                            down_min_knee = None
                            down_min_depth = None
                            down_max_torso = None
                            current_rep_frames = []
                            current_rep_start_frame = None
                            current_rep_start_time_ms = None
                        else:
                            feedback = "Hold lower or increase confidence"
                    state = "up"
                    down_since_ms = None
                    status = "Standing"
                elif knee < DOWN_THRESHOLD:
                    if state != "down":
                        down_since_ms = timestamp_ms
                        current_rep_frames = []
                        current_rep_start_frame = frame_idx
                        current_rep_start_time_ms = timestamp_ms
                        # initialize extrema trackers when entering down
                        down_min_knee = knee
                        down_min_depth = depth
                        down_max_torso = torso
                    else:
                        # update extrema while held down
                        if down_min_knee is None:
                            down_min_knee = knee
                        else:
                            down_min_knee = min(down_min_knee, knee)
                        if down_min_depth is None:
                            down_min_depth = depth
                        else:
                            down_min_depth = min(down_min_depth, depth)
                        if down_max_torso is None:
                            down_max_torso = torso
                        else:
                            down_max_torso = max(down_max_torso, torso)
                    current_rep_frames.append({
                        **feats,
                        "frame": frame_idx,
                        "timestamp_sec": timestamp_ms / 1000.0,
                        "pose_detected": 1,
                    })
                    state = "down"
                    # show status based on extrema so far
                    status = "Squat: good form" if ((down_min_depth is not None and down_min_depth <= GOOD_DEPTH_MAX_RATIO) and (down_max_torso is not None and down_max_torso >= GOOD_TORSO_MIN)) else "Squat: wrong form"
                else:
                    status = "Squat in progress"

                cv2.putText(frame, f"Frame class: {pred} ({conf:.1%})", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(frame, f"Reps: {rep_count}", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                cv2.putText(frame, f"Correct: {correct_rep_count}", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(frame, f"Incorrect: {incorrect_rep_count}", (20, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.putText(frame, f"Shallow: {shallow_rep_count}", (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
                cv2.putText(frame, f"Forward lean: {forward_lean_rep_count}", (20, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 140, 255), 2)
                cv2.putText(frame, f"State: {state}", (20, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2)
                cv2.putText(frame, f"Status: {status}", (20, 285), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                cv2.putText(frame, f"Alert: {feedback}", (20, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
                cv2.putText(frame, f"Knee: {knee:.1f}  Torso: {torso:.1f}  Depth: {depth:.2f}", (20, 355), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Squat Trainer", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()

print("\n=== Inference Summary ===")
print(f"Source: {args.video or 'webcam'}")
print(f"Model: {active_model_path}")
print(f"Total reps: {rep_count}")
print(f"Correct reps: {correct_rep_count}")
print(f"Incorrect reps: {incorrect_rep_count}")
print(f"Shallow reps: {shallow_rep_count}")
print(f"Forward lean reps: {forward_lean_rep_count}")