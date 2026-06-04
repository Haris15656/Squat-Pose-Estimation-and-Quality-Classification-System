import cv2
import mediapipe as mp
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

LANDMARK_NAMES = [
    "nose",
    "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear",
    "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_pinky", "right_pinky",
    "left_index", "right_index",
    "left_thumb", "right_thumb",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "left_heel", "right_heel",
    "left_foot_index", "right_foot_index"
]

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}

def extract_pose_landmarks(video_path, output_csv, class_label, model_path):
    print(f"Processing: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Could not open video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0

    rows = []

    try:
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=VisionRunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.3,
            min_pose_presence_confidence=0.3,
            min_tracking_confidence=0.3,
            output_segmentation_masks=False
        )

        with PoseLandmarker.create_from_options(options) as landmarker:
            frame_idx = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp_ms = int((frame_idx / fps) * 1000)

                result = landmarker.detect_for_video(mp_image, timestamp_ms)

                row = {
                    "frame": frame_idx,
                    "timestamp_sec": frame_idx / fps,
                    "video_path": str(video_path),
                    "video_name": video_path.stem,
                    "class_label": class_label,
                    "pose_detected": 0
                }

                if result.pose_landmarks and len(result.pose_landmarks) > 0:
                    row["pose_detected"] = 1
                    landmarks = result.pose_landmarks[0]
                    for i, lm in enumerate(landmarks):
                        name = LANDMARK_NAMES[i]
                        row[f"{name}_x"] = lm.x
                        row[f"{name}_y"] = lm.y
                        row[f"{name}_z"] = lm.z
                        row[f"{name}_visibility"] = getattr(lm, "visibility", None)
                        row[f"{name}_presence"] = getattr(lm, "presence", None)
                else:
                    for name in LANDMARK_NAMES:
                        row[f"{name}_x"] = None
                        row[f"{name}_y"] = None
                        row[f"{name}_z"] = None
                        row[f"{name}_visibility"] = None
                        row[f"{name}_presence"] = None

                rows.append(row)
                frame_idx += 1

                if frame_idx % 30 == 0:
                    print(f"Processed {frame_idx} frames")

    except Exception as e:
        print(f"Extraction error for {video_path}: {e}")
        cap.release()
        return

    cap.release()

    if not rows:
        print(f"No rows generated for {video_path}")
        return

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"Saved landmarks: {output_csv} ({len(df)} frames)")

def process_dataset(
    raw_root=PROJECT_ROOT / "data" / "raw_videos",
    output_root=PROJECT_ROOT / "data" / "extracted_landmarks",
    model_path=PROJECT_ROOT / "models" / "pose_landmarker.task"
):
    raw_root = Path(raw_root)
    output_root = Path(output_root)
    model_path = Path(model_path)

    print("Current working dir:", Path.cwd())
    print("Raw root exists:", raw_root.exists())
    print("Raw root absolute:", raw_root.resolve())
    print("Model exists:", model_path.exists())
    print("Model absolute:", model_path.resolve())

    all_files = list(raw_root.rglob("*")) if raw_root.exists() else []
    print("All found paths:")
    for p in all_files[:100]:
        print(" -", p, "| suffix:", p.suffix)

    video_files = [p for p in all_files if p.suffix.lower() in VIDEO_EXTENSIONS]

    print("Matched video files:", len(video_files))
    for v in video_files:
        print(" ->", v)

    if not video_files:
        print(f"No video files found in {raw_root}")
        return

    for video_path in video_files:
        class_label = video_path.parent.name
        output_csv = output_root / class_label / f"{video_path.stem}_landmarks.csv"
        extract_pose_landmarks(video_path, output_csv, class_label, model_path)

if __name__ == "__main__":
    process_dataset()