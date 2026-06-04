# Squat Pose Estimation & Quality Classification

An end-to-end Machine Learning and Computer Vision pipeline designed to analyze squat performance, segment individual repetitions, and classify form quality (e.g., correct form, shallow depth, or excessive forward lean) in real-time or from recorded videos.

The project uses **Google MediaPipe Pose Landmarker** to extract 3D skeletal landmarks and **Scikit-Learn (Random Forest)** to perform classification based on biomechanical angles and ratios.

---

## 🌟 Key Features

1. **Pose Extraction (`src/extract_pose.py`)**:
   - Captures 33 key body landmarks at each frame from raw squat videos.
   - Saves frame-by-frame coordinate data to CSV.

2. **Feature Engineering (`src/features.py`)**:
   - Calculates biomechanical features, including:
     - Left and Right Knee Angles.
     - Left and Right Hip Angles.
     - Torso Lean Angles (relative to vertical).
     - Hip Depth (hips height relative to knees and ankles).
   - Applies a rolling window average (smoothing) to reduce noise from pose detection.

3. **Repetition Segmentation (`src/rep_dataset.py`)**:
   - Automatically detects and isolates individual reps by tracking when knee angles cross UP and DOWN thresholds.
   - Extracts summary metrics per rep (e.g., duration, angle ranges, minimums/maximums) for training.

4. **Model Training (`src/train_classifier.py` and `src/train_rep_classifier.py`)**:
   - Trains Random Forest classifiers using either aggregated video features or segmented rep-level features.
   - Evaluates performance using Group K-Fold cross-validation (Leave-One-Video-Out) to ensure generalizability.

5. **Real-time Inference (`src/infer_realtime.py`)**:
   - Connects to a webcam or opens a video file.
   - Tracks skeletal landmarks and displays feedback overlay (reps count, correct vs. incorrect classification, warning alerts for shallow depth or forward lean).
   - Logs performance to a CSV file (`data/training/rep_inference_log.csv`) for future retraining.

---

## 🛠️ Local Setup Instructions

### 1. Prerequisites
- **Python**: Make sure Python 3.8 to 3.11 is installed.
- **Git**: Ensure Git is installed on your system.

### 2. Environment Setup
Clone or navigate to the project directory, then create and activate a virtual environment:

#### Windows (PowerShell):
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

#### macOS/Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
Install all required libraries using the provided `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 4. Pose Landmarker Model
The project relies on Google's pre-trained MediaPipe Pose Landmarker task file.
- The model file must be placed at: `models/pose_landmarker.task`
- If it is missing, download it from [Google MediaPipe Pose Landmarker Guide](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker#models) and place it in the `models/` directory.

---

## 🏃 Running the Pipeline

### Step 1: Prepare Training Videos
Place your training videos under `data/raw_videos/` grouped by their class labels:
- `data/raw_videos/correct/` (correct squats)
- `data/raw_videos/shallow/` (squats with shallow depth)
- `data/raw_videos/forward_lean/` (squats with excessive forward lean)

### Step 2: Extract Landmarks
Extract body coordinates from your videos:
```bash
python src/extract_pose.py
```
*Outputs: `data/extracted_landmarks/<class>/<video>_landmarks.csv`*

### Step 3: Compute Features
Convert raw coordinates into smoothed joint angles and depths:
```bash
python src/features.py
```
*Outputs: `data/features/<class>/<video>_features.csv`*

### Step 4: Train Classifiers
Train the machine learning models on your extracted features:
```bash
# Train video-level/frame-level classifier
python src/train_classifier.py

# Train repetition-level classifier
python src/train_rep_classifier.py
```
*Outputs: Models saved in `models/` (e.g., `squat_rep_classifier.pkl`)*

### Step 5: Run Real-time Inference
To test the models using your webcam:
```bash
python src/infer_realtime.py
```

To run it on a specific video file:
```bash
python src/infer_realtime.py --video path/to/your/video.mp4
```
*Press `q` to exit the video display window.*

---

## 🚀 GitHub Upload Instructions

To push this repository to GitHub, follow these step-by-step instructions.

### 1. Initialize Git Repo
If you haven't initialized Git in this folder yet, run:
```bash
git init
```

### 2. Verify `.gitignore`
Make sure you have a `.gitignore` file in your root folder. This prevents committing large local environments or temporary files. A standard `.gitignore` has been created for you, which ignores:
- Virtual environments (`.venv/`)
- Temporary CSV data exports (`data/extracted_landmarks/`, `data/features/`, `data/training/`)
- Local caches (`__pycache__/`)
- Large generated model files (`models/*.pkl`)

### 3. Stage and Commit Files
Add the codebase to your Git staging area and make your initial commit:
```bash
git add .
git commit -m "Initial commit: Squat Pose Estimation pipeline"
```

### 4. Link to GitHub and Push
1. Go to [GitHub](https://github.com/) and create a new repository (do **not** check "Initialize this repository with a README" since we already have one).
2. Copy your repository's URL (looks like `https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git`).
3. Run the following commands in your terminal (replacing the URL with yours):
   ```bash
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
   git push -u origin main
   ```
