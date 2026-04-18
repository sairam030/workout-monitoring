# 💪 AI Workout Monitoring System

Real-time exercise monitoring using **MediaPipe Pose** + **OpenCV**.  
Detects form correctness and counts reps for 3 exercises.

---

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Launch the app
python main.py
```

A GUI window will open. Select an exercise and click **▶ Start**.

---

## 🏋️ Supported Exercises

| Exercise | Tracked Joint | Rep Signal |
|---|---|---|
| **Squat** | Knee angle | < 105° → down, > 160° → rep counted |
| **Push-up** | Elbow angle | < 90° → down, > 155° → rep counted |
| **Bicep Curl** | Elbow angle | < 55° → curled, > 150° → rep counted |

---

## 📁 Project Structure

```
workout-monitoring/
├── main.py                     ← Entry point: python main.py
│
├── app/
│   ├── exercise_selector.py    ← Tkinter GUI — pick your exercise
│   ├── monitor.py              ← Real-time monitoring engine
│   └── ui_helpers.py           ← HUD drawing utilities
│
├── notebooks/
│   ├── 00_mediapipe_landmark_training.ipynb  ← Tutorial: landmarks → training
│   ├── 01_data_collection.ipynb
│   ├── 02_eda_detailed.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_data_balancing.ipynb
│   ├── 05_model_training.ipynb
│   └── 06_model_evaluation.ipynb
│
├── models/                     ← Pre-trained model files
├── data/                       ← Raw & processed pose data
├── src/                        ← Data collection utilities
├── archive/                    ← Old scripts (not needed to run)
└── requirements.txt
```

---

## 📓 Notebooks

Start with **`notebooks/00_mediapipe_landmark_training.ipynb`** to understand:

1. How MediaPipe detects 33 body landmarks
2. Feature engineering (joint angles, distances)
3. Building a labelled dataset
4. Training Random Forest + LSTM classifiers
5. Evaluating model performance

Then explore notebooks 01–06 for the full data pipeline.

---

## ⌨️ Controls (inside monitoring window)

| Key | Action |
|---|---|
| `Q` | Quit and return to selector |
| `R` | Reset rep count to 0 |

---

## 🛠️ How It Works

```
Camera Frame
    ↓
MediaPipe Pose → 33 landmarks (x, y, z, visibility)
    ↓
Feature Engineering → joint angles + normalized distances
    ↓
State Machine Rep Counter
  NEUTRAL → angle crosses DOWN threshold → DOWN
  DOWN    → angle crosses UP threshold   → COUNT REP
    ↓
Form Checker → biomechanical angle rules → feedback messages
    ↓
HUD Overlay → reps, form status, joint angles, skeleton
```

---

## Tech Stack

- **Pose Estimation**: MediaPipe BlazePose
- **Computer Vision**: OpenCV
- **GUI**: Tkinter (built-in Python)
- **ML / Training**: scikit-learn, TensorFlow/Keras
- **Notebooks**: Jupyter
