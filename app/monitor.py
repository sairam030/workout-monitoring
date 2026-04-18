"""
Real-time Exercise Monitor
==========================
Launched by the exercise selector with a chosen exercise name.
Uses MediaPipe Pose + state-machine rep counting + form feedback.

Usage (internal):
    from app.monitor import run_monitor
    run_monitor("Squat")
"""

import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import time
from app.ui_helpers import draw_hud, GOOD_GREEN, BAD_RED


# ─── MediaPipe setup ──────────────────────────────────────────────────────────
mp_pose        = mp.solutions.pose
mp_drawing     = mp.solutions.drawing_utils
mp_draw_styles = mp.solutions.drawing_styles

LANDMARK = mp_pose.PoseLandmark


# ─── Utility ─────────────────────────────────────────────────────────────────

def _angle(a, b, c) -> float:
    """Return the angle at point b (in degrees) formed by a-b-c."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cos_val = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return float(np.degrees(np.arccos(np.clip(cos_val, -1.0, 1.0))))


def _lm(results, idx):
    """Return (x, y) of a landmark (normalized coords)."""
    lm = results.pose_landmarks.landmark[idx]
    return (lm.x, lm.y)


def _lmv(results, idx):
    """Return (x, y, visibility) of a landmark."""
    lm = results.pose_landmarks.landmark[idx]
    return (lm.x, lm.y, lm.visibility)


def _lm3(results, idx):
    """Return (x, y, z) of a landmark."""
    lm = results.pose_landmarks.landmark[idx]
    return (lm.x, lm.y, lm.z)


# ─── Exercise-specific logic ──────────────────────────────────────────────────

class ExerciseAnalyzer:
    """
    State-machine rep counter + form checker for a single chosen exercise.

    State machine:
      NEUTRAL → DOWN (when angle crosses down threshold)
      DOWN    → UP   (when angle crosses up threshold)   → rep counted
      UP      → NEUTRAL (brief reset)

    Counts one rep per DOWN→UP cycle to avoid double-counting.
    """

    def __init__(self, exercise: str):
        self.exercise = exercise
        self.state    = "NEUTRAL"   # NEUTRAL | DOWN | UP
        self.reps     = 0
        self._cooldown = 0          # frames to wait before accepting next rep

        # Precision controls
        self._ema_alpha = 0.35
        self._smoothed_angle = None
        self._hold_frames_required = 2
        self._down_hold = 0
        self._up_hold = 0
        self._cycle_min_angle = None
        self._cycle_max_angle = None

        # Thresholds per exercise
        self._cfg = {
            "Squat": {
                "joint": ("hip", "knee", "ankle"),
                "down_thresh": 105,    # knee angle ≤ this → DOWN
                "up_thresh":   160,    # knee angle ≥ this → UP (rep done)
                "signal": "min",       # angle goes DOWN during squat (decreases)
                "min_rom": 35,         # min range-of-motion per rep
            },
            "Push-up": {
                "joint": ("shoulder", "elbow", "wrist"),
                "down_thresh": 90,     # elbow angle ≤ 90° → DOWN
                "up_thresh":   155,    # elbow angle ≥ 155° → back UP
                "signal": "min",
                "min_rom": 30,
            },
            "Bicep Curl": {
                "joint": ("shoulder", "elbow", "wrist"),
                "down_thresh": 55,     # elbow angle ≤ 55° → curled UP (peak)
                "up_thresh":   150,    # elbow angle ≥ 150° → extended DOWN
                # Note: for curl, "DOWN" means arm extended, "UP" means curled
                "signal": "max",
                "min_rom": 40,
            },
        }

    def reset(self):
        """Reset session counters/state without recreating analyzer."""
        self.state = "NEUTRAL"
        self.reps = 0
        self._cooldown = 0
        self._smoothed_angle = None
        self._down_hold = 0
        self._up_hold = 0
        self._cycle_min_angle = None
        self._cycle_max_angle = None

    def _angle_from_triplet(self, results, a_idx, b_idx, c_idx):
        """Return (angle, min_visibility) for a single body side triplet."""
        a = _lmv(results, a_idx)
        b = _lmv(results, b_idx)
        c = _lmv(results, c_idx)
        angle = _angle((a[0], a[1]), (b[0], b[1]), (c[0], c[1]))
        min_vis = min(a[2], b[2], c[2])
        return angle, min_vis

    def _best_side_angle(self, results, left_triplet, right_triplet):
        """Choose left/right angle based on landmark visibility."""
        l_angle, l_vis = self._angle_from_triplet(results, *left_triplet)
        r_angle, r_vis = self._angle_from_triplet(results, *right_triplet)

        # If one side is significantly cleaner, trust that side.
        if l_vis - r_vis > 0.08:
            return l_angle
        if r_vis - l_vis > 0.08:
            return r_angle
        # Otherwise average both for stability.
        return (l_angle + r_angle) / 2.0

    def _smooth_angle(self, angle):
        """Exponential smoothing to reduce frame-to-frame jitter."""
        if angle is None:
            return None
        if self._smoothed_angle is None:
            self._smoothed_angle = angle
        else:
            a = self._ema_alpha
            self._smoothed_angle = a * angle + (1.0 - a) * self._smoothed_angle
        return self._smoothed_angle

    def _update_cycle_bounds(self, angle):
        if angle is None:
            return
        if self._cycle_min_angle is None or angle < self._cycle_min_angle:
            self._cycle_min_angle = angle
        if self._cycle_max_angle is None or angle > self._cycle_max_angle:
            self._cycle_max_angle = angle

    def _cycle_rom(self):
        if self._cycle_min_angle is None or self._cycle_max_angle is None:
            return 0.0
        return self._cycle_max_angle - self._cycle_min_angle

    def _start_new_cycle(self, seed_angle):
        self._cycle_min_angle = seed_angle
        self._cycle_max_angle = seed_angle

    def _get_key_angle(self, results) -> float | None:
        """Compute the primary joint angle for this exercise."""
        if not results.pose_landmarks:
            return None

        if self.exercise == "Squat":
            try:
                return self._best_side_angle(
                    results,
                    (LANDMARK.LEFT_HIP, LANDMARK.LEFT_KNEE, LANDMARK.LEFT_ANKLE),
                    (LANDMARK.RIGHT_HIP, LANDMARK.RIGHT_KNEE, LANDMARK.RIGHT_ANKLE),
                )
            except Exception:
                return None

        elif self.exercise == "Push-up":
            try:
                return self._best_side_angle(
                    results,
                    (LANDMARK.LEFT_SHOULDER, LANDMARK.LEFT_ELBOW, LANDMARK.LEFT_WRIST),
                    (LANDMARK.RIGHT_SHOULDER, LANDMARK.RIGHT_ELBOW, LANDMARK.RIGHT_WRIST),
                )
            except Exception:
                return None

        elif self.exercise == "Bicep Curl":
            try:
                # Curl defaults to right arm but falls back to the cleaner side.
                return self._best_side_angle(
                    results,
                    (LANDMARK.LEFT_SHOULDER, LANDMARK.LEFT_ELBOW, LANDMARK.LEFT_WRIST),
                    (LANDMARK.RIGHT_SHOULDER, LANDMARK.RIGHT_ELBOW, LANDMARK.RIGHT_WRIST),
                )
            except Exception:
                return None

        return None

    def update(self, results) -> tuple[int, bool, list, dict]:
        """
        Process a new frame's pose results.

        Returns:
            (rep_count, is_correct_form, feedback_messages, angle_info_dict)
        """
        raw_angle = self._get_key_angle(results)
        angle = self._smooth_angle(raw_angle)

        if self._cooldown > 0:
            self._cooldown -= 1

        # ── State machine ────────────────────────────────────────────────
        if angle is not None:
            cfg = self._cfg[self.exercise]
            dt  = cfg["down_thresh"]
            ut  = cfg["up_thresh"]
            sig = cfg["signal"]

            self._update_cycle_bounds(angle)

            if sig == "min":
                # angle decreases going DOWN
                self._down_hold = self._down_hold + 1 if angle <= dt else 0
                self._up_hold = self._up_hold + 1 if angle >= ut else 0

                if self.state == "NEUTRAL" and self._down_hold >= self._hold_frames_required:
                    self.state = "DOWN"
                    self._start_new_cycle(angle)
                elif self.state == "DOWN" and angle >= ut:
                    if self._up_hold >= self._hold_frames_required:
                        self.state = "UP"
                    if self._cooldown == 0 and self._cycle_rom() >= cfg["min_rom"]:
                        self.reps += 1
                        self._cooldown = 10
                        self._start_new_cycle(angle)
                elif self.state == "UP" and angle <= dt:
                    if self._down_hold >= self._hold_frames_required:
                        self.state = "DOWN"   # start next rep
                elif self.state == "UP" and angle < ut - 10:
                    self.state = "NEUTRAL"

            else:  # "max" — Bicep Curl
                # angle decreases when arm is curled (peak = low angle)
                self._down_hold = self._down_hold + 1 if angle >= ut else 0
                self._up_hold = self._up_hold + 1 if angle <= dt else 0

                if self.state == "NEUTRAL" and self._down_hold >= self._hold_frames_required:
                    self.state = "DOWN"   # arm extended
                    self._start_new_cycle(angle)
                elif self.state == "DOWN" and angle <= dt:
                    if self._up_hold >= self._hold_frames_required:
                        self.state = "UP"     # arm curled
                    if self._cooldown == 0 and self._cycle_rom() >= cfg["min_rom"]:
                        self.reps += 1
                        self._cooldown = 10
                        self._start_new_cycle(angle)
                elif self.state == "UP" and angle >= ut:
                    if self._down_hold >= self._hold_frames_required:
                        self.state = "DOWN"   # start next rep

        # ── Form check ───────────────────────────────────────────────────
        is_correct, messages = self._check_form(results, angle)

        # ── Angle info dict for HUD ───────────────────────────────────────
        if angle is not None:
            if self.exercise == "Squat":
                angle_info = {"Knee": angle}
            elif self.exercise == "Push-up":
                angle_info = {"Elbow": angle}
            else:
                angle_info = {"Elbow": angle}
        else:
            angle_info = {}

        return self.reps, is_correct, messages, angle_info

    # ── Per-exercise form checks ─────────────────────────────────────────

    def _check_form(self, results, primary_angle):
        if not results.pose_landmarks:
            return True, ["Stand in full view of the camera"]

        errors = []
        if self.exercise == "Squat":
            errors = self._squat_form(results, primary_angle)
        elif self.exercise == "Push-up":
            errors = self._pushup_form(results, primary_angle)
        elif self.exercise == "Bicep Curl":
            errors = self._curl_form(results, primary_angle)

        return len(errors) == 0, errors

    def _squat_form(self, results, knee_angle):
        errors = []
        try:
            # 1. Knee cave check
            l_knee  = _lm(results, LANDMARK.LEFT_KNEE)
            r_knee  = _lm(results, LANDMARK.RIGHT_KNEE)
            l_ankle = _lm(results, LANDMARK.LEFT_ANKLE)
            r_ankle = _lm(results, LANDMARK.RIGHT_ANKLE)

            if abs((l_knee[0] - l_ankle[0])) > 0.10:
                errors.append("Keep knees over toes")
            if abs((r_knee[0] - r_ankle[0])) > 0.10:
                errors.append("Keep knees over toes")

            # 2. Torso lean
            l_shoulder = _lm(results, LANDMARK.LEFT_SHOULDER)
            l_hip      = _lm(results, LANDMARK.LEFT_HIP)
            lean = abs(l_shoulder[0] - l_hip[0])
            if lean > 0.15:
                errors.append("Keep torso upright")

            # 3. Depth
            if self.state == "DOWN" and knee_angle is not None and knee_angle > 115:
                errors.append("Squat deeper — aim for 90°")

        except Exception:
            pass
        return errors

    def _pushup_form(self, results, elbow_angle):
        errors = []
        try:
            # 1. Body alignment (hip sag)
            l_shoulder = _lm(results, LANDMARK.LEFT_SHOULDER)
            l_hip      = _lm(results, LANDMARK.LEFT_HIP)
            l_ankle    = _lm(results, LANDMARK.LEFT_ANKLE)

            body_angle = _angle(l_shoulder, l_hip, l_ankle)
            if body_angle < 155:
                errors.append("Keep body in a straight line")

            # 2. Elbow depth in DOWN phase
            if self.state == "DOWN" and elbow_angle is not None and elbow_angle > 100:
                errors.append("Lower chest to 90° elbow angle")

            # 3. Elbow flare
            l_elbow    = _lm(results, LANDMARK.LEFT_ELBOW)
            r_elbow    = _lm(results, LANDMARK.RIGHT_ELBOW)
            l_shoulder = _lm(results, LANDMARK.LEFT_SHOULDER)
            r_shoulder = _lm(results, LANDMARK.RIGHT_SHOULDER)

            elbow_width    = abs(l_elbow[0]    - r_elbow[0])
            shoulder_width = abs(l_shoulder[0] - r_shoulder[0])
            if elbow_width > shoulder_width * 1.5:
                errors.append("Tuck elbows closer to body")

        except Exception:
            pass
        return errors

    def _curl_form(self, results, elbow_angle):
        errors = []
        try:
            r_elbow    = _lm(results, LANDMARK.RIGHT_ELBOW)
            r_hip      = _lm(results, LANDMARK.RIGHT_HIP)
            r_shoulder = _lm(results, LANDMARK.RIGHT_SHOULDER)
            r_wrist    = _lm(results, LANDMARK.RIGHT_WRIST)

            # 1. Elbow drift forward
            if abs(r_elbow[0] - r_hip[0]) < 0.04:
                errors.append("Keep elbow pinned to side")

            # 2. Shoulder movement
            if abs(r_shoulder[1] - r_hip[1]) < 0.15:
                errors.append("Don't shrug — keep shoulder down")

            # 3. Full extension at bottom
            if self.state == "DOWN" and elbow_angle is not None and elbow_angle < 140:
                errors.append("Fully extend arm at bottom")

            # 4. Full curl at top
            if self.state == "UP" and elbow_angle is not None and elbow_angle > 70:
                errors.append("Curl higher for full ROM")

        except Exception:
            pass
        return errors


# ─── Main run function ────────────────────────────────────────────────────────

def run_monitor(exercise: str, camera_id: int = 0):
    """
    Open webcam and run real-time monitoring for the given exercise.

    Args:
        exercise: One of "Squat", "Push-up", "Bicep Curl"
        camera_id: OpenCV camera index (default 0)
    """
    analyzer = ExerciseAnalyzer(exercise)
    window_name = f"Workout Monitor - {exercise}"

    pose = mp_pose.Pose(
        min_detection_confidence=0.65,
        min_tracking_confidence=0.65,
        model_complexity=1,
    )

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"[ERROR] Could not open camera {camera_id}")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    def _fit_frame_for_display(img, max_w=1280, max_h=720):
        ih, iw = img.shape[:2]
        if iw <= max_w and ih <= max_h:
            return img
        scale = min(max_w / float(iw), max_h / float(ih))
        nw, nh = int(iw * scale), int(ih * scale)
        return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

    fps_buf = deque(maxlen=30)
    print(f"\n▶  Monitoring: {exercise}  |  Q = quit   R = reset reps\n")

    while cap.isOpened():
        t0 = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)                      # mirror view
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = pose.process(rgb)
        rgb.flags.writeable = True

        # ── Pose landmarks overlay ────────────────────────────────────────
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_draw_styles.get_default_pose_landmarks_style(),
            )

        # ── Update analyzer ───────────────────────────────────────────────
        reps, is_correct, messages, angle_info = analyzer.update(results)

        # ── FPS ───────────────────────────────────────────────────────────
        elapsed = time.time() - t0
        fps_buf.append(1.0 / elapsed if elapsed > 0 else 0)
        fps = float(np.mean(fps_buf))

        # ── Draw HUD ──────────────────────────────────────────────────────
        frame = draw_hud(frame, exercise, reps, is_correct,
                         messages, angle_info, fps)

        display_frame = _fit_frame_for_display(frame)
        cv2.imshow(window_name, display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('r'):
            analyzer.reset()
            print("  ↺  Rep count reset to 0")

    cap.release()
    cv2.destroyAllWindows()
    pose.close()

    print(f"\n── Session Summary ────────────────────────────────")
    print(f"   Exercise : {exercise}")
    print(f"   Total Reps : {analyzer.reps}")
    print(f"   Avg FPS  : {np.mean(fps_buf):.1f}")
    print(f"───────────────────────────────────────────────────\n")
