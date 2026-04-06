"""
Real-time Workout Monitoring System
===================================

Integrates all components for live exercise monitoring:
1. Camera feed → Pose estimation
2. Exercise classification
3. Form correctness assessment
4. Repetition counting
5. Visual feedback

Performance: < 100ms latency per frame
"""

import cv2
import mediapipe as mp
import numpy as np
from collections import deque
from scipy.signal import find_peaks
import time


class WorkoutMonitor:
    """
    Real-time workout monitoring system
    
    Pipeline:
    Camera → MediaPipe Pose → Features → Classification → Feedback
    """
    
    def __init__(self, model_path=None, buffer_size=30):
        """
        Initialize workout monitor
        
        Args:
            model_path: Path to trained model (optional)
            buffer_size: Number of frames to buffer (1 second @ 30fps)
        """
        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            model_complexity=1  # 0=Lite, 1=Full, 2=Heavy
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Frame buffer for temporal analysis
        self.buffer_size = buffer_size
        self.landmark_buffer = deque(maxlen=buffer_size)
        
        # State tracking
        self.current_exercise = "Unknown"
        self.is_correct_form = True
        self.rep_count = 0
        self.feedback_messages = []
        
        # Exercise-specific state
        self.last_peak_frame = -999
        self.in_rep = False
        self.rep_phase = "neutral"  # "down", "up", "neutral"
        
        # Load model (placeholder - implement based on your trained model)
        self.model = None
        if model_path:
            self.load_model(model_path)
        
        # Exercise detection thresholds (for demo without trained model)
        self.demo_mode = True
    
    def load_model(self, model_path):
        """Load trained exercise classification model"""
        try:
            from tensorflow import keras
            self.model = keras.models.load_model(model_path)
            self.demo_mode = False
            print(f"✓ Model loaded from {model_path}")
        except Exception as e:
            print(f"⚠ Could not load model: {e}")
            print("Running in demo mode with rule-based detection")
            self.demo_mode = True
    
    def extract_landmarks(self, results):
        """Extract 132 features from MediaPipe results"""
        if not results.pose_landmarks:
            return None
        
        landmarks = []
        for landmark in results.pose_landmarks.landmark:
            landmarks.extend([
                landmark.x,
                landmark.y,
                landmark.z,
                landmark.visibility
            ])
        return np.array(landmarks)
    
    def calculate_angle(self, p1, p2, p3):
        """Calculate angle at p2"""
        v1 = p1 - p2
        v2 = p3 - p2
        
        dot = np.dot(v1, v2)
        mag = np.linalg.norm(v1) * np.linalg.norm(v2)
        
        if mag == 0:
            return 0.0
        
        cosine = np.clip(dot / mag, -1.0, 1.0)
        angle = np.degrees(np.arccos(cosine))
        return angle
    
    def get_landmark_coords(self, landmarks, idx):
        """Get 3D coordinates for landmark index"""
        start = idx * 4
        return landmarks[start:start+3]
    
    def detect_exercise_rule_based(self, landmarks):
        """
        Rule-based exercise detection (demo mode)
        
        Uses landmark positions and angles to infer exercise
        
        Heuristics:
        - Squats: Knees bent, hips low
        - Push-ups: Prone position, elbow flexion
        - Bicep Curls: Standing, elbow flexion, wrist movement
        """
        # Get key landmarks
        left_shoulder = self.get_landmark_coords(landmarks, 11)
        right_shoulder = self.get_landmark_coords(landmarks, 12)
        left_elbow = self.get_landmark_coords(landmarks, 13)
        right_elbow = self.get_landmark_coords(landmarks, 14)
        left_wrist = self.get_landmark_coords(landmarks, 15)
        left_hip = self.get_landmark_coords(landmarks, 23)
        right_hip = self.get_landmark_coords(landmarks, 24)
        left_knee = self.get_landmark_coords(landmarks, 25)
        left_ankle = self.get_landmark_coords(landmarks, 27)
        
        # Calculate key angles
        knee_angle = self.calculate_angle(left_hip, left_knee, left_ankle)
        elbow_angle = self.calculate_angle(left_shoulder, left_elbow, left_wrist)
        
        # Calculate positions
        avg_shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2
        avg_hip_y = (left_hip[1] + right_hip[1]) / 2
        
        # Exercise detection logic
        # Squats: Upright position, knee angle varies
        if avg_shoulder_y < avg_hip_y + 0.2 and knee_angle < 150:
            return "Squat"
        
        # Push-ups: Horizontal body, elbow flexion
        elif avg_shoulder_y > 0.5 and abs(avg_shoulder_y - avg_hip_y) < 0.15:
            return "Push-up"
        
        # Bicep Curls: Upright, elbow flexion
        elif avg_shoulder_y < avg_hip_y and elbow_angle < 120:
            return "Bicep Curl"
        
        else:
            return "Unknown"
    
    def check_form_correctness(self, landmarks, exercise):
        """
        Rule-based form correctness check
        
        Returns:
            (is_correct, error_messages)
        """
        errors = []
        
        if exercise == "Squat":
            errors = self.check_squat_form(landmarks)
        elif exercise == "Push-up":
            errors = self.check_pushup_form(landmarks)
        elif exercise == "Bicep Curl":
            errors = self.check_curl_form(landmarks)
        
        is_correct = len(errors) == 0
        return is_correct, errors
    
    def check_squat_form(self, landmarks):
        """Check squat form errors"""
        errors = []
        
        # Get landmarks
        left_hip = self.get_landmark_coords(landmarks, 23)
        left_knee = self.get_landmark_coords(landmarks, 25)
        left_ankle = self.get_landmark_coords(landmarks, 27)
        left_shoulder = self.get_landmark_coords(landmarks, 11)
        
        # Knee angle
        knee_angle = self.calculate_angle(left_hip, left_knee, left_ankle)
        
        # Check depth
        if knee_angle > 110:
            errors.append("Go deeper - aim for 90°")
        
        # Check back angle
        hip_to_shoulder_angle = abs(left_hip[0] - left_shoulder[0])
        if hip_to_shoulder_angle > 0.15:
            errors.append("Keep torso more upright")
        
        # Check knee alignment
        knee_ankle_distance = abs(left_knee[0] - left_ankle[0])
        if knee_ankle_distance > 0.12:
            errors.append("Knees over toes - check form")
        
        return errors
    
    def check_pushup_form(self, landmarks):
        """Check push-up form errors"""
        errors = []
        
        # Get landmarks
        left_shoulder = self.get_landmark_coords(landmarks, 11)
        left_elbow = self.get_landmark_coords(landmarks, 13)
        left_wrist = self.get_landmark_coords(landmarks, 15)
        left_hip = self.get_landmark_coords(landmarks, 23)
        left_ankle = self.get_landmark_coords(landmarks, 27)
        
        # Elbow angle
        elbow_angle = self.calculate_angle(left_shoulder, left_elbow, left_wrist)
        
        # Check depth
        if elbow_angle > 100:
            errors.append("Lower down to 90°")
        
        # Check body line
        body_line_diff = abs(left_shoulder[1] - left_hip[1])
        if body_line_diff > 0.15:
            errors.append("Keep body straight - engage core")
        
        return errors
    
    def check_curl_form(self, landmarks):
        """Check bicep curl form errors"""
        errors = []
        
        # Get landmarks
        left_shoulder = self.get_landmark_coords(landmarks, 11)
        left_elbow = self.get_landmark_coords(landmarks, 13)
        left_wrist = self.get_landmark_coords(landmarks, 15)
        left_hip = self.get_landmark_coords(landmarks, 23)
        
        # Check elbow stability
        elbow_hip_distance = abs(left_elbow[0] - left_hip[0])
        if elbow_hip_distance < 0.05:
            errors.append("Elbows moving forward - keep stable")
        
        # Check elbow angle for ROM
        elbow_angle = self.calculate_angle(left_shoulder, left_elbow, left_wrist)
        if elbow_angle > 160:
            errors.append("Incomplete ROM - full extension")
        
        return errors
    
    def count_reps(self, exercise):
        """
        Count repetitions using peak detection
        
        Algorithm:
        1. Extract discriminative signal (e.g., wrist_y for curls)
        2. Detect peaks/valleys
        3. Validate peak prominence (full ROM)
        4. Update count
        """
        if len(self.landmark_buffer) < 10:
            return self.rep_count
        
        # Extract signal based on exercise
        signal = []
        for landmarks in self.landmark_buffer:
            if exercise == "Squat":
                # Use hip height
                hip_y = self.get_landmark_coords(landmarks, 23)[1]
                signal.append(hip_y)
            elif exercise == "Push-up":
                # Use shoulder height
                shoulder_y = self.get_landmark_coords(landmarks, 11)[1]
                signal.append(shoulder_y)
            elif exercise == "Bicep Curl":
                # Use wrist height
                wrist_y = self.get_landmark_coords(landmarks, 15)[1]
                signal.append(-wrist_y)  # Invert for peak detection
            else:
                return self.rep_count
        
        signal = np.array(signal)
        
        # Find peaks (valleys for squat/pushup)
        peaks, properties = find_peaks(
            signal,
            distance=10,       # Min 10 frames between reps
            prominence=0.05    # Min peak height
        )
        
        # Check if new peak detected
        if len(peaks) > 0:
            latest_peak = len(signal) - 1 - list(peaks)[-1]  # Frames ago
            
            # If peak is recent and not counted yet
            if latest_peak < 5 and len(self.landmark_buffer) - peaks[-1] > self.last_peak_frame:
                self.rep_count += 1
                self.last_peak_frame = len(self.landmark_buffer)
        
        return self.rep_count
    
    def draw_feedback(self, frame, landmarks_result):
        """
        Draw visual feedback on frame
        
        Elements:
        - Pose skeleton
        - Exercise type
        - Form correctness
        - Rep count
        - Error messages
        """
        h, w = frame.shape[:2]
        
        # Draw pose skeleton
        if landmarks_result.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                landmarks_result.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
        
        # Draw info panel
        panel_height = 200
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Exercise type
        cv2.putText(frame, f"Exercise: {self.current_exercise}", 
                   (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        
        # Form correctness
        form_color = (0, 255, 0) if self.is_correct_form else (0, 0, 255)
        form_text = "CORRECT FORM" if self.is_correct_form else "INCORRECT FORM"
        cv2.putText(frame, form_text, 
                   (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 1.0, form_color, 2)
        
        # Rep count
        cv2.putText(frame, f"Reps: {self.rep_count}", 
                   (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
        
        # Error messages
        y_offset = 175
        for i, msg in enumerate(self.feedback_messages[:3]):  # Max 3 messages
            cv2.putText(frame, f"• {msg}", 
                       (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 255), 1)
            y_offset += 30
        
        # Instructions
        cv2.putText(frame, "Press 'Q' to quit | 'R' to reset count", 
                   (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        return frame
    
    def process_frame(self, frame):
        """
        Process single frame
        
        Pipeline:
        1. Pose estimation
        2. Extract landmarks
        3. Classify exercise
        4. Check form
        5. Count reps
        6. Draw feedback
        """
        # Convert to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Pose estimation
        results = self.pose.process(frame_rgb)
        
        # Extract landmarks
        if results.pose_landmarks:
            landmarks = self.extract_landmarks(results)
            
            # Add to buffer
            self.landmark_buffer.append(landmarks)
            
            # Detect exercise
            if self.demo_mode:
                self.current_exercise = self.detect_exercise_rule_based(landmarks)
            else:
                # Use trained model (implement based on your model)
                pass
            
            # Check form correctness
            if self.current_exercise != "Unknown":
                self.is_correct_form, self.feedback_messages = \
                    self.check_form_correctness(landmarks, self.current_exercise)
                
                # Count reps
                self.rep_count = self.count_reps(self.current_exercise)
        
        # Draw feedback
        frame = self.draw_feedback(frame, results)
        
        return frame
    
    def run(self, camera_id=0):
        """
        Run real-time monitoring
        
        Args:
            camera_id: Camera device ID (0 for default webcam)
        """
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        print("\n" + "="*60)
        print("WORKOUT MONITORING SYSTEM - ACTIVE")
        print("="*60)
        print("\nInstructions:")
        print("  - Position yourself in front of camera")
        print("  - Perform exercises: Squats, Push-ups, or Bicep Curls")
        print("  - Press 'R' to reset rep count")
        print("  - Press 'Q' to quit")
        print("\n" + "="*60 + "\n")
        
        # FPS calculation
        fps_history = deque(maxlen=30)
        
        while cap.isOpened():
            start_time = time.time()
            
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            frame = self.process_frame(frame)
            
            # Calculate FPS
            elapsed = time.time() - start_time
            fps = 1.0 / elapsed if elapsed > 0 else 0
            fps_history.append(fps)
            avg_fps = np.mean(fps_history)
            
            # Display FPS
            cv2.putText(frame, f"FPS: {avg_fps:.1f}", 
                       (frame.shape[1] - 150, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Show frame
            cv2.imshow('Workout Monitor', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.rep_count = 0
                self.last_peak_frame = -999
                print("✓ Rep count reset")
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        print("\n" + "="*60)
        print("SESSION SUMMARY")
        print("="*60)
        print(f"Exercise: {self.current_exercise}")
        print(f"Total Reps: {self.rep_count}")
        print(f"Average FPS: {np.mean(fps_history):.1f}")
        print("="*60 + "\n")


def main():
    """Main entry point"""
    print("Initializing Workout Monitor...")
    
    # Initialize monitor
    monitor = WorkoutMonitor()
    
    # Run real-time monitoring
    monitor.run(camera_id=0)


if __name__ == "__main__":
    main()
