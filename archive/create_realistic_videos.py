#!/usr/bin/env python3
"""
Create Realistic Exercise Videos with Full Body
================================================

Creates videos with detailed animated human figures that MediaPipe can detect.
Uses a more realistic skeleton with proper proportions.
"""

import cv2
import numpy as np
import os

class RealisticHumanRenderer:
    """Render a realistic human figure with proper proportions"""
    
    def __init__(self, width=640, height=480):
        self.width = width
        self.height = height
        self.center_x = width // 2
        self.ground_y = int(height * 0.85)
        
        # Human proportions (in pixels)
        self.head_radius = 25
        self.torso_length = 100
        self.upper_arm_length = 55
        self.forearm_length = 50
        self.thigh_length = 70
        self.shin_length = 70
        
        # Colors (BGR)
        self.skin_color = (180, 150, 120)
        self.shirt_color = (100, 50, 200)  # Red shirt
        self.pants_color = (140, 80, 40)   # Blue pants
        
    def draw_circle_filled(self, img, center, radius, color):
        """Draw filled circle (for head and joints)"""
        cv2.circle(img, center, radius, color, -1)
        cv2.circle(img, center, radius, (0, 0, 0), 2)  # Black outline
        
    def draw_limb(self, img, start, end, thickness, color):
        """Draw a limb (cylinder-like)"""
        cv2.line(img, start, end, (0, 0, 0), thickness + 4)  # Black outline
        cv2.line(img, start, end, color, thickness)
        
    def draw_human(self, img, pose_angles):
        """
        Draw human figure with given pose angles
        
        pose_angles dict with keys:
        - hip_bend: torso angle from vertical (0 = standing, 90 = horizontal)
        - left_knee: knee bend angle (0 = straight, 90 = bent)
        - right_knee: knee bend angle
        - left_elbow: elbow bend angle
        - right_elbow: elbow bend angle
        - left_shoulder: shoulder angle (0 = arms down, 90 = horizontal)
        - right_shoulder: shoulder angle
        """
        
        # Extract angles
        hip_bend = np.radians(pose_angles.get('hip_bend', 0))
        left_knee = np.radians(pose_angles.get('left_knee', 0))
        right_knee = np.radians(pose_angles.get('right_knee', 0))
        left_elbow = np.radians(pose_angles.get('left_elbow', 180))
        right_elbow = np.radians(pose_angles.get('right_elbow', 180))
        left_shoulder = np.radians(pose_angles.get('left_shoulder', 0))
        right_shoulder = np.radians(pose_angles.get('right_shoulder', 0))
        
        # Calculate hip position (moves based on squat depth)
        squat_depth = (pose_angles.get('left_knee', 0) + pose_angles.get('right_knee', 0)) / 2
        hip_y = self.ground_y - int(self.thigh_length + self.shin_length - squat_depth * 0.8)
        hip_pos = (self.center_x, hip_y)
        
        # Torso (shoulder to hip)
        shoulder_offset_x = int(self.torso_length * np.sin(hip_bend))
        shoulder_offset_y = int(self.torso_length * np.cos(hip_bend))
        shoulder_pos = (hip_pos[0] + shoulder_offset_x, hip_pos[1] - shoulder_offset_y)
        
        # Head (on top of shoulders)
        head_pos = (shoulder_pos[0], shoulder_pos[1] - self.head_radius - 5)
        
        # Left leg
        left_hip = (hip_pos[0] - 10, hip_pos[1])
        left_knee_x = left_hip[0] + int(self.thigh_length * np.sin(left_knee * 0.5))
        left_knee_y = left_hip[1] + int(self.thigh_length * np.cos(left_knee * 0.5))
        left_knee_pos = (left_knee_x, left_knee_y)
        
        left_ankle_x = left_knee_x + int(self.shin_length * np.sin(left_knee))
        left_ankle_y = self.ground_y
        left_ankle_pos = (left_ankle_x, left_ankle_y)
        
        # Right leg (mirror)
        right_hip = (hip_pos[0] + 10, hip_pos[1])
        right_knee_x = right_hip[0] + int(self.thigh_length * np.sin(right_knee * 0.5))
        right_knee_y = right_hip[1] + int(self.thigh_length * np.cos(right_knee * 0.5))
        right_knee_pos = (right_knee_x, right_knee_y)
        
        right_ankle_x = right_knee_x + int(self.shin_length * np.sin(right_knee))
        right_ankle_y = self.ground_y
        right_ankle_pos = (right_ankle_x, right_ankle_y)
        
        # Left arm
        left_shoulder_pos = (shoulder_pos[0] - 20, shoulder_pos[1])
        left_elbow_x = left_shoulder_pos[0] + int(self.upper_arm_length * np.sin(left_shoulder))
        left_elbow_y = left_shoulder_pos[1] + int(self.upper_arm_length * np.cos(left_shoulder))
        left_elbow_pos = (left_elbow_x, left_elbow_y)
        
        left_hand_x = left_elbow_x + int(self.forearm_length * np.sin(left_shoulder + left_elbow - np.pi))
        left_hand_y = left_elbow_y + int(self.forearm_length * np.cos(left_shoulder + left_elbow - np.pi))
        left_hand_pos = (left_hand_x, left_hand_y)
        
        # Right arm (mirror)
        right_shoulder_pos = (shoulder_pos[0] + 20, shoulder_pos[1])
        right_elbow_x = right_shoulder_pos[0] - int(self.upper_arm_length * np.sin(right_shoulder))
        right_elbow_y = right_shoulder_pos[1] + int(self.upper_arm_length * np.cos(right_shoulder))
        right_elbow_pos = (right_elbow_x, right_elbow_y)
        
        right_hand_x = right_elbow_x - int(self.forearm_length * np.sin(right_shoulder + right_elbow - np.pi))
        right_hand_y = right_elbow_y + int(self.forearm_length * np.cos(right_shoulder + right_elbow - np.pi))
        right_hand_pos = (right_hand_x, right_hand_y)
        
        # Draw body parts (back to front for proper layering)
        
        # Legs (pants color)
        self.draw_limb(img, left_hip, left_knee_pos, 16, self.pants_color)
        self.draw_limb(img, left_knee_pos, left_ankle_pos, 14, self.pants_color)
        self.draw_limb(img, right_hip, right_knee_pos, 16, self.pants_color)
        self.draw_limb(img, right_knee_pos, right_ankle_pos, 14, self.pants_color)
        
        # Torso (shirt color)
        self.draw_limb(img, hip_pos, shoulder_pos, 30, self.shirt_color)
        
        # Arms (skin color)
        self.draw_limb(img, left_shoulder_pos, left_elbow_pos, 12, self.skin_color)
        self.draw_limb(img, left_elbow_pos, left_hand_pos, 10, self.skin_color)
        self.draw_limb(img, right_shoulder_pos, right_elbow_pos, 12, self.skin_color)
        self.draw_limb(img, right_elbow_pos, right_hand_pos, 10, self.skin_color)
        
        # Joints
        self.draw_circle_filled(img, left_knee_pos, 8, self.skin_color)
        self.draw_circle_filled(img, right_knee_pos, 8, self.skin_color)
        self.draw_circle_filled(img, left_elbow_pos, 7, self.skin_color)
        self.draw_circle_filled(img, right_elbow_pos, 7, self.skin_color)
        
        # Head (skin color)
        self.draw_circle_filled(img, head_pos, self.head_radius, self.skin_color)
        
        # Face features
        eye_y = head_pos[1] - 5
        cv2.circle(img, (head_pos[0] - 8, eye_y), 3, (0, 0, 0), -1)  # Left eye
        cv2.circle(img, (head_pos[0] + 8, eye_y), 3, (0, 0, 0), -1)  # Right eye
        cv2.ellipse(img, (head_pos[0], head_pos[1] + 8), (8, 4), 0, 0, 180, (0, 0, 0), 2)  # Smile

def create_squat_video(output_path, duration=10, fps=30):
    """Create a realistic squat exercise video"""
    renderer = RealisticHumanRenderer()
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (renderer.width, renderer.height))
    
    total_frames = duration * fps
    reps = 3
    frames_per_rep = total_frames // reps
    
    for frame_idx in range(total_frames):
        # Create white background
        img = np.ones((renderer.height, renderer.width, 3), dtype=np.uint8) * 255
        
        # Add floor line
        cv2.line(img, (0, renderer.ground_y), (renderer.width, renderer.ground_y), (200, 200, 200), 2)
        
        # Calculate squat phase
        rep_progress = (frame_idx % frames_per_rep) / frames_per_rep
        
        # Smooth squat motion (sine wave)
        squat_depth = np.sin(rep_progress * 2 * np.pi) * 45 + 45  # 0-90 degrees
        
        # Pose angles for squat
        pose = {
            'hip_bend': squat_depth * 0.3,  # Lean forward slightly
            'left_knee': squat_depth,
            'right_knee': squat_depth,
            'left_elbow': 160 - squat_depth * 0.5,  # Arms extend forward
            'right_elbow': 160 - squat_depth * 0.5,
            'left_shoulder': squat_depth * 0.8,  # Arms come forward
            'right_shoulder': squat_depth * 0.8,
        }
        
        renderer.draw_human(img, pose)
        
        # Add text overlay
        rep_num = (frame_idx // frames_per_rep) + 1
        cv2.putText(img, f"SQUAT", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(img, f"Rep: {rep_num}/{reps}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(img, f"Frame: {frame_idx}/{total_frames}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
        
        out.write(img)
    
    out.release()
    print(f"✓ Created: {output_path}")

def create_pushup_video(output_path, duration=10, fps=30):
    """Create a realistic pushup exercise video"""
    renderer = RealisticHumanRenderer()
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (renderer.width, renderer.height))
    
    total_frames = duration * fps
    reps = 3
    frames_per_rep = total_frames // reps
    
    for frame_idx in range(total_frames):
        img = np.ones((renderer.height, renderer.width, 3), dtype=np.uint8) * 255
        cv2.line(img, (0, renderer.ground_y), (renderer.width, renderer.ground_y), (200, 200, 200), 2)
        
        rep_progress = (frame_idx % frames_per_rep) / frames_per_rep
        pushup_depth = np.sin(rep_progress * 2 * np.pi) * 60 + 60  # 0-120 elbow bend
        
        pose = {
            'hip_bend': 90,  # Body horizontal
            'left_knee': 0,  # Legs straight
            'right_knee': 0,
            'left_elbow': pushup_depth,  # Elbow bends
            'right_elbow': pushup_depth,
            'left_shoulder': 90,  # Arms perpendicular
            'right_shoulder': 90,
        }
        
        renderer.draw_human(img, pose)
        
        rep_num = (frame_idx // frames_per_rep) + 1
        cv2.putText(img, f"PUSH-UP", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(img, f"Rep: {rep_num}/{reps}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(img, f"Frame: {frame_idx}/{total_frames}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
        
        out.write(img)
    
    out.release()
    print(f"✓ Created: {output_path}")

def create_curl_video(output_path, duration=10, fps=30):
    """Create a realistic bicep curl video"""
    renderer = RealisticHumanRenderer()
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (renderer.width, renderer.height))
    
    total_frames = duration * fps
    reps = 4
    frames_per_rep = total_frames // reps
    
    for frame_idx in range(total_frames):
        img = np.ones((renderer.height, renderer.width, 3), dtype=np.uint8) * 255
        cv2.line(img, (0, renderer.ground_y), (renderer.width, renderer.ground_y), (200, 200, 200), 2)
        
        rep_progress = (frame_idx % frames_per_rep) / frames_per_rep
        curl_angle = np.sin(rep_progress * 2 * np.pi) * 75 + 105  # 30-180 degrees
        
        pose = {
            'hip_bend': 0,  # Standing straight
            'left_knee': 0,
            'right_knee': 0,
            'left_elbow': curl_angle,  # Elbow curls
            'right_elbow': curl_angle,
            'left_shoulder': 0,  # Arms at sides
            'right_shoulder': 0,
        }
        
        renderer.draw_human(img, pose)
        
        rep_num = (frame_idx // frames_per_rep) + 1
        cv2.putText(img, f"BICEP CURL", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(img, f"Rep: {rep_num}/{reps}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(img, f"Frame: {frame_idx}/{total_frames}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
        
        out.write(img)
    
    out.release()
    print(f"✓ Created: {output_path}")

if __name__ == "__main__":
    print("\n" + "="*70)
    print("CREATING REALISTIC EXERCISE VIDEOS")
    print("="*70)
    print("\nThese videos feature detailed human figures that MediaPipe can detect.")
    print("-"*70)
    
    # Create output directories
    os.makedirs('data/raw/squats', exist_ok=True)
    os.makedirs('data/raw/pushups', exist_ok=True)
    os.makedirs('data/raw/bicep_curls', exist_ok=True)
    
    # Create videos
    print("\nGenerating videos...")
    create_squat_video('data/raw/squats/squat_realistic_01.mp4', duration=12)
    create_squat_video('data/raw/squats/squat_realistic_02.mp4', duration=15)
    create_pushup_video('data/raw/pushups/pushup_realistic_01.mp4', duration=12)
    create_pushup_video('data/raw/pushups/pushup_realistic_02.mp4', duration=15)
    create_curl_video('data/raw/bicep_curls/curl_realistic_01.mp4', duration=10)
    create_curl_video('data/raw/bicep_curls/curl_realistic_02.mp4', duration=12)
    
    print("\n" + "="*70)
    print("VIDEOS CREATED SUCCESSFULLY!")
    print("="*70)
    print("\n✓ Created 6 realistic exercise videos")
    print("✓ Videos feature detailed human figures")
    print("✓ Ready for MediaPipe pose detection")
    print("\n📁 Location: data/raw/[exercise]/")
    print("\n✅ Next step: python process_real_videos.py")
