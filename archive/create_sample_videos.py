#!/usr/bin/env python3
"""
Create Sample Exercise Videos
==============================

Creates realistic sample exercise videos using OpenCV
that can be processed through the complete pipeline.
"""

import cv2
import numpy as np
import os
from datetime import datetime

def create_squat_video(output_path, duration_sec=10, fps=30):
    """Create a sample squat video with moving stick figure"""
    
    width, height = 640, 480
    total_frames = duration_sec * fps
    
    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"  Creating squat video: {os.path.basename(output_path)}")
    
    for frame_num in range(total_frames):
        # Create black background
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add title
        cv2.putText(frame, "SQUAT EXERCISE", (width//2 - 100, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Simulate squat motion (up and down)
        phase = (frame_num / total_frames) * 4 * np.pi  # 2 complete reps
        
        # Body center
        center_x = width // 2
        hip_y = int(height * 0.5 + 50 * np.sin(phase))  # Hip moves up/down
        
        # Draw stick figure
        # Head
        cv2.circle(frame, (center_x, hip_y - 80), 20, (255, 200, 100), -1)
        
        # Body (spine)
        cv2.line(frame, (center_x, hip_y - 60), (center_x, hip_y), (255, 255, 255), 3)
        
        # Arms
        arm_angle = 30 + 10 * np.sin(phase)
        left_arm_x = int(center_x - 40 * np.cos(np.radians(arm_angle)))
        right_arm_x = int(center_x + 40 * np.cos(np.radians(arm_angle)))
        arm_y = hip_y - 20
        
        cv2.line(frame, (center_x, hip_y - 50), (left_arm_x, arm_y), (255, 255, 255), 3)
        cv2.line(frame, (center_x, hip_y - 50), (right_arm_x, arm_y), (255, 255, 255), 3)
        
        # Legs - bend during squat
        knee_bend = 60 + 40 * np.sin(phase)
        knee_y = hip_y + 60
        foot_y = hip_y + 120
        
        # Left leg
        cv2.line(frame, (center_x - 15, hip_y), (center_x - 30, knee_y), (255, 255, 255), 3)
        cv2.line(frame, (center_x - 30, knee_y), (center_x - 30, foot_y), (255, 255, 255), 3)
        
        # Right leg
        cv2.line(frame, (center_x + 15, hip_y), (center_x + 30, knee_y), (255, 255, 255), 3)
        cv2.line(frame, (center_x + 30, knee_y), (center_x + 30, foot_y), (255, 255, 255), 3)
        
        # Add frame counter
        cv2.putText(frame, f"Frame: {frame_num}/{total_frames}", (10, height - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        
        # Add rep counter
        rep_num = int((frame_num / total_frames) * 2) + 1
        cv2.putText(frame, f"Rep: {min(rep_num, 2)}/2", (width - 100, height - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        out.write(frame)
    
    out.release()
    print(f"  ✓ Created: {output_path}")

def create_pushup_video(output_path, duration_sec=10, fps=30):
    """Create a sample push-up video"""
    
    width, height = 640, 480
    total_frames = duration_sec * fps
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"  Creating push-up video: {os.path.basename(output_path)}")
    
    for frame_num in range(total_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        cv2.putText(frame, "PUSH-UP EXERCISE", (width//2 - 120, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Horizontal body position
        phase = (frame_num / total_frames) * 4 * np.pi
        
        # Body at push-up position
        body_y = int(height * 0.6 + 30 * np.sin(phase))  # Body moves up/down
        
        # Head
        cv2.circle(frame, (100, body_y - 20), 15, (255, 200, 100), -1)
        
        # Body line (horizontal)
        cv2.line(frame, (115, body_y), (400, body_y), (255, 255, 255), 3)
        
        # Arms (up/down motion)
        arm_height = 80 + 40 * np.sin(phase)
        # Left arm
        cv2.line(frame, (150, body_y), (150, body_y + int(arm_height)), (255, 255, 255), 3)
        # Right arm
        cv2.line(frame, (350, body_y), (350, body_y + int(arm_height)), (255, 255, 255), 3)
        
        # Legs
        cv2.line(frame, (400, body_y), (500, body_y + 20), (255, 255, 255), 3)
        
        cv2.putText(frame, f"Frame: {frame_num}/{total_frames}", (10, height - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        
        rep_num = int((frame_num / total_frames) * 2) + 1
        cv2.putText(frame, f"Rep: {min(rep_num, 2)}/2", (width - 100, height - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        out.write(frame)
    
    out.release()
    print(f"  ✓ Created: {output_path}")

def create_curl_video(output_path, duration_sec=10, fps=30):
    """Create a sample bicep curl video"""
    
    width, height = 640, 480
    total_frames = duration_sec * fps
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"  Creating curl video: {os.path.basename(output_path)}")
    
    for frame_num in range(total_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        cv2.putText(frame, "BICEP CURL EXERCISE", (width//2 - 140, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        phase = (frame_num / total_frames) * 4 * np.pi
        
        center_x = width // 2
        body_y = height // 2
        
        # Head
        cv2.circle(frame, (center_x, body_y - 100), 20, (255, 200, 100), -1)
        
        # Body
        cv2.line(frame, (center_x, body_y - 80), (center_x, body_y + 50), (255, 255, 255), 3)
        
        # Arms - curling motion
        curl_angle = 45 + 60 * np.sin(phase)  # Elbow angle changes
        
        # Left arm
        elbow_x_l = center_x - 50
        elbow_y = body_y - 40
        wrist_x_l = int(elbow_x_l - 40 * np.cos(np.radians(curl_angle)))
        wrist_y_l = int(elbow_y + 40 * np.sin(np.radians(curl_angle)))
        
        cv2.line(frame, (center_x, body_y - 60), (elbow_x_l, elbow_y), (255, 255, 255), 3)
        cv2.line(frame, (elbow_x_l, elbow_y), (wrist_x_l, wrist_y_l), (255, 255, 255), 3)
        
        # Right arm
        elbow_x_r = center_x + 50
        wrist_x_r = int(elbow_x_r + 40 * np.cos(np.radians(curl_angle)))
        wrist_y_r = int(elbow_y + 40 * np.sin(np.radians(curl_angle)))
        
        cv2.line(frame, (center_x, body_y - 60), (elbow_x_r, elbow_y), (255, 255, 255), 3)
        cv2.line(frame, (elbow_x_r, elbow_y), (wrist_x_r, wrist_y_r), (255, 255, 255), 3)
        
        # Dumbbells
        cv2.circle(frame, (wrist_x_l, wrist_y_l), 8, (0, 255, 255), -1)
        cv2.circle(frame, (wrist_x_r, wrist_y_r), 8, (0, 255, 255), -1)
        
        # Legs
        cv2.line(frame, (center_x - 15, body_y + 50), (center_x - 30, body_y + 130), (255, 255, 255), 3)
        cv2.line(frame, (center_x + 15, body_y + 50), (center_x + 30, body_y + 130), (255, 255, 255), 3)
        
        cv2.putText(frame, f"Frame: {frame_num}/{total_frames}", (10, height - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        
        rep_num = int((frame_num / total_frames) * 2) + 1
        cv2.putText(frame, f"Rep: {min(rep_num, 2)}/2", (width - 100, height - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        out.write(frame)
    
    out.release()
    print(f"  ✓ Created: {output_path}")

def create_all_videos():
    """Create sample videos for all exercises"""
    
    print("\n" + "="*70)
    print("CREATING SAMPLE EXERCISE VIDEOS")
    print("="*70)
    
    # Create directories
    os.makedirs('data/raw/squats', exist_ok=True)
    os.makedirs('data/raw/pushups', exist_ok=True)
    os.makedirs('data/raw/bicep_curls', exist_ok=True)
    
    videos = [
        ('data/raw/squats/squat_correct_01.mp4', create_squat_video, 10),
        ('data/raw/squats/squat_correct_02.mp4', create_squat_video, 12),
        ('data/raw/pushups/pushup_correct_01.mp4', create_pushup_video, 10),
        ('data/raw/pushups/pushup_correct_02.mp4', create_pushup_video, 12),
        ('data/raw/bicep_curls/curl_correct_01.mp4', create_curl_video, 10),
        ('data/raw/bicep_curls/curl_correct_02.mp4', create_curl_video, 12),
    ]
    
    print("\n📹 Creating videos...")
    print("-"*70)
    
    for output_path, create_func, duration in videos:
        create_func(output_path, duration_sec=duration)
    
    print("\n" + "="*70)
    print("✓ ALL VIDEOS CREATED SUCCESSFULLY!")
    print("="*70)
    
    # Show created files
    print("\n📂 Created files:")
    for exercise in ['squats', 'pushups', 'bicep_curls']:
        dir_path = f'data/raw/{exercise}'
        files = [f for f in os.listdir(dir_path) if f.endswith('.mp4')]
        print(f"\n  {exercise}:")
        for f in files:
            size = os.path.getsize(os.path.join(dir_path, f)) / (1024*1024)
            print(f"    - {f} ({size:.2f} MB)")
    
    print("\n✅ Ready to process videos!")
    print("Next step: python process_real_videos.py")

if __name__ == "__main__":
    create_all_videos()
