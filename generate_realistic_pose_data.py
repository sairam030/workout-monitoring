#!/usr/bin/env python3
"""
Generate Realistic Pose Data
=============================

Generates biomechanically accurate pose landmark sequences for exercises.
This simulates MediaPipe output with realistic human movement patterns.

WHY: Since MediaPipe requires real human videos (not animations), we generate
     synthetic pose data that represents what MediaPipe would extract from
     real exercise videos. This allows us to demonstrate the complete pipeline.
"""

import numpy as np
import os
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class LandmarkIndices:
    """MediaPipe Pose landmark indices"""
    NOSE = 0
    LEFT_EYE_INNER = 1
    LEFT_EYE = 2
    LEFT_EYE_OUTER = 3
    RIGHT_EYE_INNER = 4
    RIGHT_EYE = 5
    RIGHT_EYE_OUTER = 6
    LEFT_EAR = 7
    RIGHT_EAR = 8
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_PINKY = 17
    RIGHT_PINKY = 18
    LEFT_INDEX = 19
    RIGHT_INDEX = 20
    LEFT_THUMB = 21
    RIGHT_THUMB = 22
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32

class RealisticPoseGenerator:
    """Generate biomechanically realistic pose sequences"""
    
    def __init__(self):
        self.landmarks = LandmarkIndices()
        
    def create_base_skeleton(self) -> np.ndarray:
        """
        Create a base standing pose (all landmarks)
        Returns: (33, 4) array [x, y, z, visibility]
        
        Coordinate system:
        - x: horizontal (0 = left, 1 = right, 0.5 = center)
        - y: vertical (0 = top, 1 = bottom, higher = lower on screen)
        - z: depth (relative to hips, negative = toward camera)
        - visibility: confidence score (0-1)
        """
        pose = np.zeros((33, 4))
        
        # Set visibility to 1.0 (all landmarks visible)
        pose[:, 3] = 1.0
        
        # Face (top of body)
        pose[self.landmarks.NOSE] = [0.50, 0.15, 0.0, 1.0]
        pose[self.landmarks.LEFT_EYE_INNER] = [0.48, 0.14, -0.01, 1.0]
        pose[self.landmarks.LEFT_EYE] = [0.47, 0.14, -0.01, 1.0]
        pose[self.landmarks.LEFT_EYE_OUTER] = [0.46, 0.14, -0.01, 1.0]
        pose[self.landmarks.RIGHT_EYE_INNER] = [0.52, 0.14, -0.01, 1.0]
        pose[self.landmarks.RIGHT_EYE] = [0.53, 0.14, -0.01, 1.0]
        pose[self.landmarks.RIGHT_EYE_OUTER] = [0.54, 0.14, -0.01, 1.0]
        pose[self.landmarks.LEFT_EAR] = [0.45, 0.15, 0.02, 1.0]
        pose[self.landmarks.RIGHT_EAR] = [0.55, 0.15, 0.02, 1.0]
        pose[self.landmarks.MOUTH_LEFT] = [0.48, 0.17, 0.0, 1.0]
        pose[self.landmarks.MOUTH_RIGHT] = [0.52, 0.17, 0.0, 1.0]
        
        # Shoulders
        pose[self.landmarks.LEFT_SHOULDER] = [0.42, 0.25, 0.0, 1.0]
        pose[self.landmarks.RIGHT_SHOULDER] = [0.58, 0.25, 0.0, 1.0]
        
        # Arms (hanging down)
        pose[self.landmarks.LEFT_ELBOW] = [0.40, 0.40, 0.0, 1.0]
        pose[self.landmarks.RIGHT_ELBOW] = [0.60, 0.40, 0.0, 1.0]
        pose[self.landmarks.LEFT_WRIST] = [0.38, 0.55, 0.0, 1.0]
        pose[self.landmarks.RIGHT_WRIST] = [0.62, 0.55, 0.0, 1.0]
        
        # Hands
        pose[self.landmarks.LEFT_PINKY] = [0.37, 0.58, 0.0, 1.0]
        pose[self.landmarks.RIGHT_PINKY] = [0.63, 0.58, 0.0, 1.0]
        pose[self.landmarks.LEFT_INDEX] = [0.38, 0.59, -0.01, 1.0]
        pose[self.landmarks.RIGHT_INDEX] = [0.62, 0.59, -0.01, 1.0]
        pose[self.landmarks.LEFT_THUMB] = [0.39, 0.58, -0.01, 1.0]
        pose[self.landmarks.RIGHT_THUMB] = [0.61, 0.58, -0.01, 1.0]
        
        # Hips
        pose[self.landmarks.LEFT_HIP] = [0.45, 0.55, 0.0, 1.0]
        pose[self.landmarks.RIGHT_HIP] = [0.55, 0.55, 0.0, 1.0]
        
        # Legs (standing straight)
        pose[self.landmarks.LEFT_KNEE] = [0.45, 0.75, 0.0, 1.0]
        pose[self.landmarks.RIGHT_KNEE] = [0.55, 0.75, 0.0, 1.0]
        pose[self.landmarks.LEFT_ANKLE] = [0.45, 0.95, 0.0, 1.0]
        pose[self.landmarks.RIGHT_ANKLE] = [0.55, 0.95, 0.0, 1.0]
        
        # Feet
        pose[self.landmarks.LEFT_HEEL] = [0.44, 0.96, 0.01, 1.0]
        pose[self.landmarks.RIGHT_HEEL] = [0.54, 0.96, 0.01, 1.0]
        pose[self.landmarks.LEFT_FOOT_INDEX] = [0.45, 0.97, -0.01, 1.0]
        pose[self.landmarks.RIGHT_FOOT_INDEX] = [0.55, 0.97, -0.01, 1.0]
        
        return pose
    
    def generate_squat_sequence(self, num_reps=3, fps=30, correct_form=True) -> np.ndarray:
        """
        Generate squat exercise sequence
        
        WHY: Squats involve hip and knee flexion with forward torso lean
        """
        frames_per_rep = int(fps * 3.0)  # 3 seconds per rep
        total_frames = frames_per_rep * num_reps
        
        sequence = []
        
        for frame_idx in range(total_frames):
            base_pose = self.create_base_skeleton()
            
            # Calculate rep phase (0 to 1)
            rep_progress = (frame_idx % frames_per_rep) / frames_per_rep
            phase = np.sin(rep_progress * 2 * np.pi)  # -1 to 1
            
            # Squat depth (0 = standing, 1 = bottom)
            depth = (phase + 1) / 2  # 0 to 1
            
            # Knee and hip flexion
            knee_bend = depth * 0.20  # Knees move forward and down
            hip_drop = depth * 0.15   # Hips drop
            torso_lean = depth * 0.05 if correct_form else depth * 0.15  # Incorrect: excessive lean
            
            # Adjust body landmarks
            # Hips drop
            base_pose[self.landmarks.LEFT_HIP, 1] += hip_drop
            base_pose[self.landmarks.RIGHT_HIP, 1] += hip_drop
            
            # Knees bend forward
            base_pose[self.landmarks.LEFT_KNEE, 1] += hip_drop * 0.8
            base_pose[self.landmarks.RIGHT_KNEE, 1] += hip_drop * 0.8
            base_pose[self.landmarks.LEFT_KNEE, 0] -= knee_bend * 0.1
            base_pose[self.landmarks.RIGHT_KNEE, 0] += knee_bend * 0.1
            
            # Torso leans forward
            for idx in [self.landmarks.NOSE, self.landmarks.LEFT_SHOULDER, 
                       self.landmarks.RIGHT_SHOULDER] + list(range(1, 11)):
                base_pose[idx, 1] += torso_lean
                base_pose[idx, 2] -= torso_lean * 0.5  # Move toward camera
            
            # Arms extend forward for balance
            base_pose[self.landmarks.LEFT_ELBOW, 0] -= depth * 0.05
            base_pose[self.landmarks.RIGHT_ELBOW, 0] += depth * 0.05
            base_pose[self.landmarks.LEFT_WRIST, 0] -= depth * 0.10
            base_pose[self.landmarks.RIGHT_WRIST, 0] += depth * 0.10
            base_pose[self.landmarks.LEFT_WRIST, 1] -= depth * 0.15
            base_pose[self.landmarks.RIGHT_WRIST, 1] -= depth * 0.15
            
            # Add natural variance
            noise = np.random.normal(0, 0.002, base_pose.shape)
            noise[:, 3] = 0  # Don't add noise to visibility
            base_pose += noise
            
            # Flatten to (132,) format
            sequence.append(base_pose.flatten())
        
        return np.array(sequence)
    
    def generate_pushup_sequence(self, num_reps=3, fps=30, correct_form=True) -> np.ndarray:
        """
        Generate pushup exercise sequence
        
        WHY: Pushups involve elbow flexion/extension with plank position
        """
        frames_per_rep = int(fps * 2.5)
        total_frames = frames_per_rep * num_reps
        
        sequence = []
        
        for frame_idx in range(total_frames):
            base_pose = self.create_base_skeleton()
            
            rep_progress = (frame_idx % frames_per_rep) / frames_per_rep
            phase = np.sin(rep_progress * 2 * np.pi)
            depth = (phase + 1) / 2  # 0 = top, 1 = bottom
            
            # Plank position: rotate entire body
            plank_rotation = np.pi / 2  # 90 degrees (horizontal)
            
            # Transform to plank (side view)
            for idx in range(33):
                old_y = base_pose[idx, 1]
                old_z = base_pose[idx, 2]
                
                # Rotate around center
                center_y = 0.5
                rel_y = old_y - center_y
                
                new_y = center_y + rel_y * 0.3  # Compress vertically
                new_z = -rel_y * 1.5  # Convert to depth
                
                base_pose[idx, 1] = new_y
                base_pose[idx, 2] = new_z
            
            # Elbow flexion (pushup motion)
            elbow_bend = depth * 0.15
            
            # Arms move up/down
            base_pose[self.landmarks.LEFT_ELBOW, 1] += elbow_bend
            base_pose[self.landmarks.RIGHT_ELBOW, 1] += elbow_bend
            base_pose[self.landmarks.LEFT_WRIST, 1] += elbow_bend * 0.5
            base_pose[self.landmarks.RIGHT_WRIST, 1] += elbow_bend * 0.5
            
            # Body raises/lowers
            body_lift = elbow_bend
            for idx in list(range(11)) + [self.landmarks.LEFT_SHOULDER, self.landmarks.RIGHT_SHOULDER,
                                          self.landmarks.LEFT_HIP, self.landmarks.RIGHT_HIP]:
                base_pose[idx, 1] += body_lift * (1.0 if correct_form else 0.5)  # Incorrect: hips sag
            
            # Add variance
            noise = np.random.normal(0, 0.002, base_pose.shape)
            noise[:, 3] = 0
            base_pose += noise
            
            sequence.append(base_pose.flatten())
        
        return np.array(sequence)
    
    def generate_bicep_curl_sequence(self, num_reps=4, fps=30, correct_form=True) -> np.ndarray:
        """
        Generate bicep curl exercise sequence
        
        WHY: Curls involve elbow flexion with stable shoulder and torso
        """
        frames_per_rep = int(fps * 2.0)
        total_frames = frames_per_rep * num_reps
        
        sequence = []
        
        for frame_idx in range(total_frames):
            base_pose = self.create_base_skeleton()
            
            rep_progress = (frame_idx % frames_per_rep) / frames_per_rep
            phase = np.sin(rep_progress * 2 * np.pi)
            curl_amount = (phase + 1) / 2  # 0 = extended, 1 = curled
            
            # Elbow flexion
            curl_height = curl_amount * 0.25
            
            # Wrists move toward shoulders
            base_pose[self.landmarks.LEFT_WRIST, 1] -= curl_height
            base_pose[self.landmarks.RIGHT_WRIST, 1] -= curl_height
            base_pose[self.landmarks.LEFT_WRIST, 0] += curl_amount * 0.05
            base_pose[self.landmarks.RIGHT_WRIST, 0] -= curl_amount * 0.05
            base_pose[self.landmarks.LEFT_WRIST, 2] -= curl_amount * 0.10  # Move toward body
            base_pose[self.landmarks.RIGHT_WRIST, 2] -= curl_amount * 0.10
            
            # Elbows stay relatively stable (key form point)
            elbow_drift = 0.02 if not correct_form else 0.005
            base_pose[self.landmarks.LEFT_ELBOW, 1] -= curl_amount * elbow_drift
            base_pose[self.landmarks.RIGHT_ELBOW, 1] -= curl_amount * elbow_drift
            
            # Hands rotate
            for hand_idx in [self.landmarks.LEFT_INDEX, self.landmarks.LEFT_PINKY,
                           self.landmarks.RIGHT_INDEX, self.landmarks.RIGHT_PINKY]:
                base_pose[hand_idx, 1] -= curl_height * 1.1
            
            # Add variance
            noise = np.random.normal(0, 0.002, base_pose.shape)
            noise[:, 3] = 0
            base_pose += noise
            
            sequence.append(base_pose.flatten())
        
        return np.array(sequence)

def generate_complete_dataset():
    """Generate complete exercise dataset"""
    
    print("\n" + "="*70)
    print("GENERATING REALISTIC POSE DATASET")
    print("="*70)
    print("\nCreating biomechanically accurate pose sequences...")
    print("Format: MediaPipe Pose landmarks (33 landmarks × 4 values = 132 features)")
    print("-"*70)
    
    generator = RealisticPoseGenerator()
    os.makedirs('data/processed', exist_ok=True)
    
    # Generate different variations
    configs = [
        # Squats
        {'exercise': 'squats', 'form': 'correct', 'reps': 3, 'count': 5},
        {'exercise': 'squats', 'form': 'incorrect', 'reps': 3, 'count': 3},
        
        # Pushups
        {'exercise': 'pushups', 'form': 'correct', 'reps': 3, 'count': 5},
        {'exercise': 'pushups', 'form': 'incorrect', 'reps': 3, 'count': 3},
        
        # Bicep curls
        {'exercise': 'bicep_curls', 'form': 'correct', 'reps': 4, 'count': 5},
        {'exercise': 'bicep_curls', 'form': 'incorrect', 'reps': 4, 'count': 3},
    ]
    
    file_counter = 1
    generated_files = []
    
    print("\nGenerating sequences:")
    
    for config in configs:
        exercise = config['exercise']
        form = config['form']
        reps = config['reps']
        count = config['count']
        
        for i in range(count):
            # Generate sequence
            if exercise == 'squats':
                sequence = generator.generate_squat_sequence(reps, correct_form=(form=='correct'))
            elif exercise == 'pushups':
                sequence = generator.generate_pushup_sequence(reps, correct_form=(form=='correct'))
            else:  # bicep_curls
                sequence = generator.generate_bicep_curl_sequence(reps, correct_form=(form=='correct'))
            
            # Save
            filename = f"{exercise}_{form}_{file_counter:03d}.npz"
            filepath = f"data/processed/{filename}"
            
            np.savez_compressed(
                filepath,
                landmarks=sequence,
                metadata=np.array([{
                    'exercise': exercise,
                    'form': form,
                    'fps': 30.0,
                    'total_frames': len(sequence),
                    'valid_frames': len(sequence)
                }], dtype=object)
            )
            
            generated_files.append(filepath)
            print(f"  ✓ {filename:40s} ({len(sequence):3d} frames)")
            file_counter += 1
    
    print("\n" + "="*70)
    print("DATASET GENERATION COMPLETE!")
    print("="*70)
    
    # Statistics
    total_frames = 0
    for filepath in generated_files:
        data = np.load(filepath, allow_pickle=True)
        total_frames += len(data['landmarks'])
    
    print(f"\n📊 Dataset Statistics:")
    print(f"  - Total sequences: {len(generated_files)}")
    print(f"  - Total frames: {total_frames}")
    print(f"  - Features per frame: 132 (33 landmarks × 4 values)")
    print(f"  - Exercises: squats, pushups, bicep_curls")
    print(f"  - Forms: correct, incorrect")
    print(f"  - Average frames per sequence: {total_frames // len(generated_files)}")
    
    # Class distribution
    print(f"\n📈 Class Distribution:")
    for exercise in ['squats', 'pushups', 'bicep_curls']:
        correct_count = sum(1 for f in generated_files if f"{exercise}_correct" in f)
        incorrect_count = sum(1 for f in generated_files if f"{exercise}_incorrect" in f)
        print(f"  - {exercise:12s}: {correct_count} correct, {incorrect_count} incorrect")
    
    print(f"\n✅ Dataset ready for analysis!")
    print(f"📁 Location: data/processed/")
    print(f"\n🎯 Next step: Open notebooks for detailed EDA and pipeline")

if __name__ == "__main__":
    generate_complete_dataset()
