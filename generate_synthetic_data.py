#!/usr/bin/env python3
"""
Generate Synthetic Exercise Data for Testing
============================================

Creates synthetic pose landmark sequences to test the complete pipeline
without needing actual videos.
"""

import numpy as np
import os
import json

def generate_squat_sequence(num_frames=60, correct=True):
    """
    Generate synthetic squat pose sequence
    
    Args:
        num_frames: Number of frames in sequence
        correct: True for correct form, False for incorrect
        
    Returns:
        np.array: Shape (num_frames, 132) - synthetic landmarks
    """
    sequence = []
    
    for i in range(num_frames):
        # Generate 33 landmarks × 4 values = 132 features
        landmarks = np.zeros(132)
        
        # Simulate squat motion (hip going up and down)
        phase = (i / num_frames) * 2 * np.pi  # One full rep
        hip_height = 0.5 + 0.2 * np.sin(phase)  # Hip oscillates
        
        # Key landmarks for squat
        # Nose (0)
        landmarks[0:4] = [0.5, 0.2, 0, 0.9]
        
        # Shoulders (11, 12)
        landmarks[11*4:11*4+4] = [0.4, 0.3, 0, 0.95]
        landmarks[12*4:12*4+4] = [0.6, 0.3, 0, 0.95]
        
        # Hips (23, 24)
        landmarks[23*4:23*4+4] = [0.4, hip_height, 0, 0.95]
        landmarks[24*4:24*4+4] = [0.6, hip_height, 0, 0.95]
        
        # Knees (25, 26)
        knee_bend = 0.7 + 0.1 * np.sin(phase)
        if not correct:
            knee_bend += 0.15  # Incorrect: not deep enough
        landmarks[25*4:25*4+4] = [0.4, knee_bend, 0, 0.9]
        landmarks[26*4:26*4+4] = [0.6, knee_bend, 0, 0.9]
        
        # Ankles (27, 28)
        landmarks[27*4:27*4+4] = [0.4, 0.9, 0, 0.9]
        landmarks[28*4:28*4+4] = [0.6, 0.9, 0, 0.9]
        
        # Fill in remaining landmarks with reasonable values
        for j in range(33):
            if landmarks[j*4] == 0:  # Not set yet
                landmarks[j*4:j*4+4] = [
                    0.5 + np.random.randn() * 0.05,
                    0.5 + np.random.randn() * 0.05,
                    np.random.randn() * 0.1,
                    0.8 + np.random.rand() * 0.2
                ]
        
        sequence.append(landmarks)
    
    return np.array(sequence)

def generate_pushup_sequence(num_frames=60, correct=True):
    """Generate synthetic push-up sequence"""
    sequence = []
    
    for i in range(num_frames):
        landmarks = np.zeros(132)
        
        # Push-up motion (shoulders going up and down)
        phase = (i / num_frames) * 2 * np.pi
        shoulder_height = 0.6 + 0.15 * np.sin(phase)
        
        # Shoulders (11, 12)
        landmarks[11*4:11*4+4] = [0.4, shoulder_height, 0, 0.95]
        landmarks[12*4:12*4+4] = [0.6, shoulder_height, 0, 0.95]
        
        # Elbows (13, 14)
        elbow_bend = shoulder_height + 0.1
        landmarks[13*4:13*4+4] = [0.3, elbow_bend, 0, 0.9]
        landmarks[14*4:14*4+4] = [0.7, elbow_bend, 0, 0.9]
        
        # Wrists (15, 16)
        landmarks[15*4:15*4+4] = [0.3, 0.8, 0, 0.9]
        landmarks[16*4:16*4+4] = [0.7, 0.8, 0, 0.9]
        
        # Hips (23, 24)
        hip_height = shoulder_height + 0.05 if correct else shoulder_height + 0.2
        landmarks[23*4:23*4+4] = [0.4, hip_height, 0, 0.9]
        landmarks[24*4:24*4+4] = [0.6, hip_height, 0, 0.9]
        
        # Fill remaining
        for j in range(33):
            if landmarks[j*4] == 0:
                landmarks[j*4:j*4+4] = [
                    0.5 + np.random.randn() * 0.05,
                    0.5 + np.random.randn() * 0.05,
                    np.random.randn() * 0.1,
                    0.8 + np.random.rand() * 0.2
                ]
        
        sequence.append(landmarks)
    
    return np.array(sequence)

def generate_curl_sequence(num_frames=60, correct=True):
    """Generate synthetic bicep curl sequence"""
    sequence = []
    
    for i in range(num_frames):
        landmarks = np.zeros(132)
        
        # Curl motion (wrists going up and down)
        phase = (i / num_frames) * 2 * np.pi
        wrist_height = 0.6 + 0.2 * np.sin(phase)
        
        # Shoulders (11, 12) - should stay stable
        shoulder_movement = 0 if correct else 0.05 * np.sin(phase)
        landmarks[11*4:11*4+4] = [0.4, 0.3 + shoulder_movement, 0, 0.95]
        landmarks[12*4:12*4+4] = [0.6, 0.3 + shoulder_movement, 0, 0.95]
        
        # Elbows (13, 14)
        landmarks[13*4:13*4+4] = [0.35, 0.5, 0, 0.9]
        landmarks[14*4:14*4+4] = [0.65, 0.5, 0, 0.9]
        
        # Wrists (15, 16)
        landmarks[15*4:15*4+4] = [0.35, wrist_height, 0, 0.9]
        landmarks[16*4:16*4+4] = [0.65, wrist_height, 0, 0.9]
        
        # Hips (23, 24)
        landmarks[23*4:23*4+4] = [0.4, 0.5, 0, 0.9]
        landmarks[24*4:24*4+4] = [0.6, 0.5, 0, 0.9]
        
        # Fill remaining
        for j in range(33):
            if landmarks[j*4] == 0:
                landmarks[j*4:j*4+4] = [
                    0.5 + np.random.randn() * 0.05,
                    0.5 + np.random.randn() * 0.05,
                    np.random.randn() * 0.1,
                    0.8 + np.random.rand() * 0.2
                ]
        
        sequence.append(landmarks)
    
    return np.array(sequence)

def generate_dataset():
    """Generate complete synthetic dataset"""
    print("\n" + "="*60)
    print("GENERATING SYNTHETIC DATASET")
    print("="*60)
    
    # Create output directory
    os.makedirs('data/processed', exist_ok=True)
    
    exercises = [
        ('squat', generate_squat_sequence),
        ('pushup', generate_pushup_sequence),
        ('curl', generate_curl_sequence)
    ]
    
    samples_per_class = 10
    total_samples = 0
    
    for exercise_name, generator_func in exercises:
        print(f"\n📊 Generating {exercise_name} sequences...")
        
        for form in ['correct', 'incorrect']:
            is_correct = (form == 'correct')
            
            for i in range(samples_per_class):
                # Generate sequence
                sequence = generator_func(num_frames=60, correct=is_correct)
                
                # Create metadata
                metadata = {
                    'exercise': exercise_name,
                    'form': form,
                    'fps': 30.0,
                    'total_frames': 60,
                    'valid_frames': 60
                }
                
                # Save as .npz file
                filename = f"{exercise_name}_{form}_{i:03d}.npz"
                output_path = os.path.join('data/processed', filename)
                
                np.savez_compressed(
                    output_path,
                    landmarks=sequence,
                    metadata=json.dumps(metadata)
                )
                
                total_samples += 1
        
        print(f"  ✓ Generated {samples_per_class * 2} {exercise_name} samples")
    
    print("\n" + "="*60)
    print(f"✓ Generated {total_samples} synthetic samples")
    print(f"✓ Saved to: data/processed/")
    print("="*60)
    
    # Create summary
    print("\nDataset Summary:")
    print(f"  - Squats: {samples_per_class * 2} samples (correct + incorrect)")
    print(f"  - Push-ups: {samples_per_class * 2} samples")
    print(f"  - Bicep Curls: {samples_per_class * 2} samples")
    print(f"  - Total: {total_samples} sequences")
    print(f"  - Each sequence: 60 frames × 132 features")
    
    return total_samples

if __name__ == "__main__":
    total = generate_dataset()
    
    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    print("\n1. Test data collection:")
    print("   cd /home/knight/workout-monitoring-system")
    print("   source venv/bin/activate")
    print("   python -c 'from src.data_collection.collect_poses import *; print(\"✓ Import successful\")'")
    print("\n2. Test feature engineering:")
    print("   python -c 'from src.preprocessing.feature_engineering import *; print(\"✓ Features OK\")'")
    print("\n3. Run real-time demo (with webcam):")
    print("   python demo/realtime_monitor.py")
    print("\n" + "="*60)
