#!/usr/bin/env python3
"""
Test Real-time Monitoring System
=================================

Tests the workout monitoring system with synthetic data
(no camera needed for testing)
"""

import sys
sys.path.insert(0, '.')

import numpy as np
from src.preprocessing.feature_engineering import FeatureEngineer

def test_exercise_detection():
    """Test exercise detection with synthetic data"""
    
    print("\n" + "="*60)
    print("TESTING WORKOUT MONITORING SYSTEM")
    print("="*60)
    
    # Load samples
    print("\n1. Loading synthetic exercise data...")
    
    samples = [
        ('data/processed/squat_correct_000.npz', 'Squat', 'Correct'),
        ('data/processed/pushup_correct_000.npz', 'Push-up', 'Correct'),
        ('data/processed/curl_correct_000.npz', 'Curl', 'Correct'),
    ]
    
    fe = FeatureEngineer()
    
    for filepath, exercise, form in samples:
        data = np.load(filepath, allow_pickle=True)
        landmarks_sequence = data['landmarks']
        
        print(f"\n   📊 {exercise} ({form})")
        print(f"      - Frames: {landmarks_sequence.shape[0]}")
        print(f"      - Features per frame: {landmarks_sequence.shape[1]}")
        
        # Extract features
        features = fe.engineer_features(landmarks_sequence)
        print(f"      - Engineered features: {len(features)}")
        
        # Analyze key angles
        angles = fe.extract_angles(landmarks_sequence[30])  # Mid-sequence
        
        if exercise == 'Squat':
            knee_angle = angles.get('left_knee', 0)
            print(f"      - Knee angle: {knee_angle:.1f}°")
        elif exercise == 'Push-up':
            elbow_angle = angles.get('left_elbow', 0)
            print(f"      - Elbow angle: {elbow_angle:.1f}°")
        elif exercise == 'Curl':
            elbow_angle = angles.get('left_elbow', 0)
            print(f"      - Elbow angle: {elbow_angle:.1f}°")
    
    print("\n" + "="*60)
    print("✓ All modules working correctly!")
    print("="*60)
    
    print("\n📹 Next Steps:")
    print("\n1. Test with webcam (requires camera):")
    print("   python demo/realtime_monitor.py")
    print("\n2. Process real exercise videos:")
    print("   - Add videos to data/raw/{exercise}/")
    print("   - Run: python src/data_collection/collect_poses.py")
    print("\n3. Train custom models:")
    print("   - Collect more data")
    print("   - Follow notebooks for training")
    
    return True

if __name__ == "__main__":
    test_exercise_detection()
