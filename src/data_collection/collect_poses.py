"""
Data Collection Module
=====================

This module handles:
1. Video processing with MediaPipe Pose
2. Landmark extraction (33 keypoints × 4 features)
3. Dataset organization and saving

Why MediaPipe?
- Real-time performance (30+ FPS on CPU)
- 33 3D landmarks with visibility scores
- Robust to partial occlusion
- No GPU required
"""

import cv2
import mediapipe as mp
import numpy as np
import os
from pathlib import Path
import json
from tqdm import tqdm


class PoseDataCollector:
    """
    Collects pose landmarks from videos for training
    
    Architecture:
    Video → MediaPipe → Landmarks → Features → Save
    """
    
    def __init__(self, min_detection_confidence=0.7, min_tracking_confidence=0.7):
        """
        Initialize MediaPipe Pose
        
        Args:
            min_detection_confidence: Minimum confidence for pose detection
            min_tracking_confidence: Minimum confidence for pose tracking
            
        Why these values?
        - 0.7 confidence: Balance between detection rate and false positives
        - Lower values: More detections but noisier
        - Higher values: Fewer false positives but miss some frames
        """
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,  # Video mode for tracking
            model_complexity=1,        # 0=Lite, 1=Full, 2=Heavy
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
    def extract_landmarks(self, results):
        """
        Extract 33 landmarks × 4 features = 132 features per frame
        
        Landmarks (33 total):
        - 0: Nose
        - 11-12: Shoulders
        - 13-14: Elbows  
        - 15-16: Wrists
        - 23-24: Hips
        - 25-26: Knees
        - 27-28: Ankles
        
        Features per landmark:
        - x: Horizontal position [0, 1] normalized
        - y: Vertical position [0, 1] normalized  
        - z: Depth relative to hip (negative = closer)
        - visibility: Confidence score [0, 1]
        
        Returns:
            np.array: Shape (132,) - Flattened landmark features
        """
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
    
    def process_video(self, video_path, exercise_type, form_type='correct', 
                     visualize=False, max_frames=None):
        """
        Process video and extract pose sequences
        
        Args:
            video_path: Path to video file
            exercise_type: 'squat', 'pushup', 'bicep_curl'
            form_type: 'correct' or 'incorrect'
            visualize: Show processed video
            max_frames: Limit number of frames (for testing)
            
        Returns:
            dict: {
                'landmarks': np.array shape (n_frames, 132),
                'metadata': {
                    'exercise': str,
                    'form': str,
                    'fps': float,
                    'total_frames': int
                }
            }
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Video metadata
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"\nProcessing: {video_path}")
        print(f"Exercise: {exercise_type} | Form: {form_type}")
        print(f"FPS: {fps:.1f} | Total Frames: {total_frames}")
        
        landmarks_sequence = []
        frame_count = 0
        
        pbar = tqdm(total=max_frames or total_frames, desc="Extracting poses")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB for MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = self.pose.process(frame_rgb)
            
            # Extract landmarks
            landmarks = self.extract_landmarks(results)
            
            if landmarks is not None:
                landmarks_sequence.append(landmarks)
                
                # Visualization
                if visualize:
                    self.mp_drawing.draw_landmarks(
                        frame,
                        results.pose_landmarks,
                        self.mp_pose.POSE_CONNECTIONS,
                        self.mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                        self.mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                    )
                    
                    # Add text
                    cv2.putText(frame, f"Exercise: {exercise_type}", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, f"Form: {form_type}", (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, f"Frame: {frame_count}/{total_frames}", (10, 90),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    cv2.imshow('Pose Extraction', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            
            frame_count += 1
            pbar.update(1)
            
            if max_frames and frame_count >= max_frames:
                break
        
        pbar.close()
        cap.release()
        if visualize:
            cv2.destroyAllWindows()
        
        print(f"Extracted {len(landmarks_sequence)} frames with valid poses")
        
        return {
            'landmarks': np.array(landmarks_sequence),
            'metadata': {
                'exercise': exercise_type,
                'form': form_type,
                'fps': fps,
                'total_frames': frame_count,
                'valid_frames': len(landmarks_sequence)
            }
        }
    
    def save_dataset(self, data, output_dir, filename):
        """
        Save extracted landmarks to disk
        
        Format: .npz (compressed numpy)
        - Efficient storage
        - Fast loading
        - Preserves array structure
        """
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, filename)
        
        np.savez_compressed(
            output_path,
            landmarks=data['landmarks'],
            metadata=json.dumps(data['metadata'])
        )
        
        print(f"Saved to: {output_path}")
        return output_path
    
    def process_dataset(self, video_config, output_dir='data/processed', visualize=False):
        """
        Batch process multiple videos
        
        Args:
            video_config: List of dicts with video paths and labels
                [{
                    'path': 'path/to/video.mp4',
                    'exercise': 'squat',
                    'form': 'correct'
                }, ...]
            output_dir: Where to save processed data
            visualize: Show processing (slow)
            
        Returns:
            List of output file paths
        """
        output_files = []
        
        for i, config in enumerate(video_config):
            print(f"\n{'='*60}")
            print(f"Video {i+1}/{len(video_config)}")
            print(f"{'='*60}")
            
            # Process video
            data = self.process_video(
                video_path=config['path'],
                exercise_type=config['exercise'],
                form_type=config.get('form', 'correct'),
                visualize=visualize
            )
            
            # Generate filename
            filename = f"{config['exercise']}_{config.get('form', 'correct')}_{i:03d}.npz"
            
            # Save
            output_path = self.save_dataset(data, output_dir, filename)
            output_files.append(output_path)
        
        print(f"\n{'='*60}")
        print(f"✓ Processed {len(video_config)} videos")
        print(f"✓ Saved to: {output_dir}")
        print(f"{'='*60}")
        
        return output_files


def create_sample_dataset():
    """
    Example: Create dataset from sample videos
    
    Usage:
        python src/data_collection/collect_poses.py
    """
    # Example video configuration
    video_config = [
        # Squats
        {'path': 'data/raw/squats/correct_1.mp4', 'exercise': 'squat', 'form': 'correct'},
        {'path': 'data/raw/squats/correct_2.mp4', 'exercise': 'squat', 'form': 'correct'},
        {'path': 'data/raw/squats/incorrect_1.mp4', 'exercise': 'squat', 'form': 'incorrect'},
        
        # Push-ups
        {'path': 'data/raw/pushups/correct_1.mp4', 'exercise': 'pushup', 'form': 'correct'},
        {'path': 'data/raw/pushups/incorrect_1.mp4', 'exercise': 'pushup', 'form': 'incorrect'},
        
        # Bicep Curls
        {'path': 'data/raw/bicep_curls/correct_1.mp4', 'exercise': 'curl', 'form': 'correct'},
        {'path': 'data/raw/bicep_curls/incorrect_1.mp4', 'exercise': 'curl', 'form': 'incorrect'},
    ]
    
    # Initialize collector
    collector = PoseDataCollector()
    
    # Process all videos
    output_files = collector.process_dataset(
        video_config=video_config,
        output_dir='data/processed',
        visualize=False  # Set True to see processing
    )
    
    return output_files


if __name__ == "__main__":
    """
    Example usage:
    
    1. Place videos in data/raw/{exercise_type}/
    2. Run this script
    3. Processed landmarks saved to data/processed/
    """
    print("Starting Data Collection...")
    print("\nNote: Place your exercise videos in:")
    print("  - data/raw/squats/")
    print("  - data/raw/pushups/")
    print("  - data/raw/bicep_curls/")
    print("\nOr use webcam to record exercises.")
    
    # Uncomment to process sample dataset
    # output_files = create_sample_dataset()
    
    print("\n✓ Data collection module ready!")
    print("Import this module to use PoseDataCollector class")
