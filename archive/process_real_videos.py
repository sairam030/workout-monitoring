#!/usr/bin/env python3
"""
Process Real Exercise Videos
=============================

Extracts pose landmarks from real exercise videos using MediaPipe.
This is Step 2 of the complete pipeline.
"""

import sys
sys.path.insert(0, '.')

from src.data_collection.collect_poses import PoseDataCollector
import os
import glob

def process_all_videos():
    """Process all videos in data/raw/"""
    
    print("\n" + "="*70)
    print("PROCESSING EXERCISE VIDEOS WITH MEDIAPIPE")
    print("="*70)
    
    # Initialize pose collector
    collector = PoseDataCollector(
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )
    
    # Find all videos
    exercises = ['squats', 'pushups', 'bicep_curls']
    video_configs = []
    
    for exercise in exercises:
        video_dir = f'data/raw/{exercise}'
        if os.path.exists(video_dir):
            videos = glob.glob(os.path.join(video_dir, '*.mp4'))
            for video_path in videos:
                filename = os.path.basename(video_path)
                form = 'correct' if 'correct' in filename.lower() else 'incorrect'
                
                video_configs.append({
                    'path': video_path,
                    'exercise': exercise.replace('_', ' '),
                    'form': form
                })
    
    if not video_configs:
        print("\n❌ No videos found in data/raw/")
        print("Run: python create_sample_videos.py first")
        return False
    
    print(f"\nFound {len(video_configs)} videos to process")
    print("-"*70)
    
    # Process each video
    processed_files = []
    
    for i, config in enumerate(video_configs, 1):
        print(f"\n[{i}/{len(video_configs)}] Processing: {os.path.basename(config['path'])}")
        print(f"  Exercise: {config['exercise']}")
        print(f"  Form: {config['form']}")
        
        try:
            # Process video
            data = collector.process_video(
                video_path=config['path'],
                exercise_type=config['exercise'],
                form_type=config['form'],
                visualize=False  # Set True to see processing
            )
            
            # Generate filename
            exercise_clean = config['exercise'].replace(' ', '_')
            form_clean = config['form']
            video_num = i
            filename = f"{exercise_clean}_{form_clean}_{video_num:03d}.npz"
            
            # Save
            output_path = collector.save_dataset(data, 'data/processed', filename)
            processed_files.append(output_path)
            
            print(f"  ✓ Extracted {data['landmarks'].shape[0]} frames")
            print(f"  ✓ Saved to: {filename}")
            
        except Exception as e:
            print(f"  ✗ Error processing video: {e}")
            continue
    
    print("\n" + "="*70)
    print("PROCESSING COMPLETE!")
    print("="*70)
    print(f"\n✓ Successfully processed: {len(processed_files)}/{len(video_configs)} videos")
    print(f"✓ Output directory: data/processed/")
    
    # Show statistics
    print("\n📊 Dataset Statistics:")
    import numpy as np
    
    total_frames = 0
    for filepath in processed_files:
        data = np.load(filepath, allow_pickle=True)
        total_frames += len(data['landmarks'])
    
    print(f"  - Total sequences: {len(processed_files)}")
    print(f"  - Total frames: {total_frames}")
    print(f"  - Features per frame: 132 (33 landmarks × 4 values)")
    print(f"  - Average frames per sequence: {total_frames // len(processed_files) if processed_files else 0}")
    
    print("\n✅ Ready for next step!")
    print("Next: Open notebooks/02_eda_detailed.ipynb for analysis")
    
    return len(processed_files) > 0

if __name__ == "__main__":
    success = process_all_videos()
    
    if not success:
        print("\n⚠️  Processing failed or no videos found")
        sys.exit(1)
