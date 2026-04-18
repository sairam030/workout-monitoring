#!/usr/bin/env python3
"""
Download Real Exercise Videos for Training
==========================================

Downloads actual exercise demonstration videos from YouTube
for squats, push-ups, and bicep curls.
"""

import os
import subprocess
import sys

# Sample exercise video URLs (using shorter fitness demonstration videos)
VIDEO_URLS = {
    'squats': [
        'https://www.youtube.com/watch?v=aclHkVaku9U',  # How to Squat
        'https://www.youtube.com/watch?v=ultWZbUMPL8',  # Squat form
    ],
    'pushups': [
        'https://www.youtube.com/watch?v=IODxDxX7oi4',  # Perfect Pushup
        'https://www.youtube.com/watch?v=_l3ySVKYVJ8',  # Pushup technique
    ],
    'bicep_curls': [
        'https://www.youtube.com/watch?v=ykJmrZ5v0Oo',  # Bicep Curl form
        'https://www.youtube.com/watch?v=in7PaeYlhrM',  # Dumbbell Curls
    ]
}

def download_video(url, output_dir, exercise_name, video_num):
    """Download single video using yt-dlp"""
    
    output_template = os.path.join(output_dir, f'{exercise_name}_{video_num:02d}.%(ext)s')
    
    cmd = [
        'yt-dlp',
        '-f', 'best[height<=480]',  # Lower quality for faster processing
        '--max-filesize', '50M',     # Limit file size
        '--no-playlist',
        '-o', output_template,
        url
    ]
    
    try:
        print(f"  Downloading {exercise_name} video {video_num}...")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        
        if result.returncode == 0:
            print(f"  ✓ Downloaded successfully")
            return True
        else:
            print(f"  ✗ Failed: {result.stderr[:100]}")
            return False
    except subprocess.TimeoutExpired:
        print(f"  ✗ Timeout")
        return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

def download_all_videos():
    """Download all exercise videos"""
    
    print("\n" + "="*70)
    print("DOWNLOADING REAL EXERCISE VIDEOS")
    print("="*70)
    
    # Create directories
    for exercise in VIDEO_URLS.keys():
        os.makedirs(f'data/raw/{exercise}', exist_ok=True)
    
    total_downloaded = 0
    total_failed = 0
    
    for exercise, urls in VIDEO_URLS.items():
        print(f"\n📹 Downloading {exercise.upper()} videos...")
        print("-"*70)
        
        for i, url in enumerate(urls, 1):
            output_dir = f'data/raw/{exercise}'
            success = download_video(url, output_dir, exercise, i)
            
            if success:
                total_downloaded += 1
            else:
                total_failed += 1
    
    print("\n" + "="*70)
    print("DOWNLOAD SUMMARY")
    print("="*70)
    print(f"  ✓ Successfully downloaded: {total_downloaded}")
    print(f"  ✗ Failed: {total_failed}")
    print("="*70)
    
    # Check what we have
    print("\n📂 Downloaded files:")
    for exercise in VIDEO_URLS.keys():
        dir_path = f'data/raw/{exercise}'
        if os.path.exists(dir_path):
            files = [f for f in os.listdir(dir_path) if f.endswith(('.mp4', '.webm', '.mkv'))]
            print(f"  {exercise}: {len(files)} files")
            for f in files:
                size = os.path.getsize(os.path.join(dir_path, f)) / (1024*1024)
                print(f"    - {f} ({size:.1f} MB)")
    
    return total_downloaded > 0

if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║                                                                  ║
    ║           REAL EXERCISE VIDEO DOWNLOADER                        ║
    ║                                                                  ║
    ╚══════════════════════════════════════════════════════════════════╝
    
    This script will download sample exercise videos from YouTube.
    
    Note: Downloads may take several minutes depending on your internet speed.
    """)
    
    try:
        success = download_all_videos()
        
        if success:
            print("\n✅ Videos downloaded successfully!")
            print("\nNext steps:")
            print("  1. Run: python process_real_videos.py")
            print("  2. Follow notebooks for complete analysis")
        else:
            print("\n⚠️  No videos were downloaded")
            print("\nAlternative: Use your own videos")
            print("  - Place MP4 files in data/raw/{exercise}/")
            print("  - Then run: python process_real_videos.py")
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Download cancelled by user")
        sys.exit(1)
