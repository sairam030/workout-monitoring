#!/usr/bin/env python3
"""
Download Real Exercise Videos
==============================

Downloads real human exercise videos from public sources for pose estimation.
"""

import os
import subprocess
import shutil

def download_with_curl():
    """Download exercise videos using curl"""
    
    print("\n" + "="*70)
    print("DOWNLOADING REAL EXERCISE VIDEOS")
    print("="*70)
    
    # Sample videos from Pexels (free to use)
    video_sources = [
        {
            'url': 'https://videos.pexels.com/video-files/4569808/4569808-sd_640_360_25fps.mp4',
            'output': 'data/raw/squats/squat_real_01.mp4',
            'exercise': 'squats',
            'description': 'Woman doing squats'
        },
        {
            'url': 'https://videos.pexels.com/video-files/5319665/5319665-sd_640_360_25fps.mp4',
            'output': 'data/raw/squats/squat_real_02.mp4',
            'exercise': 'squats',
            'description': 'Squat exercise'
        },
        {
            'url': 'https://videos.pexels.com/video-files/4662343/4662343-sd_640_360_25fps.mp4',
            'output': 'data/raw/pushups/pushup_real_01.mp4',
            'exercise': 'pushups',
            'description': 'Woman doing pushups'
        },
        {
            'url': 'https://videos.pexels.com/video-files/4557611/4557611-sd_640_360_25fps.mp4',
            'output': 'data/raw/pushups/pushup_real_02.mp4',
            'exercise': 'pushups',
            'description': 'Pushup exercise'
        },
        {
            'url': 'https://videos.pexels.com/video-files/4569729/4569729-sd_640_360_25fps.mp4',
            'output': 'data/raw/bicep_curls/curl_real_01.mp4',
            'exercise': 'bicep_curls',
            'description': 'Bicep curl with dumbbell'
        },
        {
            'url': 'https://videos.pexels.com/video-files/5319657/5319657-sd_640_360_25fps.mp4',
            'output': 'data/raw/bicep_curls/curl_real_02.mp4',
            'exercise': 'bicep_curls',
            'description': 'Arm curl exercise'
        },
    ]
    
    print(f"\nDownloading {len(video_sources)} exercise videos...")
    print("Source: Pexels.com (free to use)")
    print("-"*70)
    
    downloaded = []
    
    for i, video in enumerate(video_sources, 1):
        print(f"\n[{i}/{len(video_sources)}] {video['description']}")
        print(f"  Exercise: {video['exercise']}")
        print(f"  Downloading...", end=' ', flush=True)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(video['output']), exist_ok=True)
        
        try:
            # Download with curl
            result = subprocess.run([
                'curl', '-L',  # Follow redirects
                '-o', video['output'],
                '-s',  # Silent mode
                '--max-time', '120',  # 2 minute timeout
                '--connect-timeout', '30',
                video['url']
            ], capture_output=True, timeout=150)
            
            if result.returncode == 0 and os.path.exists(video['output']):
                file_size = os.path.getsize(video['output']) / (1024 * 1024)
                if file_size > 0.1:  # At least 100KB
                    print(f"✓ ({file_size:.2f} MB)")
                    downloaded.append(video['output'])
                else:
                    print(f"✗ (file too small: {file_size:.2f} MB)")
                    if os.path.exists(video['output']):
                        os.remove(video['output'])
            else:
                print("✗ (download failed)")
                
        except Exception as e:
            print(f"✗ (error: {e})")
    
    print("\n" + "="*70)
    print("DOWNLOAD COMPLETE")
    print("="*70)
    print(f"\n✓ Successfully downloaded: {len(downloaded)}/{len(video_sources)} videos")
    
    if downloaded:
        print("\n📁 Downloaded videos:")
        total_size = 0
        for filepath in downloaded:
            size = os.path.getsize(filepath) / (1024 * 1024)
            total_size += size
            print(f"  - {os.path.basename(filepath)} ({size:.2f} MB)")
        
        print(f"\nTotal size: {total_size:.2f} MB")
        print("\n✅ Ready to process with MediaPipe!")
        print("Next: python process_real_videos.py")
        return True
    else:
        print("\n⚠️  No videos were downloaded successfully")
        print("This might be due to:")
        print("  - Network connectivity issues")
        print("  - URL changes (Pexels video IDs may change)")
        print("\nAlternative: Manually download exercise videos and place in:")
        print("  - data/raw/squats/")
        print("  - data/raw/pushups/")
        print("  - data/raw/bicep_curls/")
        return False

if __name__ == "__main__":
    success = download_with_curl()
    
    if not success:
        print("\n💡 TIP: You can also use these alternatives:")
        print("  1. Record your own exercise videos with a phone/webcam")
        print("  2. Download from YouTube using: yt-dlp <URL> -o filename.mp4")
        print("  3. Use stock footage from Pexels.com, Pixabay.com")
