#!/usr/bin/env python3
"""
Download Sample Exercise Videos
================================

Downloads sample exercise videos for testing the workout monitoring system.
Uses freely available exercise demonstration videos.
"""

import os
import urllib.request
import sys

def download_file(url, output_path):
    """Download file with progress"""
    print(f"Downloading: {os.path.basename(output_path)}")
    try:
        urllib.request.urlretrieve(url, output_path)
        print(f"✓ Downloaded to: {output_path}")
        return True
    except Exception as e:
        print(f"✗ Failed to download: {e}")
        return False

def create_sample_videos():
    """
    Create sample exercise videos using stock footage
    
    Note: For this demo, we'll create placeholder videos that can be
    replaced with actual exercise videos from:
    - YouTube (download with yt-dlp)
    - Stock footage sites
    - Your own recordings
    """
    
    print("\n" + "="*60)
    print("SAMPLE VIDEO SETUP")
    print("="*60)
    
    # Create directories
    os.makedirs('data/raw/squats', exist_ok=True)
    os.makedirs('data/raw/pushups', exist_ok=True)
    os.makedirs('data/raw/bicep_curls', exist_ok=True)
    
    print("\n📹 To use the full system, add exercise videos to:")
    print("  - data/raw/squats/")
    print("  - data/raw/pushups/")
    print("  - data/raw/bicep_curls/")
    
    print("\n💡 Video Sources:")
    print("  1. Record yourself doing exercises")
    print("  2. Download from YouTube with yt-dlp:")
    print("     yt-dlp -f 'best[height<=720]' -o 'data/raw/squats/%(title)s.%(ext)s' <URL>")
    print("  3. Use stock exercise footage")
    print("  4. Use the webcam for real-time demo (no videos needed!)")
    
    print("\n✅ For now, you can test with:")
    print("  python demo/realtime_monitor.py  # Uses webcam directly!")
    
    return True

if __name__ == "__main__":
    create_sample_videos()
    
    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    print("\n1. Add exercise videos to data/raw/ OR")
    print("2. Run real-time demo with webcam:")
    print("   python demo/realtime_monitor.py")
    print("\n" + "="*60)
