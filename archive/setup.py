#!/usr/bin/env python3
"""
Setup Script for Workout Monitoring System
==========================================

Automates project setup and verification
"""

import os
import sys
import subprocess


def print_header(text):
    """Print formatted header"""
    print("\n" + "="*60)
    print(text)
    print("="*60)


def check_python_version():
    """Check Python version >= 3.8"""
    print_header("Checking Python Version")
    
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ required")
        return False
    
    print("✓ Python version compatible")
    return True


def create_directories():
    """Create project directories"""
    print_header("Creating Directory Structure")
    
    directories = [
        'data/raw/squats',
        'data/raw/pushups',
        'data/raw/bicep_curls',
        'data/processed',
        'data/augmented',
        'models',
        'results/plots',
        'results/reports',
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created: {directory}")
    
    print("\n✓ All directories created")
    return True


def install_dependencies():
    """Install Python dependencies"""
    print_header("Installing Dependencies")
    
    print("Installing from requirements.txt...")
    print("(This may take a few minutes)\n")
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print("\n✓ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("\n❌ Failed to install dependencies")
        print("Try manually: pip install -r requirements.txt")
        return False


def verify_imports():
    """Verify key imports"""
    print_header("Verifying Installations")
    
    imports = {
        'cv2': 'OpenCV',
        'mediapipe': 'MediaPipe',
        'tensorflow': 'TensorFlow',
        'sklearn': 'Scikit-learn',
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'matplotlib': 'Matplotlib',
        'streamlit': 'Streamlit'
    }
    
    all_good = True
    
    for module, name in imports.items():
        try:
            __import__(module)
            print(f"✓ {name}")
        except ImportError:
            print(f"❌ {name} - not found")
            all_good = False
    
    return all_good


def test_mediapipe():
    """Test MediaPipe Pose"""
    print_header("Testing MediaPipe Pose")
    
    try:
        import mediapipe as mp
        
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        print("✓ MediaPipe Pose initialized successfully")
        pose.close()
        return True
    except Exception as e:
        print(f"❌ MediaPipe test failed: {e}")
        return False


def create_sample_script():
    """Create quick test script"""
    print_header("Creating Test Script")
    
    script_content = '''#!/usr/bin/env python3
"""Quick test of MediaPipe Pose"""

import cv2
import mediapipe as mp
import numpy as np

print("Testing MediaPipe Pose...")

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Create dummy image
dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)

# Process
results = pose.process(dummy_image)

print("✓ MediaPipe working!")
print(f"  Landmarks detected: {results.pose_landmarks is not None}")

pose.close()
'''
    
    with open('test_mediapipe.py', 'w') as f:
        f.write(script_content)
    
    os.chmod('test_mediapipe.py', 0o755)
    print("✓ Created: test_mediapipe.py")
    print("  Run with: python test_mediapipe.py")
    return True


def print_next_steps():
    """Print next steps"""
    print_header("Setup Complete! 🎉")
    
    print("""
✓ Project setup successful!

📋 Next Steps:

1. Collect Exercise Videos:
   - Place videos in: data/raw/{exercise_type}/
   - Example: data/raw/squats/correct_1.mp4

2. Process Videos:
   python src/data_collection/collect_poses.py

3. Run Real-time Monitor:
   python demo/realtime_monitor.py

4. Launch Web Demo:
   streamlit run demo/app.py

5. Follow Jupyter Notebooks:
   - notebooks/01_data_collection.ipynb
   - notebooks/02_eda.ipynb
   - (continue through all notebooks)

📚 Documentation:
   - README.md: Complete documentation
   - GUIDE.md: Step-by-step guide

🔧 Test Installation:
   python test_mediapipe.py

💪 Ready to start building your AI workout trainer!
    """)


def main():
    """Main setup routine"""
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║       WORKOUT MONITORING SYSTEM - SETUP SCRIPT           ║
    ║                                                           ║
    ║         Multimedia Analytics Project Setup               ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    steps = [
        ("Check Python version", check_python_version),
        ("Create directories", create_directories),
        ("Install dependencies", install_dependencies),
        ("Verify installations", verify_imports),
        ("Test MediaPipe", test_mediapipe),
        ("Create test script", create_sample_script),
    ]
    
    results = []
    
    for step_name, step_func in steps:
        try:
            success = step_func()
            results.append((step_name, success))
            if not success:
                print(f"\n⚠️  {step_name} had issues but continuing...")
        except Exception as e:
            print(f"\n❌ {step_name} failed: {e}")
            results.append((step_name, False))
    
    # Summary
    print_header("Setup Summary")
    
    for step_name, success in results:
        status = "✓" if success else "❌"
        print(f"{status} {step_name}")
    
    all_success = all(success for _, success in results)
    
    if all_success:
        print_next_steps()
    else:
        print("\n⚠️  Some steps failed. Please check errors above.")
        print("You can still proceed, but some features may not work.")


if __name__ == "__main__":
    main()
