#!/usr/bin/env python3
"""
Workout Monitoring System — Entry Point
=======================================

Run (from project root):
    python3 main.py           # auto-uses venv if present
    venv/bin/python3 main.py  # explicit venv

This opens the exercise selection screen.
Pick an exercise (Squat, Push-up, Bicep Curl) and click Start.
A real-time webcam monitoring window will open showing:
  • Pose skeleton overlay
  • Rep counter (state-machine based)
  • Form correctness feedback
  • Joint angle display

Press Q to quit the monitor and return to the selector.
Press R to reset the rep count.
"""

import sys
import os

# ── Auto-activate venv if packages aren't available ───────────────────────────
def _bootstrap_venv():
    """Re-execute this script inside the project venv if cv2 is not found."""
    try:
        import cv2  # noqa: F401 — just checking availability
        return  # already runnable
    except ImportError:
        pass

    project_root = os.path.dirname(os.path.abspath(__file__))
    venv_python  = os.path.join(project_root, "venv", "bin", "python3")

    if os.path.isfile(venv_python) and os.path.abspath(sys.executable) != os.path.abspath(venv_python):
        print(f"[info] cv2 not found in {sys.executable}")
        print(f"[info] Re-launching with venv: {venv_python}\n")
        os.execv(venv_python, [venv_python] + sys.argv)
        # os.execv replaces the process — code below never runs if it succeeds

    # If we reach here, venv python was not found or we're already in venv
    print("ERROR: 'cv2' (opencv-python) is not installed.")
    print("Install it with:  pip install opencv-python mediapipe")
    sys.exit(1)

_bootstrap_venv()

# Make sure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.exercise_selector import launch_selector

if __name__ == "__main__":
    launch_selector()
