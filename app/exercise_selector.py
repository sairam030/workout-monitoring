"""
Exercise Selector — Tkinter dark-themed GUI
============================================
Presents 3 exercise cards. User clicks one, then the monitoring
window opens via app.monitor.run_monitor().

Run via:  python main.py
"""

import tkinter as tk
from tkinter import font as tkfont
import threading
import subprocess
import sys
import os


# ── Color palette ────────────────────────────────────────────────────────────
BG           = "#0f0f1a"
CARD_BG      = "#1a1a2e"
CARD_HOVER   = "#16213e"
CARD_BORDER  = "#0f3460"
ACCENT       = "#e94560"
ACCENT2      = "#0f3460"
TEXT_MAIN    = "#eaeaea"
TEXT_SUB     = "#9a9ab0"
BTN_BG       = "#e94560"
BTN_FG       = "#ffffff"
BTN_HOVER    = "#c73652"

EXERCISES = [
    {
        "name":  "Squat",
        "emoji": "🏋️",
        "desc":  "Tracks knee angle\nCounts full depth reps\nChecks knee alignment & torso lean",
        "muscles": "Quads • Glutes • Core",
        "color": "#e94560",
    },
    {
        "name":  "Push-up",
        "emoji": "💪",
        "desc":  "Tracks elbow angle\nCounts chest-to-floor reps\nChecks body line & arm width",
        "muscles": "Chest • Triceps • Shoulders",
        "color": "#0f3460",
    },
    {
        "name":  "Bicep Curl",
        "emoji": "🦾",
        "desc":  "Tracks elbow angle\nCounts curl reps\nChecks elbow stability & ROM",
        "muscles": "Biceps • Forearms",
        "color": "#533483",
    },
]


class ExerciseSelector:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("AI Workout Monitor")
        self.root.geometry("900x620")
        self.root.configure(bg=BG)
        self.root.resizable(False, False)

        # Center window
        self.root.eval('tk::PlaceWindow . center')

        self._selected = None
        self._build_ui()

    # ── UI construction ──────────────────────────────────────────────────────

    def _build_ui(self):
        # Header
        header = tk.Frame(self.root, bg=BG)
        header.pack(pady=(40, 10), fill="x")

        tk.Label(
            header, text="💪 AI Workout Monitor",
            font=("Helvetica", 28, "bold"),
            bg=BG, fg=TEXT_MAIN,
        ).pack()

        tk.Label(
            header,
            text="Select an exercise to begin real-time monitoring",
            font=("Helvetica", 13),
            bg=BG, fg=TEXT_SUB,
        ).pack(pady=(6, 0))

        # Divider
        tk.Frame(self.root, bg=ACCENT, height=2).pack(
            fill="x", padx=60, pady=(18, 24)
        )

        # Card row
        card_row = tk.Frame(self.root, bg=BG)
        card_row.pack(fill="both", expand=True, padx=40)

        self.cards = []
        for ex in EXERCISES:
            card = self._make_card(card_row, ex)
            card.pack(side="left", fill="both", expand=True, padx=12)
            self.cards.append(card)

        # Footer hint
        tk.Label(
            self.root,
            text="Press  Q  inside the monitoring window to quit · R  to reset reps",
            font=("Helvetica", 10),
            bg=BG, fg=TEXT_SUB,
        ).pack(pady=(20, 10))

    def _make_card(self, parent, ex: dict):
        """Create a clickable exercise card."""
        frame = tk.Frame(
            parent, bg=CARD_BG,
            highlightbackground=CARD_BORDER,
            highlightthickness=2,
            cursor="hand2",
        )

        # Emoji
        tk.Label(
            frame, text=ex["emoji"],
            font=("Helvetica", 40),
            bg=CARD_BG,
        ).pack(pady=(24, 0))

        # Name
        tk.Label(
            frame, text=ex["name"],
            font=("Helvetica", 18, "bold"),
            bg=CARD_BG, fg=TEXT_MAIN,
        ).pack(pady=(8, 0))

        # Muscles
        tk.Label(
            frame, text=ex["muscles"],
            font=("Helvetica", 9, "italic"),
            bg=CARD_BG, fg=ex["color"],
        ).pack(pady=(4, 10))

        # Colored accent bar
        tk.Frame(frame, bg=ex["color"], height=3).pack(fill="x", padx=20)

        # Description
        tk.Label(
            frame, text=ex["desc"],
            font=("Helvetica", 10),
            bg=CARD_BG, fg=TEXT_SUB,
            justify="center", wraplength=180,
        ).pack(pady=(14, 18))

        # Start button
        btn_color = ex["color"]
        btn = tk.Button(
            frame,
            text="▶  Start",
            font=("Helvetica", 12, "bold"),
            bg=btn_color, fg=BTN_FG,
            activebackground=BTN_HOVER,
            activeforeground=BTN_FG,
            relief="flat", bd=0,
            padx=20, pady=8,
            cursor="hand2",
            command=lambda e=ex["name"]: self._on_select(e),
        )
        btn.pack(pady=(0, 24), ipadx=10)

        # Hover effects
        def on_enter(event, f=frame, b=btn_color):
            f.configure(bg=CARD_HOVER, highlightbackground=b)

        def on_leave(event, f=frame):
            f.configure(bg=CARD_BG, highlightbackground=CARD_BORDER)

        frame.bind("<Enter>", on_enter)
        frame.bind("<Leave>", on_leave)

        return frame

    # ── Selection handler ────────────────────────────────────────────────────

    def _on_select(self, exercise: str):
        """Hide the selector and launch the monitor in a thread."""
        self._selected = exercise
        self.root.withdraw()   # hide (don't destroy — we may return to it)
        self._launch_monitor(exercise)

    def _launch_monitor(self, exercise: str):
        """Run the monitor in same process, then show selector again."""
        from app.monitor import run_monitor

        def _run():
            run_monitor(exercise)
            # After monitoring closes, show selector again
            self.root.deiconify()

        t = threading.Thread(target=_run, daemon=True)
        t.start()

    def run(self):
        self.root.mainloop()


def launch_selector():
    app = ExerciseSelector()
    app.run()
