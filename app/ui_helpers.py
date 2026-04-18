"""
UI Helpers — shared drawing utilities for the monitoring overlay.
"""
import cv2
import numpy as np

# ── Color Palette (BGR) ─────────────────────────────────────────────────────
GOOD_GREEN   = (72, 199, 116)    # form correct
BAD_RED      = (56,  75, 226)    # form incorrect
ACCENT_BLUE  = (220, 155,  50)   # neutral info
WHITE        = (255, 255, 255)
BLACK        = (0,   0,   0)
DARK_PANEL   = (20,  20,  20)
YELLOW       = (0,  215, 255)    # reps count


def put_text_bg(frame, text, org, font=cv2.FONT_HERSHEY_SIMPLEX,
                scale=0.9, color=WHITE, thickness=2,
                bg_color=DARK_PANEL, pad=8):
    """Put text with a solid background rectangle."""
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = org
    cv2.rectangle(frame,
                  (x - pad, y - th - pad),
                  (x + tw + pad, y + baseline + pad),
                  bg_color, -1)
    cv2.putText(frame, text, (x, y), font, scale, color, thickness,
                cv2.LINE_AA)


def draw_rounded_rect(img, pt1, pt2, color, radius=15, thickness=-1, alpha=0.6):
    """Draw a semi-transparent rounded rectangle."""
    overlay = img.copy()
    x1, y1 = pt1
    x2, y2 = pt2
    r = radius
    # Fill with rounded corners using circle+rect approximation
    cv2.rectangle(overlay, (x1 + r, y1), (x2 - r, y2), color, thickness)
    cv2.rectangle(overlay, (x1, y1 + r), (x2, y2 - r), color, thickness)
    for cx, cy in [(x1+r, y1+r), (x2-r, y1+r), (x1+r, y2-r), (x2-r, y2-r)]:
        cv2.circle(overlay, (cx, cy), r, color, thickness)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def draw_angle_arc(frame, center, angle, max_angle=180,
                   radius=40, color=GOOD_GREEN, thickness=4):
    """Draw a partial circle representing a joint angle."""
    sweep = int((angle / max_angle) * 270)
    cv2.ellipse(frame, center, (radius, radius), 0, -135, -135 + sweep,
                color, thickness)


def draw_hud(frame, exercise: str, rep_count: int,
             is_correct: bool, messages: list,
             angle_info: dict = None, fps: float = 0.0):
    """
    Draw the complete HUD over the frame.

    Args:
        frame: OpenCV BGR frame (modified in-place)
        exercise: exercise name string
        rep_count: current rep count
        is_correct: form correctness flag
        messages: list of feedback strings
        angle_info: optional dict of {label: angle_value} to display
        fps: current frames-per-second
    """
    h, w = frame.shape[:2]

    # Scale HUD to fit smaller camera frames (e.g. 640x480) without clipping.
    ui_scale = max(0.60, min(1.0, h / 720.0))
    panel_w = max(210, min(300, int(w * 0.34)))
    panel_h = h

    title_y = max(28, int(50 * ui_scale))
    reps_label_y = max(title_y + 28, int(110 * ui_scale))
    reps_value_y = max(reps_label_y + 36, int(185 * ui_scale))
    status_top = reps_value_y + max(12, int(14 * ui_scale))
    status_bottom = status_top + max(28, int(40 * ui_scale))
    feedback_start_y = status_bottom + max(18, int(22 * ui_scale))

    # Semi-transparent left panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

    # ── Exercise Title ────────────────────────────────────────────────────
    cv2.putText(frame, exercise.upper(), (14, title_y),
                cv2.FONT_HERSHEY_DUPLEX, 0.80 * ui_scale + 0.20,
                ACCENT_BLUE, max(1, int(2 * ui_scale)), cv2.LINE_AA)

    # ── Rep Count ────────────────────────────────────────────────────────
    cv2.putText(frame, "REPS", (14, reps_label_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52 * ui_scale + 0.20,
                (180, 180, 180), max(1, int(1 * ui_scale)), cv2.LINE_AA)
    cv2.putText(frame, str(rep_count), (14, reps_value_y),
                cv2.FONT_HERSHEY_DUPLEX, 2.2 * ui_scale + 0.5,
                YELLOW, max(2, int(4 * ui_scale)), cv2.LINE_AA)

    # ── Form Status ───────────────────────────────────────────────────────
    form_color = GOOD_GREEN if is_correct else BAD_RED
    form_label = "✓  GOOD FORM" if is_correct else "✗  FIX FORM"
    cv2.rectangle(frame, (10, status_top), (panel_w - 10, status_bottom), form_color, -1)
    cv2.putText(frame, form_label, (16, status_bottom - max(8, int(10 * ui_scale))),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58 * ui_scale + 0.22,
                BLACK, max(1, int(2 * ui_scale)), cv2.LINE_AA)

    # ── Feedback Messages ─────────────────────────────────────────────────
    y = feedback_start_y
    feedback_step = max(20, int(28 * ui_scale))
    feedback_font = 0.40 * ui_scale + 0.20
    for msg in messages[:4]:
        if y > h - 120:
            break
        cv2.putText(frame, f"- {msg}", (12, y),
                    cv2.FONT_HERSHEY_SIMPLEX, feedback_font,
                    (210, 210, 255), max(1, int(1 * ui_scale)), cv2.LINE_AA)
        y += feedback_step

    # ── Joint Angles ─────────────────────────────────────────────────────
    if angle_info:
        y_a = min(max(y + 8, int(340 * ui_scale)), h - 120)
        cv2.putText(frame, "JOINT ANGLES", (12, y_a),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45 * ui_scale + 0.20,
                    (150, 150, 150), max(1, int(1 * ui_scale)), cv2.LINE_AA)
        y_a += max(16, int(22 * ui_scale))
        bar_h = max(8, int(14 * ui_scale))
        block_h = max(28, int(44 * ui_scale))
        angle_font = 0.38 * ui_scale + 0.20
        for label, val in angle_info.items():
            if y_a + block_h > h - 26:
                break
            bar_val = int(min(val, 180) / 180 * (panel_w - 28))
            cv2.rectangle(frame, (12, y_a), (panel_w - 12, y_a + bar_h),
                          (40, 40, 40), -1)
            cv2.rectangle(frame, (12, y_a), (12 + bar_val, y_a + bar_h),
                          ACCENT_BLUE, -1)
            cv2.putText(frame, f"{label}: {val:.0f} deg", (12, y_a + bar_h + max(10, int(14 * ui_scale))),
                        cv2.FONT_HERSHEY_SIMPLEX, angle_font, WHITE,
                        max(1, int(1 * ui_scale)), cv2.LINE_AA)
            y_a += block_h

    # ── FPS ───────────────────────────────────────────────────────────────
    cv2.putText(frame, f"FPS {fps:.1f}", (12, h - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40 * ui_scale + 0.20,
                (100, 100, 100), max(1, int(1 * ui_scale)), cv2.LINE_AA)

    # ── Keymap hint ───────────────────────────────────────────────────────
    hint_x = max(panel_w + 8, w - int(190 * ui_scale))
    cv2.putText(frame, "Q=quit  R=reset", (hint_x, h - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38 * ui_scale + 0.20,
                (120, 120, 120), max(1, int(1 * ui_scale)), cv2.LINE_AA)

    return frame
