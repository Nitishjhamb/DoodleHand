import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time

# ───────── CONFIG ─────────
FINGER_UP_TH = 0.05
SMOOTHING = 0.65
UI_WIDTH = 100                  # left panel for UI gestures
GESTURE_HOLD_TIME = 1.0         # hold sec for CLR/SAVE
ERASER_RADIUS = 40              # duster size
ERASER_THICKNESS = -1           # -1 = filled circle (proper duster/rubber effect)

COLORS = [
    (255, 59, 48),    # red-ish
    (52, 199, 89),    # green-ish
    (0, 122, 255),    # blue
    (255, 214, 10),   # yellow
    (175, 82, 222)    # purple
]

# ───────── CAMERA SETUP ─────────
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
canvas = np.ones((h, w, 3), np.uint8) * 255

# ───────── STATE ─────────
prev_x = 0
prev_y = 0
has_prev = False

color = COLORS[2]           # start with blue
thickness = 8
stroke = []
erasing = False

# Hold timers for UI actions
fist_hold_start = None
palm_hold_start = None

# ───────── UTILITIES ─────────
def finger_up(lm, tip, base):
    return lm[tip].y < lm[base].y - FINGER_UP_TH

def fingers(lm):
    return (
        lm[4].x < lm[3].x,                # thumb
        finger_up(lm, 8, 5),              # index
        finger_up(lm, 12, 9),             # middle
        finger_up(lm, 16, 13),            # ring
        finger_up(lm, 20, 17)             # pinky
    )

def is_fist(thumb, index, middle, ring, pinky):
    return not (thumb or index or middle or ring or pinky)

def is_open_palm(thumb, index, middle, ring, pinky):
    return thumb and index and middle and ring and pinky

# ───────── AUTO-PERFECT SHAPE ─────────
def auto_perfect(points):
    if len(points) < 12:
        return

    cnt = np.array(points, np.int32)
    peri = cv2.arcLength(cnt, True)
    if peri == 0:
        return
    approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
    x, y, w0, h0 = cv2.boundingRect(cnt)

    if len(approx) == 3:
        cv2.drawContours(canvas, [approx], -1, color, thickness)
    elif len(approx) == 4:
        cv2.rectangle(canvas, (x, y), (x + w0, y + h0), color, thickness)
    else:
        area = cv2.contourArea(cnt)
        if area <= 0:
            return
        circ = 4 * np.pi * area / (peri * peri)
        if circ > 0.72:
            cx = x + w0 // 2
            cy = y + h0 // 2
            r = max(w0, h0) // 2
            cv2.circle(canvas, (cx, cy), r, color, thickness)

# ───────── UI ─────────
def draw_ui(img):
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (UI_WIDTH, h), (30, 30, 30), -1)
    img[:] = cv2.addWeighted(overlay, 0.88, img, 0.12, 0)

    for i, c in enumerate(COLORS):
        cy = 50 + i * 70
        cv2.circle(img, (UI_WIDTH//2, cy), 24, c, -1)
        cv2.circle(img, (UI_WIDTH//2, cy), 26, (220,220,220), 2)

    cv2.putText(img, "CLR", (20, h - 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200,200,255), 2)
    cv2.putText(img, "SAVE", (15, h - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200,255,200), 2)

def handle_ui_gestures(cx, cy, thumb, index, middle, ring, pinky):
    global color, canvas, fist_hold_start, palm_hold_start

    if cx >= UI_WIDTH:
        fist_hold_start = None
        palm_hold_start = None
        return

    now = time.time()

    # Color selection by hovering
    if index:
        for i, c in enumerate(COLORS):
            cy_circle = 50 + i * 70
            if abs(cx - UI_WIDTH//2) < 35 and abs(cy - cy_circle) < 35:
                color = c
                cv2.circle(frame, (cx, cy), 35, (255,255,255), 3)

    # Hold fist → clear
    if is_fist(thumb, index, middle, ring, pinky):
        if fist_hold_start is None:
            fist_hold_start = now
        elif now - fist_hold_start >= GESTURE_HOLD_TIME:
            canvas[:] = 255
            fist_hold_start = None
            cv2.putText(frame, "CLEARED", (w//2-150, h//2), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 5)
    else:
        fist_hold_start = None

    # Hold open palm in UI → save
    if is_open_palm(thumb, index, middle, ring, pinky):
        if palm_hold_start is None:
            palm_hold_start = now
        elif now - palm_hold_start >= GESTURE_HOLD_TIME:
            filename = f"drawing_{int(now)}.png"
            cv2.imwrite(filename, canvas)
            palm_hold_start = None
            cv2.putText(frame, f"SAVED {filename[-12:]}", (w//2-200, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0,255,0), 4)
    else:
        palm_hold_start = None

# ───────── HAND LANDMARKER SETUP ─────────
model_path = "hand_landmarker.task"

try:
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6
    )
    landmarker = vision.HandLandmarker.create_from_options(options)
except Exception as e:
    print(f"Model error: {e}")
    exit()

# ───────── MAIN LOOP ─────────
frame_timestamp = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    frame_timestamp += 1
    detection_result = landmarker.detect_for_video(mp_image, frame_timestamp)

    erasing = False

    if detection_result.hand_landmarks:
        hand_landmarks_list = detection_result.hand_landmarks[0]

        # ── Subtle visual feedback: small dots on all landmarks ──
        for lm in hand_landmarks_list:
            x = int(lm.x * w)
            y = int(lm.y * h)
            cv2.circle(frame, (x, y), 5, (0, 255, 100), -1)

        # ── Active finger indicator (only one big circle) ──
        index_tip = hand_landmarks_list[8]
        ix = int(index_tip.x * w)
        iy = int(index_tip.y * h)

        # ── Finger states ──
        lm = hand_landmarks_list
        thumb, index, middle, ring, pinky = fingers(lm)

        cx, cy = ix, iy

        if has_prev:
            cx = int(SMOOTHING * cx + (1 - SMOOTHING) * prev_x)
            cy = int(SMOOTHING * cy + (1 - SMOOTHING) * prev_y)

        # ── Eraser mode: open palm ──
        erasing = is_open_palm(thumb, index, middle, ring, pinky)

        if erasing:
            cv2.putText(frame, "DUSTER ERASER", (w//2 - 220, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, (200, 200, 200), 4)
            # Big gray/white circle for duster
            cv2.circle(frame, (cx, cy), ERASER_RADIUS + 10, (180, 180, 180), 4)
            cv2.circle(frame, (cx, cy), ERASER_RADIUS, (255, 255, 255), -1)
        elif index and not (middle or ring or pinky):
            # Normal drawing mode: red circle on index finger
            cv2.circle(frame, (cx, cy), 18, (0, 0, 255), -1)
            cv2.circle(frame, (cx, cy), 22, (0, 0, 255), 3)
        else:
            # Idle hand: smaller neutral circle
            cv2.circle(frame, (cx, cy), 12, (200, 200, 200), -1)

        # ── UI gestures ──
        handle_ui_gestures(cx, cy, thumb, index, middle, ring, pinky)

        # ── Drawing / Erasing ──
        if cx >= UI_WIDTH:
            if erasing:
                # Circular erase (duster style)
                cv2.circle(canvas, (cx, cy), ERASER_RADIUS, (255, 255, 255), ERASER_THICKNESS)
                has_prev = True
            elif index and not (middle or ring or pinky):
                # Normal draw
                if has_prev:
                    cv2.line(canvas, (prev_x, prev_y), (cx, cy), color, thickness)
                    stroke.append((cx, cy))
                has_prev = True
            else:
                if stroke:
                    auto_perfect(stroke)
                    stroke.clear()
                has_prev = False

        prev_x, prev_y = cx, cy

    else:
        if stroke:
            auto_perfect(stroke)
            stroke.clear()
        has_prev = False
        fist_hold_start = None
        palm_hold_start = None

    draw_ui(frame)
    combined = np.hstack((frame, canvas))
    cv2.imshow("Gesture Draw Pro", combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
landmarker.close()