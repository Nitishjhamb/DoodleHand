# DoodleHand ✋🎨

**Draw freely in the air using only your hand gestures.**

A real-time virtual drawing application powered by your webcam.  
Use natural hand movements to sketch, erase like a real duster, switch colors, clear the canvas, and save your artwork — no mouse or touchscreen required.

## Features

- **Index finger drawing** — Extend only your index finger to draw smooth lines
- **Duster-style eraser** — Open palm gesture activates a circular erase area (rubbing motion)
- **Color switching** — Hover your index finger over on-screen color palette
- **Automatic shape perfection** — Rough strokes are intelligently converted to triangles, rectangles, or circles
- **Gesture controls**:
  - Hold fist in left panel → clear canvas
  - Hold open palm in left panel → save drawing as PNG
- **Visual feedback** — Subtle hand landmarks + prominent circle on active finger
- **Smooth & natural** — Position smoothing + real-time processing
- **Mouse fallback** — Click colors / CLR / SAVE on left panel if desired

## Demo Controls

| Action         | Gesture                           | Location / Condition            |
| -------------- | --------------------------------- | ------------------------------- |
| Draw           | Index finger only extended        | Outside left panel              |
| Erase (duster) | Open palm (all 5 fingers spread)  | Circular area follows index tip |
| Change color   | Hover index tip over color circle | Left panel                      |
| Clear canvas   | Hold fist ~1 second               | In left panel                   |
| Save drawing   | Hold open palm ~1 second          | In left panel                   |
| Quit           | Press `q`                         | —                               |

## Requirements

- Windows / macOS / Linux
- Webcam (built-in or external)
- Good lighting & right-hand palm facing camera recommended
- Python 3.8–3.11

## Installation & Run (Source Code)

1. Clone the repository

```bash
git clone https://github.com/yourusername/DoodleHand.git
cd DoodleHand

```

2. Install dependencies using the requirements file

```bash
pip install -r requirements.txt

```

3. Launch the app

```bash

python gesture_draw.py

```
