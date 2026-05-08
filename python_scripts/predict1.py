import serial
import numpy as np
import joblib
import os
from collections import deque

PORT = "/dev/cu.usbmodem1203"
BAUD = 9600
WINDOW = 20
CONFIDENCE_THRESHOLD = 0.6   # suppress output below this confidence

COLS_COUNT = 6

def extract_features(window):
    # window: list of 20 rows, each row is [ax, ay, az, gx, gy, gz]
    arr = np.array(window)
    feats = []
    for i in range(COLS_COUNT):
        v = arr[:, i]
        feats += [
            v.mean(),
            v.std(),
            v.min(),
            v.max(),
            v.max() - v.min(),
            float(np.abs(v).max()),
            float(np.sum(v**2)) / len(v),
        ]
    return feats

folder = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(folder, "model.pkl")

if not os.path.exists(model_path):
    print("model.pkl not found — run train.py first.")
    exit(1)

model = joblib.load(model_path)
print(f"Model loaded. Classes: {model.classes_}")
print(f"Listening on {PORT} at {BAUD} baud. Ctrl+C to stop.\n")

buffer = deque(maxlen=WINDOW)

with serial.Serial(PORT, BAUD, timeout=2) as ser:
    ser.reset_input_buffer()

    while True:
        line = ser.readline().decode(errors="ignore").strip()
        parts = line.split(",")
        if len(parts) != 6:
            continue
        try:
            row = list(map(int, parts))
        except ValueError:
            continue

        buffer.append(row)

        if len(buffer) < WINDOW:
            continue

        feats = extract_features(list(buffer))
        probs = model.predict_proba([feats])[0]
        confidence = probs.max()
        prediction = model.classes_[probs.argmax()]

        if confidence >= CONFIDENCE_THRESHOLD:
            print(f"\r{prediction:<20} {confidence:.0%}    ", end="", flush=True)
        else:
            print(f"\r{'?':<20} {confidence:.0%}    ", end="", flush=True)
