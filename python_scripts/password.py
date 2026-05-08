import serial
import numpy as np
import joblib
import os
import sys
import tty
import termios
import threading
import time
from collections import deque

PORT = "/dev/cu.usbmodem1203"
BAUD = 9600
WINDOW = 20
CONFIDENCE_THRESHOLD = 0.6
SAMPLE_INTERVAL = 0.1

COLS_COUNT = 6

SECRET = ["swipe_right", "swipe_left", "swipe_right", "swipe_right"]

def extract_features(window):
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

def print_progress(sequence):
    filled = "".join("R" if g == "swipe_right" else "L" for g in sequence)
    remaining = "_" * (len(SECRET) - len(sequence))
    return f"[{filled}{remaining}]"

# --- Keyboard listener ---
delete_requested = threading.Event()
confirm_requested = threading.Event()

def key_listener():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while True:
            ch = sys.stdin.read(1)
            if ch in ('z', 'Z', '\x7f', '\x08'):
                delete_requested.set()
            elif ch in ('\n', '\r'):
                confirm_requested.set()
    except Exception:
        pass
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

threading.Thread(target=key_listener, daemon=True).start()

# --- Load model ---
folder = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(folder, "model.pkl")

if not os.path.exists(model_path):
    print("model.pkl not found — run train.py first.")
    exit(1)

model = joblib.load(model_path)
print(f"Model loaded. Classes: {model.classes_}")
print(f"Listening on {PORT} at {BAUD} baud. Ctrl+C to stop.")
print(f"Press 'z'/backspace to delete last gesture. Press Enter to confirm after 4 gestures.\n")
print("Enter gesture password:")
print(f"Progress: {print_progress([])}")

buffer = deque(maxlen=WINDOW)

gesture_state = "idle"
sequence = []
idle_count = 0
IDLE_REQUIRED = 10
waiting_for_confirm = False

with serial.Serial(PORT, BAUD, timeout=2) as ser:
    ser.reset_input_buffer()

    while True:

        # --- Handle delete ---
        if delete_requested.is_set():
            delete_requested.clear()
            if sequence:
                removed = sequence.pop()
                gesture_state = "active"
                idle_count = 0
                waiting_for_confirm = False
                print(f"\n[Deleted: {removed}] | Progress: {print_progress(sequence)}")
            else:
                print(f"\n[Nothing to delete]")

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

        if confidence < CONFIDENCE_THRESHOLD:
            prediction = "idle"

        # --- Waiting for Enter to confirm 4th gesture ---
        if waiting_for_confirm:
            if confirm_requested.is_set():
                confirm_requested.clear()
                waiting_for_confirm = False
                unlocked = (sequence == SECRET)
                sequence = []
                gesture_state = "active"
                idle_count = 0
                if unlocked:
                    lines = [
                        "",
                        "  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *",
                        "  *                                            *",
                        "  *        A C C E S S   G R A N T E D         *",
                        "  *          Password Unlocked!                *",
                        "  *                                            *",
                        "  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *",
                        "",
                    ]
                    for l in lines:
                        print(l, flush=True)
                        time.sleep(0.07)
                    break
                else:
                    lines = [
                        "",
                        "  X  X  X  X  X  X  X  X  X  X  X  X  X  X  X",
                        "  X                                            X",
                        "  X          A C C E S S   D E N I E D         X",
                        "  X           Incorrect password.              X",
                        "  X                  Try again.                X",
                        "  X                                            X",
                        "  X  X  X  X  X  X  X  X  X  X  X  X  X  X  X",
                        "",
                    ]
                    for l in lines:
                        print(l, flush=True)
                        time.sleep(0.07)
                    time.sleep(1.0)
                    ser.reset_input_buffer()
                    print(f"\nEnter gesture password:")
                    print(f"Progress: {print_progress([])}")
            else:
                print(f"\r  Current: {prediction:<12} ({confidence:.0%}) | Press Enter to confirm {print_progress(sequence)}   ", end="", flush=True)
            continue

        # --- State machine ---
        if prediction != "idle" and gesture_state == "idle":
            gesture_state = "active"
            idle_count = 0
            sequence.append(prediction)
            print(f"\nGesture {len(sequence)}: {prediction:<20} | Progress: {print_progress(sequence)}")

            if len(sequence) == len(SECRET):
                waiting_for_confirm = True
                print(f"Press Enter to confirm or 'z' to delete last gesture.")

        elif prediction == "idle" and gesture_state == "active":
            idle_count += 1
            secs_remaining = (IDLE_REQUIRED - idle_count) * SAMPLE_INTERVAL
            print(f"\r  Current: {prediction:<12} | Remain idle for {secs_remaining:.1f}s                                          ", end="", flush=True)
            if idle_count >= IDLE_REQUIRED:
                gesture_state = "idle"
                idle_count = 0
                next_gesture = len(sequence) + 1
                print(f"\r  Ready — swipe gesture {next_gesture} of {len(SECRET)}                                                    ", flush=True)

        else:
            # Non-idle during active state: reset idle count
            if gesture_state == "active":
                idle_count = 0

            if gesture_state == "idle":
                next_gesture = len(sequence) + 1
                print(f"\r  Current: {prediction:<12} ({confidence:.0%}) | Waiting for gesture {next_gesture} of {len(SECRET)}   ", end="", flush=True)
            elif gesture_state == "active":
                print(f"\r  Current: {prediction:<12} ({confidence:.0%}) | Remain idle to continue                               ", end="", flush=True)
