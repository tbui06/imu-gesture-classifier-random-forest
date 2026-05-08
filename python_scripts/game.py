import serial
import numpy as np
import joblib
import os
import sys
import threading
import time
import curses
import random
from collections import deque

PORT = "/dev/cu.usbmodem1203"
BAUD = 9600
WINDOW = 20
CONFIDENCE_THRESHOLD = 0.6
COLS_COUNT = 6

GAME_WIDTH  = 40   # inner playfield width (columns)
GAME_HEIGHT = 20   # inner playfield height (rows)
PLAYER_ROW  = GAME_HEIGHT - 1
PLAYER_CHAR = "^"
OBSTACLE    = "*"

GESTURE_COOLDOWN = 0.4  # seconds before next swipe can fire (auto-reset, no idle needed)
MOVE_CELLS      = 3    # how many columns one swipe moves the player
OBSTACLE_RATE   = 1.2  # seconds between new obstacle spawns (decreases over time)
MIN_RATE        = 0.5  # fastest obstacle spawn rate

# ── Feature extraction (same as predict.py) ──────────────────────────────────
def extract_features(window):
    arr = np.array(window)
    feats = []
    for i in range(COLS_COUNT):
        v = arr[:, i]
        feats += [
            v.mean(), v.std(), v.min(), v.max(),
            v.max() - v.min(),
            float(np.abs(v).max()),
            float(np.sum(v**2)) / len(v),
        ]
    return feats

# ── Shared gesture state ──────────────────────────────────────────────────────
gesture_event = threading.Event()   # set when a new gesture fires
gesture_value = {"v": "idle"}       # latest fired gesture
current_prediction = {"v": "idle", "conf": 0.0}  # live prediction every window

def gesture_thread(model):
    buf = deque(maxlen=WINDOW)
    state = "idle"
    last_fire = 0.0

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

            buf.append(row)
            if len(buf) < WINDOW:
                continue

            feats = extract_features(list(buf))
            probs = model.predict_proba([feats])[0]
            confidence = probs.max()
            prediction = model.classes_[probs.argmax()]

            if confidence < CONFIDENCE_THRESHOLD:
                prediction = "idle"

            current_prediction["v"]    = prediction
            current_prediction["conf"] = confidence

            now = time.time()

            # Auto-reset to idle after cooldown (no need to physically go idle)
            if state == "active" and now - last_fire >= GESTURE_COOLDOWN:
                state = "idle"

            # Fire on idle→gesture edge
            if prediction != "idle" and state == "idle":
                state = "active"
                last_fire = now
                gesture_value["v"] = prediction
                gesture_event.set()

# ── Game ──────────────────────────────────────────────────────────────────────
def run_game(stdscr, model):
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.timeout(50)

    # Colors
    curses.start_color()
    curses.init_pair(1, curses.COLOR_GREEN,  curses.COLOR_BLACK)  # player
    curses.init_pair(2, curses.COLOR_RED,    curses.COLOR_BLACK)  # obstacle
    curses.init_pair(3, curses.COLOR_YELLOW, curses.COLOR_BLACK)  # border / score
    curses.init_pair(4, curses.COLOR_CYAN,   curses.COLOR_BLACK)  # title

    player_col = GAME_WIDTH // 2
    obstacles  = []   # list of [row, col]
    score      = 0
    alive      = True
    last_obstacle = time.time()
    start_time    = time.time()

    def safe_addch(r, c, ch, attr=0):
        max_rows, max_cols = stdscr.getmaxyx()
        if 0 <= r < max_rows and 0 <= c < max_cols - 1:
            try:
                stdscr.addch(r, c, ch, attr)
            except curses.error:
                pass

    def safe_addstr(r, c, s, attr=0):
        max_rows, max_cols = stdscr.getmaxyx()
        if 0 <= r < max_rows and 0 <= c < max_cols - 1:
            try:
                stdscr.addstr(r, c, s[:max_cols - c - 1], attr)
            except curses.error:
                pass

    def draw_border(win_row, win_col):
        for c in range(GAME_WIDTH + 2):
            safe_addch(win_row,                   win_col + c, ord("-"), curses.color_pair(3))
            safe_addch(win_row + GAME_HEIGHT + 1, win_col + c, ord("-"), curses.color_pair(3))
        for r in range(GAME_HEIGHT + 2):
            safe_addch(win_row + r, win_col,                  ord("|"), curses.color_pair(3))
            safe_addch(win_row + r, win_col + GAME_WIDTH + 1, ord("|"), curses.color_pair(3))

    def draw_frame(win_row, win_col):
        stdscr.erase()
        elapsed = time.time() - start_time
        rate = max(MIN_RATE, OBSTACLE_RATE - elapsed * 0.01)

        safe_addstr(win_row - 2, win_col,
                    "  GESTURE DODGE -- swipe left/right to move  ",
                    curses.color_pair(4) | curses.A_BOLD)
        safe_addstr(win_row - 1, win_col,
                    f"  Score: {score}   Speed: {rate:.2f}s/obstacle",
                    curses.color_pair(3))

        pred  = current_prediction["v"]
        conf  = current_prediction["conf"]
        arrow = " >> " if pred == "swipe_right" else " << " if pred == "swipe_left" else "    "
        safe_addstr(win_row + GAME_HEIGHT + 2, win_col,
                    f"  Gesture: {pred:<12} {conf:4.0%} {arrow}   ",
                    curses.color_pair(4))

        draw_border(win_row, win_col)

        for obs in obstacles:
            r, c = obs
            if 0 <= r < GAME_HEIGHT and 0 <= c < GAME_WIDTH:
                safe_addch(win_row + 1 + r, win_col + 1 + c,
                           ord(OBSTACLE), curses.color_pair(2) | curses.A_BOLD)

        safe_addch(win_row + 1 + PLAYER_ROW, win_col + 1 + player_col,
                   ord(PLAYER_CHAR), curses.color_pair(1) | curses.A_BOLD)

        safe_addstr(win_row + GAME_HEIGHT + 3, win_col,
                    "  Press 'q' to quit", curses.color_pair(3))
        stdscr.refresh()

    # Centre the game window — shrink GAME_HEIGHT if terminal is too small
    max_rows, max_cols = stdscr.getmaxyx()
    usable_height = max_rows - 5   # 2 header rows + border top/bottom + footer
    game_h = min(GAME_HEIGHT, usable_height)
    globals()["GAME_HEIGHT"] = game_h
    globals()["PLAYER_ROW"]  = game_h - 1
    win_row = max(2, (max_rows - game_h - 4) // 2)
    win_col = max(0, (max_cols - GAME_WIDTH - 2) // 2)

    while alive:
        now = time.time()
        elapsed = now - start_time
        rate = max(MIN_RATE, OBSTACLE_RATE - elapsed * 0.01)

        # Keyboard quit
        key = stdscr.getch()
        if key == ord('q'):
            break

        # Gesture input
        if gesture_event.is_set():
            gesture_event.clear()
            g = gesture_value["v"]
            if g == "swipe_right":
                player_col = min(GAME_WIDTH - 1, player_col + MOVE_CELLS)
            elif g == "swipe_left":
                player_col = max(0, player_col - MOVE_CELLS)

        # Spawn obstacle
        if now - last_obstacle >= rate:
            obstacles.append([0, random.randint(0, GAME_WIDTH - 1)])
            last_obstacle = now

        # Move obstacles down
        obstacles = [[r + 1, c] for r, c in obstacles]

        # Remove off-screen
        obstacles = [[r, c] for r, c in obstacles if r < GAME_HEIGHT]

        # Collision detection
        for r, c in obstacles:
            if r == PLAYER_ROW and c == player_col:
                alive = False
                break

        # Score: 1 point per obstacle that passes the player row
        score += sum(1 for r, c in obstacles if r == PLAYER_ROW and c != player_col)

        draw_frame(win_row, win_col)
        time.sleep(0.12)

    # Game over screen — r to restart, q to quit
    while True:
        stdscr.erase()
        elapsed = time.time() - start_time
        go_lines = [
            "",
            "  + + + + + + + + + + + + + + + + + + +",
            "  +                                    +",
            "  +           G A M E  O V E R         +",
            f"  +         Final Score : {score:<6}       +",
            f"  +         Time alive  : {elapsed:.1f}s        +",
            "  +                                    +",
            "  + + + + + + + + + + + + + + + + + + +",
            "",
            "      'r' to play again   'q' to quit",
        ]
        for i, line in enumerate(go_lines):
            try:
                stdscr.addstr(win_row + i, win_col, line, curses.color_pair(2) | curses.A_BOLD)
            except curses.error:
                pass
        stdscr.refresh()
        stdscr.nodelay(False)
        key = stdscr.getch()
        if key == ord('q'):
            return
        if key == ord('r'):
            run_game(stdscr, model)
            return

# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    folder = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(folder, "model.pkl")
    if not os.path.exists(model_path):
        print("model.pkl not found — run train.py first.")
        sys.exit(1)

    model = joblib.load(model_path)
    print(f"Model loaded. Classes: {model.classes_}")
    print(f"Connecting to {PORT} at {BAUD} baud...")

    t = threading.Thread(target=gesture_thread, args=(model,), daemon=True)
    t.start()

    time.sleep(1.0)   # let serial stabilise
    print("Starting game...")
    time.sleep(0.5)

    curses.wrapper(run_game, model)
    print("Thanks for playing!")
