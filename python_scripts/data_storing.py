import serial
import csv
import os
import time
import sys
import select

PORT = "/dev/cu.usbmodem1203"
BAUD = 9600
ROWS_PER_GESTURE = 20        # ~2 seconds at 10 Hz
TOTAL_REPS = 30

label = input("Enter gesture label (e.g. swipe_right, swipe_left, idle): ").strip()
filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"{label}.csv")

# Find the next rep number by checking the existing file
start_rep = 0
if os.path.exists(filename):
    with open(filename, "r") as existing:
        for line in existing:
            parts = line.strip().split(",")
            if len(parts) == 8:
                try:
                    start_rep = max(start_rep, int(parts[7]) + 1)
                except ValueError:
                    pass
if start_rep > 0:
    print(f"Existing data found — continuing from rep {start_rep}")

with serial.Serial(PORT, BAUD, timeout=2) as ser, \
     open(filename, "a", newline="") as f:
    writer = csv.writer(f)
    rep = start_rep

    while rep < start_rep + TOTAL_REPS:
        print(f"\nRep {rep + 1}/{start_rep + TOTAL_REPS} — press Enter, then perform gesture...", end="", flush=True)

        # While waiting for Enter, continuously drain the serial buffer so it never accumulates
        while True:
            r, _, _ = select.select([sys.stdin], [], [], 0.05)  # check stdin every 50 ms
            if r:
                sys.stdin.readline()  # consume the Enter
                # Drain any extra queued Enter presses
                while select.select([sys.stdin], [], [], 0)[0]:
                    sys.stdin.readline()
                break
            # No Enter yet — drain whatever arrived on serial
            if ser.in_waiting:
                ser.read(ser.in_waiting)

        # One final flush for any bytes in-flight at the moment Enter was pressed
        time.sleep(0.05)
        ser.reset_input_buffer()

        rows = 0
        while rows < ROWS_PER_GESTURE:
            line = ser.readline().decode(errors="ignore").strip()
            if not line or len(line.split(",")) != 6:
                continue
            writer.writerow(line.split(",") + [label, rep])
            f.flush()
            rows += 1

        print(f"\n  Recorded {ROWS_PER_GESTURE} rows. Return to start position.")
        rep += 1

    print(f"\nDone! {TOTAL_REPS} reps saved to {filename}")
