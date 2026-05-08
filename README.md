# IMU Gesture Classifier — Random Forest

The long-term goal of this project is to build a low-cost personal coaching tool for badminton. An IMU worn on the wrist can capture swing motion data for machine learning models, without a camera (expensive) or human observer. If the system can classify different swing gestures reliably, it could eventually provide real-time feedback on technique and consistency. This project is the first step toward that goal: demonstrating that basic hand gestures can be recognized from raw IMU data in real time. Specifically, I investigated whether an MSP430F5529 microcontroller, an MPU-6050 IMU, and a simple machine learning pipeline can reliably distinguish three gestures: idle, swipe left, and swipe right.

---

## How it works

```
MPU-6050 --I2C--> MSP430F5529 --UART/USB--> Python ---> Applications
```

The MSP430 reads 6-axis IMU data (ax, ay, az, gx, gy, gz) at 10 Hz over I2C and streams it as CSV over USB serial at 9600 baud (sending one sample every 0.1 seconds).

**The key insight for ML:** you can't label individual sensor rows — a gesture takes 1–3 seconds, not 0.1 s. So instead of labeling rows, I label *windows* of 20 consecutive rows (2 seconds). From each window I extract 7 statistics per axis (mean, std, min, max, range, abs peak, energy), giving 42 features per window. In real time, the window slides forward one row at a time, so a new prediction fires every 0.1 s.

**Why Random Forest:** it works well with small datasets (~230 training samples here), rarely overfits, and gives a confidence score per prediction — useful for ignoring uncertain results. A neural network would need far more data for a 3-class problem this simple.

The top features turned out to be gyroscope z-axis mean and accelerometer x-axis std. That makes sense — swiping left vs. right produces opposite wrist rotation, which shows up clearly as angular velocity along z.

---

## Results

- **5-fold CV accuracy: 0.948 ± 0.084**
- A clear swipe produces a confident prediction within ~1 second
- Predictions below 60% confidence are suppressed

| Class | Raw Rows | Training Windows |
|---|---|---|
| idle | 1,620 | 81 |
| swipe_left | 1,480 | 74 |
| swipe_right | 1,500 | 75 |
| **Total** | **4,600** | **~230** |

---

## Hardware

MPU-6050 on a breadboard connected to an MSP430F5529 LaunchPad via jumper wires.

| Signal | MPU-6050 | MSP430F5529 |
|---|---|---|
| Power | VCC | 3.3 V |
| Ground | GND | GND |
| I2C Data | SDA | P3.0 (UCB0SDA) |
| I2C Clock | SCL | P3.1 (UCB0SCL) |
| I2C Address | AD0 | GND → address 0x68 |
| UART TX | — | P4.4 (UCA1TXD) |

Clock: ~1.048576 MHz internal DCOCLKDIV. UART: BR0 = 109, UCBRSx = 2 (matched to actual clock, not the nominal 1 MHz).

---

## Project structure

```
msp430_firmware/main.c   — reads IMU over I2C, streams CSV over UART
data_storing.py          — collect and label gesture data into CSV files
inspect_data.py          — scan CSVs for outliers, clean interactively
train.py                 — extract features, train Random Forest, save model.pkl
predict1.py              — live gesture prediction from serial stream
password.py              — gesture-based password app (4-gesture sequence)
game.py                  — gesture-controlled terminal dodge game
```

---

## How to run

**1. Flash firmware**

Open `msp430_firmware/` in Code Composer Studio and flash `main.c` to the MSP430F5529.

**2. Install dependencies**

```bash
pip install pyserial numpy pandas scikit-learn joblib
```

**3. Collect data**

```bash
python data_storing.py
```

Enter a label (`idle`, `swipe_left`, `swipe_right`), press Enter before each rep, perform the gesture. Repeat until you have enough reps per class.

**4. Train**

```bash
python train.py
```

Prints cross-validation accuracy and top feature importances, saves `model.pkl`.

**5. Run live prediction**

```bash
python predict1.py
```

**6. Run the apps**

```bash
python password.py   # gesture password
python game.py       # terminal dodge game
```

> Update the `PORT` variable in each script to match your system (e.g., `/dev/cu.usbmodem1203` on macOS).

---

## Bugs that took the most time

**UART baud mismatch** — no serial output, or garbled data. The baud divider was calculated for exactly 1 MHz, but the actual clock is 1,048,576 Hz. Also, setting only one clock register accidentally switched other clocks to 32 kHz. Fixed by recalculating BR0 = 109 and setting all clock fields explicitly.

**I2C NACK / firmware hang** — MPU-6050 not responding, firmware froze in a loop waiting for an ACK that never came. SDA and SCL were swapped. Fixed the wiring and added NACK detection so the firmware fails gracefully instead of hanging.

**Serial buffer contamination** — data collection was saving stale idle data instead of the fresh gesture. The MSP430 streams continuously, so the laptop's serial buffer fills up while waiting for Enter. Fixed by continuously draining the buffer during the wait, so recording starts clean.

**Duplicate repetition IDs** — restarting a collection session reset rep numbering to zero, causing ID collisions in the CSV. Fixed by reading the existing CSV on startup and continuing numbering from the last value.

---

## Limitations

Only 2 active gesture classes (plus idle). Adding more gestures would need significantly more training data and would increase classification difficulty. The accelerometer is also sensitive to sensor tilt — if the orientation at inference time differs from training, accuracy drops. A gravity-removal step (high-pass filter or calibration) would help.
