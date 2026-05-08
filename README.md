# IMU Gesture Classifier — Random Forest

Real-time hand gesture recognition using an **MSP430F5529 microcontroller**, **MPU-6050 IMU**, and a **Random Forest classifier**. Physical swipe gestures are recognized in real time and used to drive interactive applications: a gesture password system and a gesture-controlled terminal dodge game.

> Motivation: first step toward a low-cost wrist-worn IMU coaching tool for badminton — classifying swing gestures without a camera or human observer.

---

## System Pipeline

```
MPU-6050 --I2C--> MSP430F5529 --UART/USB--> Python ML Pipeline --> Applications
  (sensor)           (firmware)               (laptop)               (game / password)
```

1. MSP430 firmware reads 6-axis IMU data (ax, ay, az, gx, gy, gz) at 10 Hz over I2C
2. Data is streamed to the laptop as CSV over USB serial at 9600 baud
3. Python collects labeled gesture data, extracts features, and trains a Random Forest model
4. Real-time prediction drives a gesture password system and a terminal dodge game

---

## Hardware

| Component | Details |
|---|---|
| Microcontroller | MSP430F5529 LaunchPad |
| IMU Sensor | MPU-6050 (accelerometer + gyroscope) |
| Communication | I2C (sensor to MCU), UART over USB (MCU to laptop) |
| Clock | ~1.048576 MHz (internal DCOCLKDIV) |
| Baud Rate | 9600 (BR0 = 109, UCBRSx = 2) |

### Wiring

| Signal | MPU-6050 Pin | MSP430F5529 Pin |
|---|---|---|
| Power | VCC | 3.3 V |
| Ground | GND | GND |
| I2C Data | SDA | P3.0 (UCB0SDA) |
| I2C Clock | SCL | P3.1 (UCB0SCL) |
| I2C Address | AD0 | GND (address = 0x68) |
| UART TX | — | P4.4 (UCA1TXD) |
| UART RX | — | P4.5 (UCA1RXD) |

---

## Machine Learning

### Feature Extraction
Each gesture window = 20 consecutive sensor rows (2 seconds at 10 Hz).
For each of the 6 sensor axes, 7 statistical features are computed:

`mean, std, min, max, range, abs_peak, energy`

This gives **42 features per window**, each window labeled with one gesture class.
In real-time mode, the window slides by 1 sample, producing a new prediction every 0.1 seconds.

### Classifier
- **Model:** Random Forest (100 decision trees)
- **Classes:** `idle`, `swipe_left`, `swipe_right`
- **Evaluation:** 5-fold cross-validation
- **Accuracy:** 0.948 +/- 0.084
- **Top features:** gyroscope z-axis mean, accelerometer x-axis std — consistent with wrist rotation producing distinctive angular velocity during swipes

### Dataset

| Class | Raw Rows | Training Windows |
|---|---|---|
| idle | 1,620 | 81 |
| swipe_left | 1,480 | 74 |
| swipe_right | 1,500 | 75 |
| **Total** | **4,600** | **~230** |

---

## Project Structure

```
.
├── 319 Motion Sensor Collection/
│   └── main.c               # MSP430 firmware — reads IMU, streams data over UART
├── data_storing.py           # Collect and label gesture data into CSV files
├── train.py                  # Extract features, train Random Forest, save model.pkl
├── predict1.py               # Live gesture prediction from serial stream
├── password.py               # Gesture-based password application
├── game.py                   # Gesture-controlled terminal dodge game
├── inspect_data.py           # Dataset inspection and outlier cleaning tool
├── idle.csv                  # Collected training data — idle
├── swipe_left.csv            # Collected training data — swipe left
├── swipe_right.csv           # Collected training data — swipe right
└── model.pkl                 # Trained Random Forest model
```

---

## How to Run

### 1. Flash the Firmware
Open `319 Motion Sensor Collection/` in **Code Composer Studio** and flash `main.c` to the MSP430F5529. The board will immediately start streaming sensor data over USB.

### 2. Install Python Dependencies
```bash
pip install pyserial numpy pandas scikit-learn joblib
```

### 3. Collect Gesture Data
```bash
python data_storing.py
```
Enter a label (`swipe_left`, `swipe_right`, or `idle`) and follow the prompts. Repeat for each gesture class. More repetitions = better model.

### 4. Train the Model
```bash
python train.py
```
Outputs `model.pkl` and prints cross-validation accuracy and top feature importances.

### 5. Test Live Prediction
```bash
python predict1.py
```
Prints the predicted gesture and confidence score in real time as you move the sensor.

### 6. Play the Dodge Game
```bash
python game.py
```
Swipe left or right physically to move the player and dodge falling obstacles.

> **Note:** Update the `PORT` variable in each Python script to match your system (e.g., `/dev/cu.usbmodem1203` on macOS).

---

## Key Bugs Resolved

| Bug | Symptom | Fix |
|---|---|---|
| UART baud mismatch | Garbled or no serial output | Recalculated divider for actual 1,048,576 Hz clock (BR0=109); rewrote clock init to set all fields explicitly |
| I2C NACK / firmware hang | MPU-6050 not responding, firmware froze | Swapped SDA/SCL wires; added NACK detection to return error instead of looping forever |
| Serial buffer contamination | Old idle data saved instead of fresh gesture | Continuously drain serial buffer while waiting for Enter keypress |
| Duplicate repetition IDs | Training windows collided across sessions | Script now reads existing CSV and continues rep numbering from last value |

---

## Dependencies

| Library | Purpose |
|---|---|
| `pyserial` | Serial port communication |
| `numpy` | Feature computation |
| `pandas` | CSV loading and manipulation |
| `scikit-learn` | RandomForestClassifier, cross_val_score |
| `joblib` | Save/load model.pkl |
| `curses` (stdlib) | Terminal game rendering |
| `threading` (stdlib) | Background gesture thread in game and password |

---

## References

1. InvenSense. *MPU-6000 and MPU-6050 Product Specification*, Rev. 3.4. TDK InvenSense, 2013.
2. Texas Instruments. *MSP430F5529 LaunchPad User's Guide*. Texas Instruments, 2013.
3. Texas Instruments. *MSP430x5xx and MSP430x6xx Family User's Guide (SLAU208)*. ti.com.
4. Scikit-learn Developers. *Random Forests — scikit-learn documentation*. scikit-learn.org.
5. Breiman, L. Random Forests. *Machine Learning*, vol. 45, no. 1, pp. 5–32, 2001.
