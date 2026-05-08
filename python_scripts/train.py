import pandas as pd
import numpy as np
import glob
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import joblib

WINDOW = 20
COLS = ["ax", "ay", "az", "gx", "gy", "gz"]

def extract_features(window_df):
    feats = []
    for col in COLS:
        v = window_df[col].values
        feats += [
            v.mean(),                        # average level (captures direction)
            v.std(),                         # spread
            v.min(),                         # most negative peak
            v.max(),                         # most positive peak
            v.max() - v.min(),               # range (total swing)
            np.abs(v).max(),                 # largest absolute peak (direction-independent intensity)
            float(np.sum(v**2)) / len(v),    # energy (intensity)
        ]
    return feats

def feature_names():
    names = []
    for col in COLS:
        for stat in ["mean", "std", "min", "max", "range", "abs_peak", "energy"]:
            names.append(f"{col}_{stat}")
    return names

# ── Load all CSVs in the same folder ──────────────────────────────────────────
folder = os.path.dirname(os.path.abspath(__file__))
dfs = []
for path in glob.glob(os.path.join(folder, "*.csv")):
    fname = os.path.basename(path)
    if fname == ".csv":
        continue
    df = pd.read_csv(path, header=None,
                     names=["ax", "ay", "az", "gx", "gy", "gz", "label", "rep"])
    dfs.append(df)
    print(f"  Loaded {fname}: {len(df)} rows")

data = pd.concat(dfs, ignore_index=True)

# Renumber reps sequentially per label so appended collections never collide
def renumber_reps(df):
    df = df.copy()
    complete = (len(df) // WINDOW) * WINDOW
    df = df.iloc[:complete]
    df["rep"] = [i // WINDOW for i in range(len(df))]
    return df

data = data.groupby("label", group_keys=False).apply(renumber_reps).reset_index(drop=True)
print(f"\nTotal rows: {len(data)}")

# ── Build feature matrix ───────────────────────────────────────────────────────
X, y = [], []
for (label, rep), group in data.groupby(["label", "rep"]):
    if len(group) != WINDOW:
        continue
    X.append(extract_features(group))
    y.append(label)

X = np.array(X)
y = np.array(y)
print(f"Samples: {len(y)}  |  Classes: {np.unique(y)}")

# ── Train ──────────────────────────────────────────────────────────────────────
clf = RandomForestClassifier(n_estimators=100, random_state=42)
scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy")
print(f"\n5-fold CV accuracy: {scores.mean():.3f} ± {scores.std():.3f}")

clf.fit(X, y)

# ── Feature importances (top 10) ───────────────────────────────────────────────
names = feature_names()
importances = sorted(zip(names, clf.feature_importances_), key=lambda x: -x[1])
print("\nTop 10 most important features:")
for name, imp in importances[:10]:
    bar = "█" * int(imp * 200)
    print(f"  {name:<20} {imp:.4f}  {bar}")

model_path = os.path.join(folder, "model.pkl")
joblib.dump(clf, model_path)
print(f"\nModel saved → {model_path}")
