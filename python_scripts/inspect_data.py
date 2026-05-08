"""
inspect_data.py  —  detect and clean outlier rows in gesture CSVs

For each rep (20-row window), flags any row where at least one channel
has a Z-score > THRESHOLD (i.e. far from the rep's own mean).

Outlier options:
  k = keep as-is
  d = delete the row  (rep shrinks to 19 rows — train.py will skip it)
  r = replace with the mean of the other 19 rows in the same rep
  a = replace ALL remaining outliers in this file automatically
"""

import pandas as pd
import numpy as np
import glob
import os

COLS        = ["ax", "ay", "az", "gx", "gy", "gz"]
THRESHOLD   = 2.5   # Z-score cutoff within each rep
WINDOW      = 20

folder = os.path.dirname(os.path.abspath(__file__))

for path in sorted(glob.glob(os.path.join(folder, "*.csv"))):
    fname = os.path.basename(path)
    if fname == ".csv":
        continue

    df = pd.read_csv(path, header=None,
                     names=["ax", "ay", "az", "gx", "gy", "gz", "label", "rep"])

    outlier_indices = []

    for (label, rep), group in df.groupby(["label", "rep"]):
        if len(group) != WINDOW:
            continue
        for col in COLS:
            v = group[col].values
            mean, std = v.mean(), v.std()
            if std == 0:
                continue
            z = np.abs((v - mean) / std)
            bad_positions = np.where(z > THRESHOLD)[0]
            for pos in bad_positions:
                idx = group.index[pos]
                if idx not in outlier_indices:
                    outlier_indices.append((idx, label, rep, pos, col, z[pos],
                                            group[col].iloc[pos], mean, std))

    if not outlier_indices:
        print(f"\n{fname}: no outliers found above Z={THRESHOLD}")
        continue

    print(f"\n{'='*60}")
    print(f"FILE: {fname}  —  {len(outlier_indices)} outlier row(s) found")
    print(f"{'='*60}")

    auto_replace = False
    rows_to_drop = []

    seen_rows = set()   # avoid prompting twice for same row
    for (idx, label, rep, pos, col, z_score, val, mean, std) in outlier_indices:
        if idx in seen_rows:
            continue
        seen_rows.add(idx)

        # Show the flagged row in context
        group = df[(df["label"] == label) & (df["rep"] == rep)]
        row_data = df.loc[idx, COLS].tolist()
        group_means = group[COLS].mean().round(1).tolist()

        print(f"\n  Rep {rep}, row {pos+1}/20  (df index {idx})")
        print(f"  Flagged channel : {col}  |  value={val:.0f}  "
              f"mean={mean:.0f}  std={std:.0f}  Z={z_score:.2f}")
        print(f"  Full row        : {[int(x) for x in row_data]}")
        print(f"  Rep means       : {[round(x,1) for x in group_means]}")

        if auto_replace:
            action = "r"
        else:
            action = input("  Action — [k]eep / [d]elete / [r]eplace with rep mean "
                           "/ [a]uto-replace all: ").strip().lower()

        if action == "a":
            auto_replace = True
            action = "r"

        if action == "d":
            rows_to_drop.append(idx)
            print("  → marked for deletion")

        elif action == "r":
            # Replace with mean of the OTHER 19 rows in the same rep
            other = group.drop(index=idx)
            rep_mean = other[COLS].mean().round(0).astype(int)
            df.loc[idx, COLS] = rep_mean.values
            print(f"  → replaced with rep mean: {rep_mean.tolist()}")

        else:
            print("  → kept")

    if rows_to_drop:
        df = df.drop(index=rows_to_drop).reset_index(drop=True)
        print(f"\n  Deleted {len(rows_to_drop)} row(s).")

    # Save cleaned file
    out_path = path  # overwrite original
    df.to_csv(out_path, header=False, index=False)
    print(f"\n  Saved cleaned file → {fname}")

print("\nDone.")
