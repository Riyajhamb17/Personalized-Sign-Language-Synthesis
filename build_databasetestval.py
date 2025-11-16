import os
import json
import pandas as pd
import numpy as np
import pickle
import re
from tqdm import tqdm

# === CONFIG ===
CSV_FILES = [
    "how2sign_realigned_val.csv",
    "how2sign_realigned_test.csv",
    "how2sign_realigned_train.csv"

]
KEYPOINTS_ROOT = "keypoints"
OUTPUT_FILE = "gloss_pose_library_testval.pkl"

# === Helper: Normalize Pose ===
def normalize_pose(pose):
    """Normalize coordinates between 0–1"""
    pose = np.array(pose)
    pose[:, 0] = (pose[:, 0] - np.min(pose[:, 0])) / (np.ptp(pose[:, 0]) + 1e-8)
    pose[:, 1] = (pose[:, 1] - np.min(pose[:, 1])) / (np.ptp(pose[:, 1]) + 1e-8)
    return pose

# === Helper: Normalize Names (for matching) ===
def normalize_name(name):
    """Normalize names: remove dashes, underscores, lowercase"""
    return re.sub(r'[^a-z0-9]', '', str(name).lower().strip('_'))

# === Load CSV metadata ===
print("📥 Loading How2Sign metadata...")
metadata_list = []
for f in CSV_FILES:
    df = pd.read_csv(f, sep="\t")
    df.columns = df.columns.str.upper()
    metadata_list.append(df)

metadata = pd.concat(metadata_list, ignore_index=True)
print(f"✅ Loaded {len(metadata)} rows from CSVs")

# === Create a mapping from normalized VIDEO_NAME → row ===
metadata_map = {
    normalize_name(row["VIDEO_NAME"]): row for _, row in metadata.iterrows()
}

# === Build database ===
gloss_db = {}
print("⚙️ Building gloss→pose database...")
print("Example CSV name:", metadata['VIDEO_NAME'].iloc[0])
print("Example folder name:", os.listdir(KEYPOINTS_ROOT)[:5])


# missed = []
# matched = 0
# for folder in os.listdir(KEYPOINTS_ROOT):
#     key = normalize_name(folder)
#     if key in metadata_map:
#         matched += 1
#     else:
#         missed.append(folder)

# print(f"Matched: {matched}, Missed: {len(missed)}")


for folder in tqdm(os.listdir(KEYPOINTS_ROOT)):
    folder_path = os.path.join(KEYPOINTS_ROOT, folder)
    if not os.path.isdir(folder_path):
        continue

    folder_key = normalize_name(folder).lstrip('_')

    # --- Matching logic ---
    if folder_key in metadata_map:
        row = metadata_map[folder_key]
    else:
        # Fuzzy match for minor name differences (_0-1-, -1-, etc.)
        possible = []
        for k, r in metadata_map.items():
            if k in folder_key or folder_key in k:
                possible.append(r)
            if k.replace("01", "1") in folder_key or folder_key.replace("01", "1") in k:
                possible.append(r)
        if not possible:
            continue
        row = possible[0]

    # --- Extract glosses ---
    glosses = str(row["SENTENCE"]).strip().upper().split()

    # --- Pick representative frame ---
    json_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".json")])
    if not json_files:
        continue

    mid_file = json_files[len(json_files)//2]
    json_path = os.path.join(folder_path, mid_file)

    try:
        with open(json_path, "r") as f:
            data = json.load(f)
        if not data["people"]:
            continue
        keypoints = np.array(data["people"][0]["pose_keypoints_2d"]).reshape(-1, 3)[:, :2]
        keypoints = normalize_pose(keypoints)
    except Exception:
        continue

    for g in glosses:
        gloss_db.setdefault(g, []).append(keypoints)

print(f"✅ Created database with {len(gloss_db)} unique glosses")

# === Save the database ===
with open(OUTPUT_FILE, "wb") as f:
    pickle.dump(gloss_db, f)

print(f"💾 Saved database to {OUTPUT_FILE}")
