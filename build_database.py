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
# === Build database ===
gloss_db = {}
print("⚙️ Building gloss→pose database...")

import re

def extract_video_key(name: str) -> str:
    """
    Extracts canonical video key ignoring frame indices.
    Handles patterns like '_0-1-', '_12-2-', etc.
    """
    name = name.lower().strip()
    name = name.lstrip('-_')

    # Remove patterns like '_12-1-', '_0-2-', '-3-1-' (any number-number block before rgb)
    name = re.sub(r'[_-]\d+-\d+-', '-', name)

    # Remove single index forms like '_0-' or '-10-'
    name = re.sub(r'[_-]\d+-', '-', name)

    # Remove trailing camera or underscores
    name = re.sub(r'[^a-z0-9-]', '', name)
    return name

# === Before building ===
sample_csv = metadata['VIDEO_NAME'].iloc[0]
sample_folder = os.listdir(KEYPOINTS_ROOT)[0]
print("🔍 Sample comparison:")
print("CSV:", sample_csv, "→", extract_video_key(sample_csv))
print("Folder:", sample_folder, "→", extract_video_key(sample_folder))


print("⚙️ Building gloss→pose database...")

metadata_map_clean = {
    extract_video_key(row["VIDEO_NAME"]): row for _, row in metadata.iterrows()
}



from tqdm import tqdm

missed = []
matched = 0

for folder in tqdm(os.listdir(KEYPOINTS_ROOT), desc="Building gloss→pose database"):
    folder_path = os.path.join(KEYPOINTS_ROOT, folder)
    if not os.path.isdir(folder_path):
        continue

    folder_key = extract_video_key(folder)

    # --- Matching logic ---
    if folder_key in metadata_map_clean:
        row = metadata_map_clean[folder_key]
        matched += 1
    else:
        # Try fallback fuzzy match
        found = False
        for k, r in metadata_map_clean.items():
            if folder_key.startswith(k) or k.startswith(folder_key):
                row = r
                found = True
                matched += 1
                break
        if not found:
            missed.append(folder_key)
            continue

    # --- Extract glosses ---
    glosses = str(row["SENTENCE"]).strip().upper().split()

    # --- Collect frames ---
    json_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".json")])
    if not json_files:
        continue

    frames = []
    for jf in json_files[::5]:  # sample every 5th frame instead of 3rd for speed
        json_path = os.path.join(folder_path, jf)
        try:
            with open(json_path, "r") as f:
                data = json.load(f)
            if not data["people"]:
                continue

            pose = np.array(data["people"][0]["pose_keypoints_2d"]).reshape(-1, 3)[:, :2]
            pose = normalize_pose(pose)
            frames.append(pose)
        except Exception:
            continue

    # --- Skip low-frame folders ---
    if len(frames) < 3:
        continue

    # --- Assign frames to each gloss ---
    for g in glosses:
        gloss_db.setdefault(g, []).append(frames)




print(f"✅ Matched folders: {matched}, Missed folders: {len(missed)}")
print(f"✅ Created database with {len(gloss_db)} unique glosses")


# === Save the database ===
with open(OUTPUT_FILE, "wb") as f:
    pickle.dump(gloss_db, f)

print(f"💾 Saved database to {OUTPUT_FILE}")
