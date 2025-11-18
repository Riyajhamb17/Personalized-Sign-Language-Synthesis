import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import random
import whisper
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# -----------------------------
# Parse Arguments
# -----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--text", type=str, default=None)
parser.add_argument("--audio", type=str, default=None)
args = parser.parse_args()


# -----------------------------
# Load Whisper once
# -----------------------------
whisper_model = whisper.load_model("base")


# -----------------------------
# CONFIG
# -----------------------------
DB_PATH = "gloss_pose_library_testval.pkl"
INTERP_STEPS = 8
MODEL_DIR = "best_model"


# -----------------------------
# Load Text → Gloss Model
# -----------------------------
print("Loading Text→Gloss model from", MODEL_DIR)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)
model.eval()
print("Model and tokenizer loaded successfully.")


# -----------------------------
# Load Gloss → Pose Database
# -----------------------------
with open(DB_PATH, "rb") as f:
    gloss_db = pickle.load(f)

print(f"Loaded database with {len(gloss_db)} glosses.")


# -----------------------------
# BODY EDGES (OpenPose Body25)
# -----------------------------
BODY_EDGES = [
    (0,1), (1,8),
    (1,2), (2,3), (3,4),
    (1,5), (5,6), (6,7),
    (8,9), (9,10), (10,11),
    (8,12), (12,13), (13,14),
    (0,15), (15,17), (0,16), (16,18),
    (11,22), (22,23), (11,24),
    (14,19), (19,20), (14,21)
]


# -----------------------------
# Gloss Prediction
# -----------------------------
def text_to_gloss(text: str):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=64,
            num_beams=4,
            early_stopping=True
        )

    gloss_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    glosses = [g.strip().upper() for g in gloss_text.split()]
    cleaned = [g.replace("-", "").replace("?", "").upper() for g in glosses]

    gloss_seq = [g for g in cleaned if g in gloss_db]
    print(f"Predicted glosses: {glosses}")
    print(f"Valid in DB: {gloss_seq}")
    return gloss_seq


# -----------------------------
# Normalize Keypoints
# -----------------------------
def normalize_pose(pose_xy):
    pose = np.array(pose_xy, dtype=float)
    valid = (pose != 0).any(axis=1)

    if valid.any():
        pts = pose[valid]
        rng = np.maximum(pts.max(axis=0) - pts.min(axis=0), 1e-8)
        pose[valid] = (pts - pts.min(axis=0)) / rng

    return pose, valid


# -----------------------------
# Animate Glosses (frames only)
# -----------------------------
def animate_sequence(gloss_sequence):
    frames = []
    valids = []

    for g in gloss_sequence:
        if g not in gloss_db:
            print(f"⚠️ Gloss '{g}' not found in database.")
            continue

        seq = random.choice(gloss_db[g])  # list of frames
        print(f"🎬 Gloss {g}: {len(seq)} frames")

        for pose in seq:
            pose, valid = normalize_pose(pose)
            frames.append(pose)
            valids.append(valid)

        # Pause for readability
        for _ in 5 * [None]:
            frames.append(pose)
            valids.append(valid)

    print(f"Total frames loaded: {len(frames)}")
    return frames, valids


# -----------------------------
# Transition Interpolation
# -----------------------------
def add_transitions(frames, valids, gloss_seq, interp_steps=8):
    if not frames or not gloss_seq:
        return frames, valids

    smooth_frames, smooth_valids = [], []
    gloss_index = 0
    frame_counter = 0

    for i in range(len(frames) - 1):
        smooth_frames.append(frames[i])
        smooth_valids.append(valids[i])
        frame_counter += 1

        if gloss_index < len(gloss_seq):
            if frame_counter >= len(gloss_db[gloss_seq[gloss_index]][0]) - 1:
                # add interpolated frames
                p1, p2 = frames[i], frames[i + 1]
                for j in range(interp_steps):
                    t = j / interp_steps
                    smooth_frames.append(p1 + (p2 - p1) * t)
                    smooth_valids.append(valids[i])

                gloss_index += 1
                frame_counter = 0

    smooth_frames.append(frames[-1])
    smooth_valids.append(valids[-1])
    return smooth_frames, smooth_valids


# -----------------------------
# Whisper Audio-to-Text
# -----------------------------
def speech_to_text(audio_path):
    """Transcribe speech from an audio file using local Whisper."""
    print(f"Transcribing audio: {audio_path}")

    # Ensure format is readable by Whisper
    try:
        result = whisper_model.transcribe(audio_path)
        text = result["text"].strip()
        print(f"🎙️ Transcribed Speech: {text}")
        return text
    except Exception as e:
        print("Whisper transcription failed:", e)
        return ""



# -----------------------------
# Save MP4 Animation
# -----------------------------
def save_animation(poses, valid_masks, gloss_seq, outpath="sign_output.mp4"):
    fig, ax = plt.subplots()
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)
    ax.set_aspect('equal')
    ax.axis('off')

    skeleton_lines = [ax.plot([], [], lw=2)[0] for _ in BODY_EDGES]
    scatter = ax.scatter([], [], s=12)

    def update(i):
        pose = poses[i]
        valid = valid_masks[i]

        for (edge, line) in zip(BODY_EDGES, skeleton_lines):
            a, b = edge
            if valid[a] and valid[b]:
                line.set_data([pose[a][0], pose[b][0]],
                              [pose[a][1], pose[b][1]])
            else:
                line.set_data([], [])

        scatter.set_offsets(pose[valid])
        return skeleton_lines + [scatter]

    anim = FuncAnimation(fig, update, frames=len(poses), interval=70)
    writer = FFMpegWriter(fps=5)

    anim.save(outpath, writer=writer)
    plt.close(fig)


# -----------------------------
# INPUT HANDLING
# -----------------------------
if args.text:
    sentence = args.text
elif args.audio:
    sentence = speech_to_text(args.audio)
else:
    raise ValueError("No input provided. Use --text or --audio.")

print("Input sentence:", sentence)


# -----------------------------
# MAIN PIPELINE
# -----------------------------
gloss_seq = text_to_gloss(sentence)
poses, valid_masks = animate_sequence(gloss_seq)
poses, valid_masks = add_transitions(poses, valid_masks, gloss_seq, interp_steps=INTERP_STEPS)

print("Saving animation...")
save_animation(poses, valid_masks, gloss_seq, "sign_output.mp4")
print(" Video saved successfully.")
