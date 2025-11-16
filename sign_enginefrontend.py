# sign_engine.py
import pickle
import torch
import whisper
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

DB_PATH = "gloss_pose_library_testval.pkl"
MODEL_DIR = "best_model"

# Load once
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR).eval()
gloss_db = pickle.load(open(DB_PATH, "rb"))
whisper_model = whisper.load_model("base")

print("SignAI Engine Loaded!")
