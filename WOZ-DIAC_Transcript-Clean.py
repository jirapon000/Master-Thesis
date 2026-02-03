#Import Library
from pathlib import Path
import pandas as pd
import numpy as np
import csv
import json, re, csv
from textblob import TextBlob
import re
from pathlib import Path
import sys

# ====== CONFIG ======
PARTICIPANT_ID = 310  # <--- Change this ID
TRANSCRIPT_CSV = f"Dataset/Transcript (300-340)/{PARTICIPANT_ID}_TRANSCRIPT.csv"
OUTPUT_FOLDER = Path("Clean_Dataset/transcripts")
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
OUT_TURNS = OUTPUT_FOLDER / f"{PARTICIPANT_ID}_clean_turns.csv"

# ====== 1. Read transcript ======
print(f"Processing: {TRANSCRIPT_CSV}...")
try:
    df = pd.read_csv(TRANSCRIPT_CSV, sep=None, engine="python")
except Exception as e:
    print(f"Error reading CSV: {e}")
    sys.exit(1)

# Normalize columns
df.columns = [c.strip().lower() for c in df.columns]

# Ensure 'value' or 'text' exists
text_col = "value" if "value" in df.columns else "text"
speaker_col = "speaker" if "speaker" in df.columns else "participant_id"

if text_col not in df.columns:
    raise ValueError(f"Could not find text column. Found: {df.columns}")

# ====== 2. Robust Speaker Mapping (MODIFIED) ======
def standardize_speaker(s):
    s = str(s).strip().lower()
    
    # 1. Identify Agent/Ellie
    if s in ["ellie", "agent", "interviewer"]:
        return "AGENT"
    
    # 2. Identify Participant (Map to "PARTICIPANT")
    # Checks if the ID (305) is in the string, or if it says 'participant'/'user'
    if str(PARTICIPANT_ID) in s or s in ["participant", "user", "subject", "client", "patient"]:
        return "PARTICIPANT"  # <--- CHANGED FROM "USER"
        
    return "OTHER"

df["speaker_norm"] = df[speaker_col].apply(standardize_speaker)
df = df[df["speaker_norm"] != "OTHER"]

# ====== 3. Clean Text ======
def clean_utt(s):
    if not isinstance(s, str): s = str(s)
    s = re.sub(r"\bl_[a-z]+\b", "", s, flags=re.I) # Remove l_losangeles etc
    s = re.sub(r"\[.*?\]|\(.*?\)|<.*?>", "", s)      # Remove brackets
    s = re.sub(r"\s+", " ", s).strip()
    return s

df["utterance"] = df[text_col].apply(clean_utt)
df = df[df["utterance"].str.len() > 1]

# ====== 4. Standardize Output Columns ======
df["participant_id"] = PARTICIPANT_ID
df["turn_id"] = range(1, len(df) + 1)
df["speaker"] = df["speaker_norm"] # This will now be "PARTICIPANT" or "AGENT"

final_df = df[["participant_id", "turn_id", "speaker", "utterance"]]

# ====== 5. Save ======
final_df.to_csv(OUT_TURNS, index=False)
print(f"✅ Saved clean transcript to: {OUT_TURNS}")
print(f"   -> Found {len(final_df[final_df['speaker']=='PARTICIPANT'])} PARTICIPANT turns.")
