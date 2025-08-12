# clip_ingest.py
import os
import subprocess
import json
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm
import torch
from transformers import CLIPProcessor, CLIPModel


# --- CONFIG ---
INPUT_DIR = "D:\Principal\Vidéo\Banque de donnée Vidéo\Perso"
FRAMES_DIR = "frames_tmp"
EMB_DIR = "embeddings"
FPS = 1                 # frames per second to extract (ajuste)
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "openai/clip-vit-base-patch32"
# ----------------

os.makedirs(FRAMES_DIR, exist_ok=True)
os.makedirs(EMB_DIR, exist_ok=True)

model = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE)
processor = CLIPProcessor.from_pretrained(MODEL_NAME)

def extract_frames(video_path, out_dir, n_frames=5):
    os.makedirs(out_dir, exist_ok=True)
    # Obtenir la durée de la vidéo avec ffprobe
    FFMPEG_PATH = r"D:\Principal\Vidéo\Developpement_outil_automatisation_video\venv\ffmpeg-master-latest-win64-gpl-shared\ffmpeg-master-latest-win64-gpl-shared\bin\ffmpeg.exe"
    FFPROBE_PATH = FFMPEG_PATH.replace("ffmpeg.exe", "ffprobe.exe")
    cmd = [
        FFPROBE_PATH, "-v", "error", "-show_entries",
        "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    try:
        duration = float(result.stdout.strip())
    except Exception:
        duration = None

    if not duration or duration <= 0:
        # fallback: extraire la première image seulement
        timestamps = [0]
    else:
        # répartir n_frames timestamps sur la durée
        timestamps = [duration * i / (n_frames + 1) for i in range(1, n_frames + 1)]

    for idx, ts in enumerate(timestamps):
        out_img = os.path.join(out_dir, f"frame_{idx+1:05d}.jpg")
        cmd = [
            FFMPEG_PATH, "-y", "-ss", str(ts), "-i", str(video_path),
            "-frames:v", "1", out_img
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def load_image(path):
    return Image.open(path).convert("RGB")

def compute_clip_embedding_for_frames(frame_paths):
    # load images in batches, compute image embeddings, then mean-pool
    all_feats = []
    for i in range(0, len(frame_paths), BATCH_SIZE):
        batch_paths = frame_paths[i:i+BATCH_SIZE]
        images = [load_image(p) for p in batch_paths]
        inputs = processor(images=images, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            feats = model.get_image_features(**inputs)  # shape (batch, dim)
            feats = feats.cpu().numpy()
            all_feats.append(feats)
    if not all_feats:
        return None
    all_feats = np.vstack(all_feats)
    # L2 normalize then mean-pool
    norms = np.linalg.norm(all_feats, axis=1, keepdims=True)
    norms[norms==0] = 1.0
    all_feats = all_feats / norms
    clip_emb = all_feats.mean(axis=0)
    # normalize clip embedding
    clip_emb = clip_emb / (np.linalg.norm(clip_emb) + 1e-12)
    return clip_emb

index = {}
for fname in os.listdir(INPUT_DIR):
    if not fname.lower().endswith((".mp4", ".mov", ".mkv", ".avi")):
        continue
    video_path = os.path.join(INPUT_DIR, fname)
    base = Path(fname).stem
    video_frames_dir = os.path.join(FRAMES_DIR, base)
    print(f"Processing {fname} -> extracting frames...")
    # extract_frames(video_path, video_frames_dir, fps=FPS)
    extract_frames(video_path, video_frames_dir, n_frames=5)
    # gather frames
    frames = sorted([os.path.join(video_frames_dir, p) for p in os.listdir(video_frames_dir) if p.endswith(".jpg")])
    if not frames:
        print("  No frames extracted, skipping.")
        continue
    print(f"  {len(frames)} frames; computing CLIP embeddings...")
    emb = compute_clip_embedding_for_frames(frames)
    if emb is None:
        print("  No embedding, skipping.")
        continue
    out_emb_path = os.path.join(EMB_DIR, base + ".npy")
    np.save(out_emb_path, emb)
    index[fname] = {"embedding": out_emb_path, "n_frames": len(frames)}
    # cleanup frames (optionnel) -> commente si tu veux garder les images
    for p in frames:
        os.remove(p)
    os.rmdir(video_frames_dir)

# save index
with open(os.path.join(EMB_DIR, "index.json"), "w", encoding="utf-8") as f:
    json.dump(index, f, ensure_ascii=False, indent=2)

print("Done. Embeddings saved in", EMB_DIR)
