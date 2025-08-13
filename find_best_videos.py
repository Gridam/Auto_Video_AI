import os
import numpy as np
import json
import torch
from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics.pairwise import cosine_similarity

# --- Charger modèle CLIP ---
model_name = "openai/clip-vit-base-patch32"
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained(model_name).to(device)
processor = CLIPProcessor.from_pretrained(model_name)

EMBEDDINGS_DIR = "embeddings"
PHRASES_FILE = "phrases.json"

def time_str_to_seconds(t):
    h, m, s = t.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)

def find_best_videos_for_text(text, top_k=3):
    # Encoder le texte
    inputs = processor(text=[text], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_emb = model.get_text_features(**inputs)
    text_emb = text_emb.cpu().numpy()  # shape (1, 512)

    results = []
    for fname in os.listdir(EMBEDDINGS_DIR):
        if fname.endswith(".npy"):
            npy_path = os.path.abspath(os.path.join(EMBEDDINGS_DIR, fname))
            video_emb = np.load(npy_path)
            sim = cosine_similarity(text_emb, video_emb.reshape(1, -1))[0][0]
            results.append({
                "npy_path": npy_path,
                "score": float(sim)
            })

    # Trier par score décroissant
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]

# --- Traitement de toutes les phrases ---
if __name__ == "__main__":
    # Charger toutes les phrases
    with open(PHRASES_FILE, "r", encoding="utf-8") as f:
        phrases = json.load(f)

    # Calculer l'embedding moyen de toutes les phrases
    all_embeddings = []
    for p in phrases:
        # Tronquer chaque phrase à 77 tokens si besoin
        text = p["text"]
        tokens = processor.tokenizer.tokenize(text)
        if len(tokens) > 77:
            tokens = tokens[:77]
            text = processor.tokenizer.convert_tokens_to_string(tokens)
        inputs = processor(text=[text], return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            emb = model.get_text_features(**inputs)
        all_embeddings.append(emb.cpu().numpy())

    # Moyenne des embeddings (shape: (1, 512))
    global_text_emb = np.mean(np.vstack(all_embeddings), axis=0, keepdims=True)

    used_clips = set()
    last_video_emb = None

    output = []
    for idx, phrase in enumerate(phrases):
        start_sec = time_str_to_seconds(phrase["start"])
        end_sec = time_str_to_seconds(phrase["end"])
        duration = end_sec - start_sec
        n_videos = max(1, int(duration // 2) + 1)

        candidates = find_best_videos_for_text(phrase["text"], top_k=20)  # On prend plus de candidats pour avoir du choix

        alpha = 1.0
        beta = 1.0
        gamma = 2.0

        selected = []
        for _ in range(n_videos):
            filtered = [c for c in candidates if c["npy_path"] not in used_clips]
            if not filtered:
                filtered = candidates

            best_score = -float("inf")
            best_cand = None
            for cand in filtered:
                video_emb = np.load(cand["npy_path"])
                diversity = 0
                if last_video_emb is not None:
                    diversity = 1 - cosine_similarity(video_emb.reshape(1, -1), last_video_emb.reshape(1, -1))[0][0]
                global_score = cosine_similarity(video_emb.reshape(1, -1), global_text_emb)[0][0]
                score = alpha * cand["score"] + beta * diversity + gamma * global_score
                if score > best_score:
                    best_score = score
                    best_cand = cand
            chosen = best_cand

            selected.append(chosen)
            used_clips.add(chosen["npy_path"])
            last_video_emb = np.load(chosen["npy_path"])

        output.append({
            "start": phrase["start"],
            "end": phrase["end"],
            "text": phrase["text"],
            "best_videos": selected
        })

    with open("phrases_with_best_videos.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"[OK] Fichier phrases_with_best_videos.json généré")
