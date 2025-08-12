import os
import datetime
import shutil
import json
import numpy as np
import subprocess
from transformers import CLIPProcessor, CLIPModel
from scipy.spatial.distance import cosine

# --------------------
# CONFIGURATION
# --------------------
EMBEDDINGS_DIR = "embeddings"
VIDEO_DIR = "clips_input"  # Où se trouvent tes vidéos originales
TAGS_FILE = "D:\\Principal\\Vidéo\\Developpement_outil_automatisation_video\\clip-ingest\\tags.txt"  # Liste de tags, un par ligne
TOP_N = 7  # Nombre de tags à retenir
EXIFTOOL_PATH = r"D:\Principal\Vidéo\Developpement_outil_automatisation_video\venv\exiftool-13.33_64\exiftool-13.33_64\exiftool(-k).exe"  # Chemin vers exiftool.exe

# --------------------
# 1. Charger le modèle CLIP
# --------------------
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# --------------------
# 2. Charger la liste de tags
# --------------------
with open(TAGS_FILE, "r", encoding="utf-8") as f:
    tags = [line.strip() for line in f if line.strip()]
print("Tags chargés :", tags)

# Générer embeddings pour chaque tag
text_inputs = processor(text=tags, return_tensors="pt", padding=True)
text_embeddings = model.get_text_features(**text_inputs).detach().numpy()

# --------------------
# 3. Parcourir chaque embedding vidéo
# --------------------
print("Fichiers dans EMBEDDINGS_DIR :", os.listdir(EMBEDDINGS_DIR))
for fname in os.listdir(EMBEDDINGS_DIR):
    print("Fichier trouvé :", fname)
    if fname.endswith(".npy"):
        print("Traitement du fichier :", fname)
        video_name = os.path.splitext(fname)[0]  # enlève juste l'extension .npy
        video_path = os.path.join(VIDEO_DIR, f"{video_name}.MP4")
        video_embedding = np.load(os.path.join(EMBEDDINGS_DIR, fname))

        # --------------------
        # 4. Calculer similarités cosinus
        # --------------------
        similarities = []
        for tag, text_emb in zip(tags, text_embeddings):
            sim = 1 - cosine(video_embedding, text_emb)
            similarities.append((tag, sim))

        # --------------------
        # 5. Sélectionner top-N tags
        # --------------------
        similarities.sort(key=lambda x: x[1], reverse=True)
        best_tags = [tag for tag, _ in similarities[:TOP_N]]

        print(f"Vidéo {video_name} → Tags : {best_tags}")

        # --------------------
        # 6. Renommer la vidéo avec date + tags (sans ExifTool bloquant)
        # --------------------

            # Essayer d'obtenir la date système
        ts = os.path.getctime(video_path)
        creation_date = datetime.datetime.fromtimestamp(ts).strftime("%Y_%m_%d")

        # Nettoyer les tags pour éviter caractères interdits
        safe_tags = [tag.replace(" ", "_") for tag in best_tags]

        # Déterminer l'extension (si absente on laisse vide)
        ext = os.path.splitext(video_path)[1]

        # Construire le nouveau nom
        base_dir = os.path.dirname(video_path)
        new_name = f"{creation_date}-" + "-".join(safe_tags) + ext
        new_path = os.path.join(base_dir, new_name)

        # Éviter d'écraser un fichier existant
        counter = 1
        while os.path.exists(new_path):
            name_no_ext = os.path.splitext(new_name)[0]
            new_name = f"{name_no_ext}_{counter}{ext}"
            new_path = os.path.join(base_dir, new_name)
            counter += 1

        os.rename(video_path, new_path)

        print(f"✅ Fichier renommé : {new_path}")


        # --------------------
        # 7. Supprimer les fichiers temporaires
        # --------------------
        os.remove(os.path.join(EMBEDDINGS_DIR, fname))
        print(f"🗑️ Fichier embedding supprimé : {fname}")

