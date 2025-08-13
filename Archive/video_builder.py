import os
import re
import json

# --------------------------
# CONFIG
# --------------------------
PHRASES_FILE = "phrases.json"  # Format : [{"start": "00:00:01", "end": "00:00:03", "text": "..."}, ...]
VIDEOS_DIR = "clips_input"     # Répertoire contenant les vidéos renommées avec les mots clés
OUTPUT_FILE = "video_build.json"

# --------------------------
# UTILS
# --------------------------
def clean_word(word):
    return re.sub(r"[^a-z0-9àâçéèêëîïôûùüÿñæœ]", "", word.lower())

def tokenize(text):
    return [clean_word(w) for w in text.split() if clean_word(w)]

def count_common_words(phrase_words, filename_words):
    return len(set(phrase_words) & set(filename_words))

# --------------------------
# 1 - Charger les phrases
# --------------------------
with open(PHRASES_FILE, "r", encoding="utf-8") as f:
    phrases = json.load(f)

# --------------------------
# 2 - Lister les vidéos avec tags
# --------------------------
video_files = [f for f in os.listdir(VIDEOS_DIR) if f.lower().endswith((".mp4", ".mov", ".mkv"))]

videos_data = []
for vf in video_files:
    base_name = os.path.splitext(vf)[0]
    words = tokenize(base_name)
    abs_path = os.path.abspath(os.path.join(VIDEOS_DIR, vf))
    videos_data.append({
        "file": vf,
        "path": abs_path,  # Ajoute le chemin absolu
        "tags": words
    })

# --------------------------
# 3 - Associer chaque phrase à la meilleure vidéo
# --------------------------
build_plan = []

for phrase in phrases:
    phrase_words = tokenize(phrase["text"])

    best_match = None
    best_score = -1

    for vid in videos_data:
        score = count_common_words(phrase_words, vid["tags"])
        if score > best_score:
            best_score = score
            best_match = vid

    build_plan.append({
        "start": phrase["start"],
        "end": phrase["end"],
        "text": phrase["text"],
        "video": best_match["file"] if best_match else None,
        "video_path": best_match["path"] if best_match else None,  # Ajoute le chemin absolu ici
        "score": best_score
    })


# --------------------------
# 4 - Sauvegarder le plan de construction
# --------------------------
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(build_plan, f, indent=4, ensure_ascii=False)

print(f"[OK] Plan de construction créé dans {OUTPUT_FILE}")
