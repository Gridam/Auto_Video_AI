from sentence_transformers import SentenceTransformer, util
import pysrt
from datetime import timedelta
import json

# ---------------------
# PARAMÈTRES
# ---------------------
TAGS_FILE = "tags.txt"
SRT_FILE = "Main_sequence.srt" #"test.srt"
MAX_TAGS = 5  # nombre max de tags par phrase

# Charger modèle d'embedding déjà utilisé
model = SentenceTransformer("all-mpnet-base-v2")

# Charger liste de tags
with open(TAGS_FILE, "r", encoding="utf-8") as f:
    tags_list = [line.strip() for line in f if line.strip()]

# Charger sous-titres
subs = pysrt.open(SRT_FILE)

# ---------------------
# UTILITAIRES
# ---------------------
def time_to_seconds(t):
    return t.hours * 3600 + t.minutes * 60 + t.seconds + t.milliseconds / 1000

def seconds_to_time(sec):
    return str(timedelta(seconds=sec))

def fusionner_phrases(transcriptions):
    blocs = []
    courant_start = time_to_seconds(transcriptions[0].start)
    courant_text = transcriptions[0].text.strip()
    courant_end = time_to_seconds(transcriptions[0].end)

    for sub in transcriptions[1:]:
        text = sub.text.strip()
        courant_text += " " + text
        courant_end = time_to_seconds(sub.end)

        if text.endswith((".", "?", "!")):
            blocs.append({
                "start": seconds_to_time(courant_start),
                "end": seconds_to_time(courant_end),
                "text": courant_text.strip()
            })
            courant_text = ""
            if sub != transcriptions[-1]:
                courant_start = time_to_seconds(sub.end)

    if courant_text:
        blocs.append({
            "start": seconds_to_time(courant_start),
            "end": seconds_to_time(courant_end),
            "text": courant_text.strip()
        })

    return blocs

def trouver_tags(phrase):
    phrase_emb = model.encode(phrase, convert_to_tensor=True)
    tags_emb = model.encode(tags_list, convert_to_tensor=True)
    scores = util.cos_sim(phrase_emb, tags_emb)[0]
    tag_score_pairs = list(zip(tags_list, scores))
    tag_score_pairs.sort(key=lambda x: x[1], reverse=True)
    return [tag for tag, _ in tag_score_pairs[:MAX_TAGS]]

def decouper_en_segments(phrase, start_tc, end_tc):
    mots = phrase.split()
    segments = []
    seg_words = []
    start_time = start_tc
    total_duration = end_tc - start_tc
    duration_per_word = total_duration / len(mots)

    for i, mot in enumerate(mots, 1):
        seg_words.append(mot)
        if mot.endswith((".", "?", "!")):
            seg_text = " ".join(seg_words)
            seg_end_time = start_time + len(seg_words) * duration_per_word
            segments.append({
                "start": seconds_to_time(start_time),
                "end": seconds_to_time(seg_end_time),
                "text": seg_text
            })
            start_time = seg_end_time
            seg_words = []

    if seg_words:
        seg_text = " ".join(seg_words)
        seg_end_time = start_time + len(seg_words) * duration_per_word
        segments.append({
            "start": seconds_to_time(start_time),
            "end": seconds_to_time(seg_end_time),
            "text": seg_text
        })

    return segments

# ---------------------
# PIPELINE
# ---------------------
phrases = fusionner_phrases(subs)
resultats = []

for bloc in phrases:
    start_sec = sum(float(x) * 60 ** i for i, x in enumerate(reversed(bloc["start"].split(":"))))
    end_sec = sum(float(x) * 60 ** i for i, x in enumerate(reversed(bloc["end"].split(":"))))
    
    tags = trouver_tags(bloc["text"])
    segments = decouper_en_segments(bloc["text"], start_sec, end_sec)
    
    resultats.append({
        "phrase": bloc["text"],
        "start": bloc["start"],
        "end": bloc["end"],
        "tags": tags,
        "segments": segments
    })

# Exemple d'affichage
for r in resultats:
    print("\nPhrase:", r["phrase"])
    print("Tags:", r["tags"])
    print("Segments:", r["segments"])


# ---------------------
# PIPELINE
# ---------------------
phrases = fusionner_phrases(subs)

# Construction de la liste pour le JSON
phrases_json = []
for bloc in phrases:
    phrases_json.append({
        "start": bloc["start"],
        "end": bloc["end"],
        "text": bloc["text"]
    })

# Sauvegarde dans phrases.json
with open("phrases.json", "w", encoding="utf-8") as f:
    json.dump(phrases_json, f, ensure_ascii=False, indent=4)

print("Fichier phrases.json généré avec", len(phrases_json), "entrées.")