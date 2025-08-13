import json
import os
from datetime import timedelta

def time_str_to_seconds(t):
    h, m, s = t.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)

def seconds_to_time_str(sec):
    td = timedelta(seconds=sec)
    total_sec = td.total_seconds()
    hours = int(total_sec // 3600)
    minutes = int((total_sec % 3600) // 60)
    seconds = total_sec % 60
    return f"{hours}:{minutes:02d}:{seconds:09.6f}"

input_file = "phrases_with_best_videos.json"
output_file = "video_build.json"

# Chemin absolu vers les vidéos
VIDEOS_BASE_DIR = r"D:\Principal\Vidéo\Banque de donnée Vidéo\Perso"

with open(input_file, "r", encoding="utf-8") as f:
    phrases_data = json.load(f)

video_build_data = []

for phrase in phrases_data:
    start_sec = time_str_to_seconds(phrase["start"])
    end_sec = time_str_to_seconds(phrase["end"])
    phrase_duration = end_sec - start_sec

    best_videos = phrase.get("best_videos", [])
    if len(best_videos) == 0 or phrase_duration <= 0:
        continue

    # On ne prend que les 3 meilleures vidéos max
    best_videos = best_videos[:3]

    # Durée de chaque segment (équitablement répartie)
    segment_duration = phrase_duration / len(best_videos)

    current_start = start_sec
    for vid in best_videos:
        current_end = current_start + segment_duration

        npy_path = vid["npy_path"]
        video_name = os.path.basename(npy_path).replace(".npy", ".mp4")
        video_path = os.path.join(VIDEOS_BASE_DIR, video_name)

        video_build_data.append({
            "start": seconds_to_time_str(current_start),
            "end": seconds_to_time_str(current_end),
            "video": video_name,
            "video_path": video_path
        })

        current_start = current_end  # On enchaine

# Sauvegarde
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(video_build_data, f, indent=4, ensure_ascii=False)

print(f" Fichier généré : {output_file}")
