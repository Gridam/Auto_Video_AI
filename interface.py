import tkinter as tk
from tkinter import scrolledtext
import subprocess
import threading
import os

# Chemin vers tes scripts
SCRIPTS = {
    "Analyse vidéo (.npy)": "clip_ingest.py",
    "Tag auto des vidéos": "clip_tag.py",
    "Trouver meilleures vidéos": "find_best_videos.py",
    "Construire fichier JSON final": "build_video_json.py",
    "Créer vidéo (à venir)": None  # pas encore implémenté
}

# Récupère le chemin absolu de l'interpréteur Python du venv
VENV_PYTHON = os.path.join(os.path.dirname(os.path.dirname(__file__)), "venv", "Scripts", "python.exe")

# Fonction pour exécuter un script et afficher la sortie
def run_script(script_path):
    if not script_path:
        log(" Script non défini pour ce bouton.")
        return

    if not os.path.exists(script_path):
        log(f"Script introuvable : {script_path}")
        return

    log(f"Lancement de {script_path}...")

    def task():
        process = subprocess.Popen(
            [VENV_PYTHON, script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        for line in process.stdout:
            log(line.strip())
        process.stdout.close()
        process.wait()
        log(f"Script terminé : {script_path}")

    threading.Thread(target=task, daemon=True).start()

# Fonction pour afficher un message dans la zone de log
def log(message):
    text_area.insert(tk.END, message + "\n")
    text_area.see(tk.END)

# Création de la fenêtre
root = tk.Tk()
root.title("Gestion des scripts vidéo")
root.geometry("700x500")

# Ligne 1 : Première suite
frame1 = tk.LabelFrame(root, text="Suite 1 : Analyse et Tag", padx=10, pady=10)
frame1.pack(fill="x", padx=10, pady=5)

tk.Button(frame1, text="Analyse vidéo (.npy)", width=25,
          command=lambda: run_script(SCRIPTS["Analyse vidéo (.npy)"])).pack(side="left", padx=5)

tk.Button(frame1, text="Tag auto des vidéos", width=25,
          command=lambda: run_script(SCRIPTS["Tag auto des vidéos"])).pack(side="left", padx=5)

# Ligne 2 : Deuxième suite
frame2 = tk.LabelFrame(root, text="Suite 2 : Choix et Placement", padx=10, pady=10)
frame2.pack(fill="x", padx=10, pady=5)

tk.Button(frame2, text="Trouver meilleures vidéos", width=25,
          command=lambda: run_script(SCRIPTS["Trouver meilleures vidéos"])).pack(side="left", padx=5)

tk.Button(frame2, text="Construire JSON final", width=25,
          command=lambda: run_script(SCRIPTS["Construire fichier JSON final"])).pack(side="left", padx=5)

# Bouton final
frame3 = tk.LabelFrame(root, text="Export final", padx=10, pady=10)
frame3.pack(fill="x", padx=10, pady=5)

tk.Button(frame3, text="Créer vidéo (à venir)", width=25,
          command=lambda: run_script(SCRIPTS["Créer vidéo (à venir)"])).pack(side="left", padx=5)

# Zone de log
text_area = scrolledtext.ScrolledText(root, wrap=tk.WORD, height=15)
text_area.pack(fill="both", padx=10, pady=10, expand=True)

root.mainloop()
