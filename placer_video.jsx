/*
Script ExtendScript pour Premiere Pro
Lit un fichier JSON avec plan de montage et place les vidéos au bon timecode
*/

var jsonFilePath = "D:/Principal/Vidéo/Developpement_outil_automatisation_video/clip-ingest/video_build.json";

// Lecture du fichier JSON
function readJSONFile(path) {
    var file = new File(path);
    if (!file.exists) {
        alert("Fichier JSON introuvable : " + path);
        return null;
    }
    file.open("r");
    var content = file.read();
    file.close();
    return JSON.parse(content);
}

// Convertit un timecode string en secondes (0:00:06.840000)
function timecodeToSeconds(tc) {
    var parts = tc.split(":");
    var h = parseInt(parts[0], 10);
    var m = parseInt(parts[1], 10);
    var s = parseFloat(parts[2]);
    return h * 3600 + m * 60 + s;
}

// Importation et insertion dans la séquence
function insertClipsFromPlan(plan) {
    if (app.project.activeSequence == null) {
        alert("Aucune séquence active !");
        return;
    }
    var seq = app.project.activeSequence;

    for (var i = 0; i < plan.length; i++) {
        var item = plan[i];
        var videoPath = item.video_path;

        // Importer la vidéo si elle n'existe pas déjà dans le chutier
        app.project.importFiles([videoPath]);

        // Trouver le clip importé
        var importedClip = null;
        var allItems = app.project.rootItem.children;
        for (var j = 0; j < allItems.length; j++) {
            if (allItems[j].name == decodeURI(videoPath.split("/").pop()) ||
                allItems[j].name == decodeURI(videoPath.split("\\").pop())) {
                importedClip = allItems[j];
                break;
            }
        }

        if (importedClip) {
            var startSec = timecodeToSeconds(item.start);
            var endSec = timecodeToSeconds(item.end);
            var duration = endSec - startSec;

            // Poser le clip dans la piste V1 au bon timecode
            seq.videoTracks[0].insertClip(importedClip, startSec);

            // Raccourcir le clip à la bonne durée
            var placedClip = seq.videoTracks[0].clips[seq.videoTracks[0].clips.numItems - 1];
            if (placedClip) {
                placedClip.end = placedClip.start.seconds + duration;
            }
        } else {
            $.writeln("Impossible de trouver le clip importé : " + videoPath);
        }
    }
}

var plan = readJSONFile(jsonFilePath);
if (plan) {
    insertClipsFromPlan(plan);
}
