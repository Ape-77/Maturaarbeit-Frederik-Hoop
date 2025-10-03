from ultralytics import YOLO
import csv
import os
from collections import defaultdict
# Importieren des Modells
model = YOLO("yolov8n.pt") # Für anderes Modell als Nano, bitte n durch s(Small), m(Medium), etc. ersetzen

sources = [
   r"C:\Users\Banane\Desktop\XXX.MP4" # Banane = Username, beim "XXX" muss der Name des Videos eingefügt werden z.B. Burgbach_Rain
# Es können auch mehrere Videos aufeinmal klassifiziert werde, dafür einfach Zeile 9 erneut einfügen und Namen abändern
    ]
# Ordner für CSV-Ergebnisse erstellen
os.makedirs("csv_results", exist_ok=True)
# Alle Videos durchgehen
for src in sources:
   video_name = os.path.basename(src).split(".")[0]   # Dateiname ohne Endung
   csv_path = f"csv_results/{video_name}_avg_results.csv"
   # Speicher für Wahrscheinlichkeiten pro Klasse
   class_scores = defaultdict(list)
   # YOLO auf das Video anwenden
   results_list = model.predict(src, save=True, imgsz=640)
   # Ergebnisse pro Frame sammeln
   for r in results_list:
       for box in r.boxes:
           conf = float(box.conf[0])     # Wahrscheinlichkeit
           cls = int(box.cls[0])         # Klassen-ID
           label = model.names[cls]      # Klassenname
           class_scores[label].append(conf)
   # Durchschnittswerte berechnen und in CSV speichern
   with open(csv_path, "w", newline="", encoding="utf-8") as f:
       writer = csv.writer(f)
       writer.writerow(["Objekt", "Durchschnitts-Wahrscheinlichkeit"])
       for label, confs in class_scores.items():
           avg_conf = sum(confs) / len(confs)
           writer.writerow([label, f"{avg_conf:.2f}"])
   print(f" Durchschnittswerte gespeichert: {csv_path}") # Durchschnittswerte gespeichert, erscheindt wenn alles geklappt hat