# Semesterarbeit: Werkzeug-Erkennung mit Deep Learning

Diese Semesterarbeit beschäftigt sich mit der automatischen Erkennung von Werkzeugen in Bildern mithilfe von Deep Learning. Das Projekt umfasst drei Hauptschritte: Objektextraktion aus Bildern, Training eines neuronalen Netzes und Evaluation des trainierten Modells.

## 📋 Inhaltsverzeichnis

- [Übersicht](#übersicht)
- [Projektstruktur](#projektstruktur)
- [Voraussetzungen](#voraussetzungen)
- [Installation](#installation)
- [Notebooks](#notebooks)
- [Verwendung](#verwendung)
- [Konfiguration](#konfiguration)
- [Ergebnisse](#ergebnisse)

## 🎯 Übersicht

Das Projekt zielt darauf ab, drei verschiedene Werkzeugtypen automatisch zu erkennen:

- **Schraubenschlüssel** (schraubenschluessel)
- **Schraubenzieher** (schraubenzieher)
- **Seidenschneider** (seidenschneider)

Der Workflow besteht aus drei Hauptphasen:

1. **Objektextraktion**: Extraktion von Werkzeugen aus Originalbildern
2. **Training**: Training eines neuronalen Netzes zur Klassifikation
3. **Evaluation**: Bewertung der Modell-Performance auf Testdaten

## 📁 Projektstruktur

```
dsci-semesterarebeit/
├── Baumstruktur/
│   ├── Baumstruktur_train/     # Trainingsdaten (nach Extraktion)
│   └── Baumstruktur_test/      # Testdaten (nach Extraktion)
├── data/                        # Original-Bilddaten
│   ├── schraubenschluessel/
│   ├── schraubenzieher/
│   └── seidenschneider/
├── scripts/                     # Python-Skripte für Verarbeitung
│   ├── semesterarbeit_config.py        # Zentrale Konfiguration
│   ├── semesterarbeit_preparation.py   # Vorbereitungsfunktionen
│   ├── object_extraction.py            # Objektextraktions-Pipeline
│   ├── schritt0_bild_laden.py          # Bildladen
│   ├── schritt1_aufloesungsreduktion.py # Auflösungsreduktion
│   ├── schritt2_raender_abschneiden.py # Ränder abschneiden
│   ├── schritt3_hintergrundschwellwert.py # Hintergrund-Erkennung
│   ├── schritt4_morphologische_transformation.py # Morphologische Operationen
│   ├── schritt5_objekte_extrahieren.py # Objektextraktion
│   ├── schritt6_kleine_objekte_filtern.py # Filterung kleiner Objekte
│   ├── model_simple_cnn.py             # Einfaches CNN-Modell
│   ├── model_transfer_learning.py      # Transfer Learning Modell
│   ├── train_model.py                  # Training-Funktionen
│   └── evaluate_model.py               # Evaluation-Funktionen
├── checkpoints/                 # Gespeicherte Modell-Checkpoints
├── evaluation_results/         # Evaluationsergebnisse (Plots, Matrizen)
├── model_output/               # Trainierte Modelle
├── semester-arbeit-objekt-abstraktion.ipynb  # Notebook 1: Objektextraktion
├── semesterarbeit-training.ipynb             # Notebook 2: Training
├── semesterarbeit-evaluation.ipynb          # Notebook 3: Evaluation
└── requirements.txt            # Python-Abhängigkeiten
```

## 🔧 Voraussetzungen

### Systemanforderungen

- **Python**: Version 3.8 oder höher
- **Betriebssystem**: Windows, Linux oder macOS
- **RAM**: Mindestens 8 GB empfohlen (für TensorFlow)
- **GPU**: Optional, aber empfohlen für schnelleres Training (CUDA-kompatible GPU)

### Software-Abhängigkeiten

Alle benötigten Python-Pakete sind in `requirements.txt` aufgelistet:

- **NumPy** (>=1.20.0): Numerische Berechnungen
- **Matplotlib** (>=3.3.0): Visualisierung
- **scikit-image** (>=0.18.0): Bildverarbeitung
- **scikit-learn** (>=1.0.0): Machine Learning Utilities
- **SciPy** (>=1.7.0,<1.11.0): Wissenschaftliche Berechnungen
- **progressbar2** (>=4.0.0): Fortschrittsanzeigen
- **Pillow** (>=8.0.0): Bildverarbeitung
- **Pandas** (>=1.3.0): Datenmanipulation
- **TensorFlow** (>=2.10.0): Deep Learning Framework
- **scikit-plot** (>=0.3.7): Visualisierung von Metriken
- **Seaborn** (>=0.11.0): Statistische Visualisierung

### Zusätzliche Anforderungen

- **Jupyter Notebook** oder **JupyterLab** für die Ausführung der Notebooks
- **Git** (optional) für Versionskontrolle

## 📦 Installation

### 1. Repository klonen oder herunterladen

```bash
git clone <repository-url>
cd dsci-semesterarebeit
```

### 2. Python-Umgebung erstellen (empfohlen)

```bash
# Mit venv
python -m venv venv
source venv/bin/activate  # Linux/macOS
# oder
venv\Scripts\activate     # Windows

# Mit conda
conda create -n semesterarbeit python=3.9
conda activate semesterarbeit
```

### 3. Abhängigkeiten installieren

```bash
pip install -r requirements.txt
```

**Hinweis**: Falls Probleme mit NumPy 2.x auftreten, können Sie NumPy auf Version 1.x downgraden:

```bash
pip install "numpy<2"
```

### 4. Jupyter Notebook installieren (falls noch nicht vorhanden)

```bash
pip install jupyter jupyterlab
```

## 📓 Notebooks

Das Projekt besteht aus drei Hauptnotebooks, die in sequenzieller Reihenfolge ausgeführt werden sollten:

### 1. `semester-arbeit-objekt-abstraktion.ipynb`

**Zweck**: Extraktion von Werkzeugen aus Originalbildern

**Funktionalität**:

- Lädt Bilder aus dem `data/` Verzeichnis
- Führt eine Bildverarbeitungspipeline durch:
  1. **Auflösungsreduktion**: Reduziert die Bildgröße (längere Seite auf 400px)
  2. **Ränder abschneiden**: Entfernt Ränder basierend auf konfigurierbaren Parametern
  3. **Hintergrund-Erkennung**: Erstellt Masken basierend auf Schwellwerten (HSV-Farbraum)
  4. **Morphologische Transformation**: Verbessert die Masken durch Erosion und Dilatation
  5. **Objektextraktion**: Extrahiert einzelne Objekte aus den Bildern
  6. **Filterung**: Entfernt kleine Objekte unter einer Mindestgröße
- Speichert extrahierte Objekte in einer Baumstruktur (`Baumstruktur_train/` und `Baumstruktur_test/`)
- Erstellt ZIP-Dateien für einfachen Datentransfer (`training_Baumstruktur.zip`, `testing_Baumstruktur.zip`)

**Ausgabe**:

- Extrahiertes Objektbilder in `Baumstruktur/Baumstruktur_train/` und `Baumstruktur/Baumstruktur_test/`
- ZIP-Dateien mit der Baumstruktur für Training und Test

**Wichtige Parameter** (in `scripts/semesterarbeit_config.py`):

- `pr`: Auflösung (längere Seite in Pixeln, Standard: 400)
- `vth`: Value-Threshold für Hintergrund-Erkennung (Standard: 140)
- `mpx`: Mindestanzahl Pixel pro Objekt (Standard: 2000)
- `es`, `ds`: Erosions- und Dilatationsgröße für morphologische Operationen

### 2. `semesterarbeit-training.ipynb`

**Zweck**: Training von neuronalen Netzen zur Werkzeug-Klassifikation

**Funktionalität**:

- Entpackt die ZIP-Dateien mit extrahierten Objekten
- Lädt Trainings- und Validierungsdaten mit TensorFlow
- Definiert zwei Modell-Architekturen:
  - **Simple CNN**: Einfaches Convolutional Neural Network (für größere Datensätze)
  - **Transfer Learning**: MobileNetV2-basiertes Modell (empfohlen für kleine Datensätze)
- Trainiert das Modell mit:
  - Data Augmentation (Rotation, Verschiebung, Zoom, etc.)
  - Early Stopping (verhindert Overfitting)
  - Model Checkpointing (speichert beste Gewichte)
  - Class Weights (für unausgewogene Datensätze)
- Visualisiert den Trainingsverlauf (Accuracy, Loss)
- Speichert das trainierte Modell

**Ausgabe**:

- Trainiertes Modell (`.keras` Datei) in `checkpoints/` oder `model_output/`
- Trainingshistorie-Plots in `evaluation_results/`
- Confusion Matrix für Validierungsdaten

**Wichtige Parameter** (in `scripts/semesterarbeit_config.py`):

- `image_size`: Eingabebildgröße (Standard: (400, 400))
- `batch_size`: Batch-Größe (Standard: 32)
- `epochs`: Maximale Epochen (Standard: 100)
- `patience`: Early Stopping Patience (Standard: 100)
- `learning_rate`: Lernrate (Standard: 0.0001)
- `validation_split`: Anteil der Validierungsdaten (Standard: 0.2)

### 3. `semesterarbeit-evaluation.ipynb`

**Zweck**: Evaluation des trainierten Modells auf Testdaten

**Funktionalität**:

- Lädt das trainierte Modell
- Lädt Testdaten aus `Baumstruktur/Baumstruktur_test/`
- Führt Vorhersagen auf allen Testbildern durch
- Berechnet Metriken:
  - **Classification Report**: Precision, Recall, F1-Score pro Klasse
  - **Confusion Matrix**: Visualisierung der Klassifikationsergebnisse
  - **Overall Accuracy**: Gesamtgenauigkeit
- Visualisiert Ergebnisse mit Plots
- Speichert Evaluationsergebnisse

**Ausgabe**:

- Classification Report (Text)
- Confusion Matrix (Plot) in `evaluation_results/`
- Detaillierte Metriken pro Klasse

**Wichtig**: Die Klassenreihenfolge muss mit dem Training-Notebook übereinstimmen:

- Index 0: schraubenschluessel
- Index 1: schraubenzieher
- Index 2: seidenschneider

## 🚀 Verwendung

### Schritt-für-Schritt Anleitung

1. **Objektextraktion durchführen**

   ```bash
   jupyter notebook semester-arbeit-objekt-abstraktion.ipynb
   ```

   - Führen Sie alle Zellen nacheinander aus
   - Überprüfen Sie die extrahierten Objekte
   - Stellen Sie sicher, dass ZIP-Dateien erstellt wurden

2. **Modell trainieren**

   ```bash
   jupyter notebook semesterarbeit-training.ipynb
   ```

   - Entpacken Sie die ZIP-Dateien (erste Zelle)
   - Wählen Sie ein Modell (Transfer Learning empfohlen)
   - Führen Sie das Training durch
   - Notieren Sie sich die Klassenreihenfolge für die Evaluation

3. **Modell evaluieren**
   ```bash
   jupyter notebook semesterarbeit-evaluation.ipynb
   ```
   - Laden Sie das trainierte Modell
   - Stellen Sie sicher, dass die Klassenreihenfolge korrekt ist
   - Führen Sie die Evaluation durch
   - Analysieren Sie die Ergebnisse

### Verwendung der Skripte direkt

Alternativ können Sie die Python-Skripte auch direkt verwenden:

```python
# Beispiel: Objektextraktion
from scripts.object_extraction import process_file
from scripts.semesterarbeit_config import config

# Beispiel: Training
from scripts.train_model import train_transfer_learning

# Beispiel: Evaluation
from scripts.evaluate_model import evaluate_model
```

## ⚙️ Konfiguration

Die zentrale Konfiguration befindet sich in `scripts/semesterarbeit_config.py`. Hier können Sie alle wichtigen Parameter anpassen:

### Extraktions-Parameter

- `pr`: Auflösung (längere Seite in Pixeln)
- `vth`: Value-Threshold für Hintergrund-Erkennung
- `mpx`: Mindestanzahl Pixel pro Objekt
- `es`, `ds`: Morphologische Operationen (Erosion, Dilatation)

### Training-Parameter

- `image_size`: Eingabebildgröße (muss mit Extraktion übereinstimmen!)
- `batch_size`: Batch-Größe
- `epochs`: Maximale Epochen
- `learning_rate`: Lernrate

### Evaluation-Parameter

- `image_size`: Muss mit Training übereinstimmen!
- `batch_size`: Batch-Größe für Evaluation

**Wichtig**: Die `image_size` in Training und Evaluation muss mit der Auflösung `pr` aus der Extraktion übereinstimmen!

## 📊 Ergebnisse

Die Evaluationsergebnisse werden in `evaluation_results/` gespeichert:

- `*_training_history.png`: Trainingsverlauf (Accuracy, Loss über Epochen)
- `*_confusion_matrix.png`: Confusion Matrix für Validierungs- und Testdaten
- `*_final_*.png`: Finale Ergebnisse nach vollständigem Training

Trainierte Modelle werden in `checkpoints/` oder `model_output/` gespeichert:

- `model_Werkzeuge_*.keras`: Trainierte Modell-Dateien

## 🔍 Troubleshooting

### Häufige Probleme

1. **Import-Fehler**: Stellen Sie sicher, dass alle Abhängigkeiten installiert sind:

   ```bash
   pip install -r requirements.txt
   ```

2. **NumPy-Versionskonflikt**: Downgrade auf NumPy 1.x:

   ```bash
   pip install "numpy<2"
   ```

3. **TensorFlow-Fehler**: Stellen Sie sicher, dass TensorFlow korrekt installiert ist:

   ```bash
   pip install tensorflow
   ```

4. **Pfad-Probleme**: Stellen Sie sicher, dass Sie im richtigen Verzeichnis arbeiten und die Notebooks im Hauptverzeichnis ausführen.

5. **Klassenreihenfolge**: Die Reihenfolge der Klassen muss in Training und Evaluation identisch sein!

## 📝 Hinweise

- Die Notebooks sollten in der angegebenen Reihenfolge ausgeführt werden
- Stellen Sie sicher, dass genügend Speicherplatz für die extrahierten Bilder vorhanden ist
- Für bessere Ergebnisse können Sie die Parameter in `semesterarbeit_config.py` anpassen
- Das Transfer Learning Modell funktioniert besser bei kleinen Datensätzen als das Simple CNN

## 📄 Lizenz

Dieses Projekt wurde im Rahmen einer Semesterarbeit erstellt.

## 👤 Autor

Semesterarbeit - Werkzeug-Erkennung mit Deep Learning

---

**Viel Erfolg mit dem Projekt!** 🚀
