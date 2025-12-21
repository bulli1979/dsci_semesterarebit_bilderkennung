"""
Schritt 0: Bild laden
Lädt ein Bild aus dem Dataset und gibt es als PIL Image zurück.
"""

import numpy as np
from PIL import Image
from pathlib import Path
import pandas as pd
import os


def load_file(file_path):
    """Lädt eine Bilddatei und gibt ein PIL Image zurück."""
    return Image.open(file_path)


def choose_an_image(dataset_paths):
    """
    Wählt zufällig ein Bild aus dem Dataset aus.
    
    Args:
        dataset_paths: Dictionary mit Klassennamen als Keys und Path-Objekten als Values
        
    Returns:
        PIL Image und den Klassennamen
    """
    # Sammle alle Bilddateien
    all_files = []
    for klasse, path in dataset_paths.items():
        if path.exists():
            # Suche nach Bilddateien (jpg, jpeg, png, etc.)
            for ext in ['*.jpg', '*.jpeg', '*.JPG', '*.JPEG', '*.png', '*.PNG']:
                all_files.extend(list(path.glob(ext)))
    
    if not all_files:
        raise FileNotFoundError("Keine Bilddateien im Dataset gefunden!")
    
    # Wähle zufällig ein Bild
    fn = pd.Series(all_files).sample(1).values[0]
    
    # Bestimme die Klasse
    klasse = [k for k, p in dataset_paths.items() if str(fn).startswith(str(p))][0]
    
    print("Gewählte Klasse:", klasse)
    print("Datei:", fn.name)
    
    im = load_file(fn)
    return im, klasse


if __name__ == "__main__":
    # Test
    base_path = Path(os.getcwd()) / "data"
    dataset_paths = {
        "schraubenschluessel": base_path / "schraubenschluessel",
        "schraubenzieher": base_path / "schraubenezieher",
        "seidenschneider": base_path / "seidenschneider"
    }
    
    im, klasse = choose_an_image(dataset_paths)
    print(f"Bild geladen: {im.size}, Mode: {im.mode}")

