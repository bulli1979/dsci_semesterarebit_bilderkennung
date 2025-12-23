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


def choose_an_image(dataset_paths_or_folder, klasse_name=None):
    """
    Wählt ein Bild aus dem Dataset oder einem spezifischen Ordner aus.
    
    Args:
        dataset_paths_or_folder: 
            - Dictionary mit Klassennamen als Keys und Path-Objekten als Values, ODER
            - Ein einzelner Path-Objekt zu einem Ordner
        klasse_name: Optional, Klassename 
            - Wenn ein Dictionary übergeben wird: Wähle aus diesem spezifischen Ordner
            - Wenn ein einzelner Ordner übergeben wird: Verwende diesen Namen
        
    Returns:
        PIL Image und den Klassennamen
    """
    from pathlib import Path
    
    # Prüfe ob es ein Dictionary oder ein einzelner Pfad ist
    if isinstance(dataset_paths_or_folder, dict):
        dataset_paths = dataset_paths_or_folder
        
        # Wenn klasse_name angegeben ist, wähle aus diesem spezifischen Ordner
        if klasse_name is not None:
            if klasse_name not in dataset_paths:
                raise KeyError(f"Klasse '{klasse_name}' nicht im dataset_paths gefunden!")
            
            folder_path = dataset_paths[klasse_name]
            if not folder_path.exists():
                raise FileNotFoundError(f"Ordner existiert nicht: {folder_path}")
            
            # Sammle alle Bilddateien im Ordner
            all_files = []
            for ext in ['*.jpg', '*.jpeg', '*.JPG', '*.JPEG', '*.png', '*.PNG']:
                all_files.extend(list(folder_path.glob(ext)))
            
            if not all_files:
                raise FileNotFoundError(f"Keine Bilddateien im Ordner gefunden: {folder_path}")
            
            # Wähle das erste Bild
            fn = all_files[0]
            klasse = klasse_name
            
            print(f"Gewählte Klasse: {klasse}")
            print(f"Datei: {fn.name}")
            
            im = load_file(fn)
            return im, klasse
        else:
            # Alte Funktionalität: Wähle zufällig aus allen Ordnern
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
    else:
        # Neue Funktionalität: Wähle ein Bild aus einem spezifischen Ordner
        folder_path = Path(dataset_paths_or_folder)
        if not folder_path.exists():
            raise FileNotFoundError(f"Ordner existiert nicht: {folder_path}")
        
        # Sammle alle Bilddateien im Ordner
        all_files = []
        for ext in ['*.jpg', '*.jpeg', '*.JPG', '*.JPEG', '*.png', '*.PNG']:
            all_files.extend(list(folder_path.glob(ext)))
        
        if not all_files:
            raise FileNotFoundError(f"Keine Bilddateien im Ordner gefunden: {folder_path}")
        
        # Wähle das erste Bild (oder zufällig, wenn gewünscht)
        fn = all_files[0]
        
        # Verwende übergebenen Klassennamen oder versuche aus Pfad zu extrahieren
        if klasse_name is None:
            klasse = folder_path.name
        else:
            klasse = klasse_name
        
        print(f"Gewählte Klasse: {klasse}")
        print(f"Datei: {fn.name}")
        
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

