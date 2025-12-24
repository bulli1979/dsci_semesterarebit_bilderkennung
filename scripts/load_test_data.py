"""
Funktionen zum Laden von Testdaten für die Evaluation
"""
import tensorflow as tf
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional


def lade_testdaten_mit_pfad(
    baumstrukturpfad: Path,
    image_size: Tuple[int, int] = (128, 128),
    batch_size: int = 32,
    suffixes: Optional[List[str]] = None,
    shuffle: bool = False
) -> Tuple[tf.data.Dataset, np.ndarray, np.ndarray, List[str]]:
    """
    Baut ein tf.Dataset basierend auf den Daten in der Baumstruktur.
    
    Args:
        baumstrukturpfad: Pfad zur Baumstruktur mit Klassen-Unterordnern
        image_size: Größe der Bilder (height, width)
        batch_size: Batch-Größe für das Dataset
        suffixes: Liste der Dateiendungen (z.B. ['.jpg', '.jpeg'])
        shuffle: Ob die Daten gemischt werden sollen
        
    Returns:
        Tuple (dataset, file_paths, labels, class_names)
        - dataset: TensorFlow Dataset
        - file_paths: Array mit Dateipfaden
        - labels: Array mit numerischen Labels
        - class_names: Liste der Klassennamen (alphabetisch sortiert)
    """
    if suffixes is None:
        suffixes = ['.jpg', '.jpeg', '.JPG', '.JPEG', '.png', '.PNG']
    
    # Alle Bildpfade erfassen (verschiedene Dateiendungen)
    all_files = []
    for suffix in suffixes:
        all_files.extend(list(Path(baumstrukturpfad).rglob(f"*/*{suffix}")))
    
    all_files = sorted(list(set(all_files)))  # Entferne Duplikate
    class_names = sorted({p.parent.name for p in all_files})
    label_map = {name: idx for idx, name in enumerate(class_names)}

    # Pfade und Labels vorbereiten
    file_paths = np.array([str(p) for p in all_files])
    labels = np.array([label_map[Path(p).parent.name] for p in all_files])
    
    def lade_bild(pfad, label):
        """Lädt und verarbeitet ein einzelnes Bild"""
        image = tf.io.read_file(pfad)
        # Versuche verschiedene Decoder
        try:
            image = tf.image.decode_jpeg(image, channels=3)
        except:
            try:
                image = tf.image.decode_png(image, channels=3)
            except:
                # Fallback: als JPEG behandeln
                image = tf.image.decode_image(image, channels=3)
        image = tf.image.resize(image, image_size)
        image = tf.cast(image, tf.float32) / 255.0  # Normalisierung
        return image, label, pfad
        
    if shuffle:
        # Gleicher Shuffle-Index für beide
        shuffle_idx = np.random.permutation(len(file_paths))
        # Koordiniert shuffeln
        file_paths_shuffled = file_paths[shuffle_idx]
        labels_shuffled = labels[shuffle_idx]
        # Dataset bauen
        ds_fn = tf.data.Dataset.from_tensor_slices((file_paths_shuffled, labels_shuffled))
        ds = ds_fn.map(lade_bild).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return ds, file_paths_shuffled, labels_shuffled, class_names
    else:
        # Dataset bauen
        ds_fn = tf.data.Dataset.from_tensor_slices((file_paths, labels))
        ds = ds_fn.map(lade_bild).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return ds, file_paths, labels, class_names

