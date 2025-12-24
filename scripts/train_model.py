"""
Training-Script für das Transfer Learning Modell
Extrahiert aus semesterarbeit-training.ipynb
"""
import tensorflow as tf
from pathlib import Path
from typing import Dict, Tuple, Optional
import numpy as np

# Importiere die Abhängigkeiten - versuche verschiedene Wege
try:
    # Versuche relative Import (wenn aus scripts-Ordner importiert)
    from model_transfer_learning import make_model_transfer_learning
except ImportError:
    try:
        # Fallback: absoluter Import (wenn von außerhalb aufgerufen)
        from scripts.model_transfer_learning import make_model_transfer_learning
    except ImportError:
        # Letzter Fallback: direkter Import
        import sys
        from pathlib import Path
        scripts_dir = Path(__file__).parent
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        from model_transfer_learning import make_model_transfer_learning

try:
    # Versuche relative Import (wenn aus scripts-Ordner importiert)
    from model_simple_cnn import make_model_simple_cnn
except ImportError:
    try:
        # Fallback: absoluter Import (wenn von außerhalb aufgerufen)
        from scripts.model_simple_cnn import make_model_simple_cnn
    except ImportError:
        # Letzter Fallback: direkter Import
        import sys
        from pathlib import Path
        scripts_dir = Path(__file__).parent
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        from model_simple_cnn import make_model_simple_cnn


def calculate_class_weights(train_ds, class_names) -> Dict[int, float]:
    """
    Berechnet Class Weights für unausgewogene Datensätze.
    
    Args:
        train_ds: TensorFlow Dataset mit Trainingsdaten
        class_names: Liste der Klassennamen
        
    Returns:
        Dictionary mit Class Weights {class_index: weight}
    """
    class_counts = {}
    
    # Zähle Bilder pro Klasse
    for images, labels in train_ds.unbatch():
        label_idx = labels.numpy()
        class_counts[label_idx] = class_counts.get(label_idx, 0) + 1
    
    total_samples = sum(class_counts.values())
    class_weights = {}
    
    print(f"\nKlassenverteilung im Training:")
    for label_idx in sorted(class_counts.keys()):
        count = class_counts[label_idx]
        class_name = class_names[label_idx]
        # Inverse Frequenz: Klassen mit weniger Bildern bekommen höheres Gewicht
        weight = total_samples / (len(class_names) * count)
        class_weights[label_idx] = weight
        print(f"  {class_name:20s}: {count:4d} Bilder → Gewicht: {weight:.3f}")
    
    print(f"\n✓ Class Weights berechnet")
    print(f"  → Diese werden im Training verwendet, um unausgewogene Daten auszugleichen")
    
    return class_weights


def create_callbacks(checkpoint_dir: Path, patience: int = 100) -> Tuple:
    """
    Erstellt Callbacks für das Training.
    
    Args:
        checkpoint_dir: Verzeichnis für Model-Checkpoints
        patience: Patience für EarlyStopping
        
    Returns:
        Tuple (early_stopping_callback, model_checkpoint_callback)
    """
    checkpoint_dir.mkdir(exist_ok=True)
    
    # EarlyStopping: Stoppt das Training, wenn sich die Validierungsgenauigkeit nicht mehr verbessert
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=patience,
        restore_best_weights=True
    )
    
    # ModelCheckpoint: Speichert das beste Modell während des Trainings
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(checkpoint_dir / 'best_model.keras'),
        monitor='val_accuracy',
        save_best_only=True,
        save_weights_only=False,
        mode='max',
        verbose=1
    )
    
    print("✓ Callbacks erstellt:")
    print(f"  - EarlyStopping: Stoppt bei keiner Verbesserung (Patience: {patience})")
    print(f"  - ModelCheckpoint: Speichert bestes Modell in '{checkpoint_dir}/best_model.keras'")
    
    return early_stopping_callback, model_checkpoint_callback


def train_model(
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    class_names: list,
    image_size: Tuple[int, int],
    model_name: str = "model",
    epochs: int = 200,
    patience: int = 100,
    fine_tune: bool = True,
    checkpoint_dir: Optional[Path] = None,
    model_output_path: Optional[Path] = None
) -> Tuple[tf.keras.Model, tf.keras.callbacks.History]:
    """
    Trainiert ein Transfer Learning Modell.
    
    Args:
        train_ds: TensorFlow Dataset mit Trainingsdaten
        val_ds: TensorFlow Dataset mit Validierungsdaten
        class_names: Liste der Klassennamen
        image_size: Tuple (height, width) der Eingabebilder
        model_name: Name des Modells (für Speicherung)
        epochs: Maximale Anzahl Epochen
        patience: Patience für EarlyStopping
        fine_tune: Wenn True, werden alle Layer trainiert
        checkpoint_dir: Verzeichnis für Checkpoints (default: 'checkpoints')
        model_output_path: Pfad zum Speichern des finalen Modells
        
    Returns:
        Tuple (trained_model, training_history)
    """
    # Standard-Pfade
    if checkpoint_dir is None:
        checkpoint_dir = Path('checkpoints')
    if model_output_path is None:
        model_output_path = Path('model_output') / 'model'
    
    model_output_path.mkdir(parents=True, exist_ok=True)
    
    # Modell erstellen
    num_classes = len(class_names)
    print(f"\n🔄 Erstelle Transfer Learning Modell (MobileNetV2)...")
    model = make_model_transfer_learning(
        image_size=image_size,
        num_classes=num_classes,
        fine_tune=fine_tune
    )
    print(f"✓ Transfer Learning Modell erstellt!")
    print(f"  - Eingabegröße: {image_size}")
    print(f"  - Anzahl Klassen: {num_classes}")
    
    # Modell kompilieren
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy']
    )
    
    # Class Weights berechnen
    class_weights = calculate_class_weights(train_ds, class_names)
    
    # Callbacks erstellen
    early_stopping, model_checkpoint = create_callbacks(checkpoint_dir, patience)
    
    # Training durchführen
    print(f"\n🚀 Starte Training...")
    print(f"  - Maximale Epochen: {epochs}")
    print(f"  - Early Stopping Patience: {patience}")
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[early_stopping, model_checkpoint],
        class_weight=class_weights
    )
    
    # Finales Modell speichern
    final_model_path = model_output_path / f'{model_name}.keras'
    model.save(str(final_model_path))
    print(f"\n✓ Finales Modell gespeichert: {final_model_path}")
    
    return model, history


def train_simple_cnn_model(
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    class_names: list,
    image_size: Tuple[int, int],
    model_name: str = "model_simple_cnn",
    epochs: int = 100,
    checkpoint_dir: Optional[Path] = None,
    model_output_path: Optional[Path] = None
) -> Tuple[tf.keras.Model, tf.keras.callbacks.History]:
    """
    Trainiert ein einfaches CNN-Modell (wie in giraffenpuzzle-training.ipynb).
    
    Args:
        train_ds: TensorFlow Dataset mit Trainingsdaten
        val_ds: TensorFlow Dataset mit Validierungsdaten
        class_names: Liste der Klassennamen
        image_size: Tuple (height, width) der Eingabebilder
        model_name: Name des Modells (für Speicherung)
        epochs: Anzahl Epochen (default: 100 wie in giraffenpuzzle)
        checkpoint_dir: Verzeichnis für Checkpoints (default: 'checkpoints')
        model_output_path: Pfad zum Speichern des finalen Modells
        
    Returns:
        Tuple (trained_model, training_history)
    """
    # Standard-Pfade
    if checkpoint_dir is None:
        checkpoint_dir = Path('checkpoints')
    if model_output_path is None:
        model_output_path = Path('model_output') / 'model'
    
    model_output_path.mkdir(parents=True, exist_ok=True)
    
    # Modell erstellen (einfaches CNN wie in giraffenpuzzle)
    num_classes = len(class_names)
    print(f"\n🔄 Erstelle einfaches CNN-Modell (wie in giraffenpuzzle)...")
    model = make_model_simple_cnn(
        image_size=image_size,
        num_classes=num_classes
    )
    print(f"✓ Einfaches CNN-Modell erstellt!")
    print(f"  - Eingabegröße: {image_size}")
    print(f"  - Anzahl Klassen: {num_classes}")
    
    # Modell kompilieren
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy']
    )
    
    # Callbacks erstellen (ohne EarlyStopping für einfaches CNN, wie in giraffenpuzzle)
    checkpoint_dir.mkdir(exist_ok=True)
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(checkpoint_dir / f'{model_name}_best.keras'),
        monitor='val_accuracy',
        save_best_only=True,
        save_weights_only=False,
        mode='max',
        verbose=1
    )
    
    # Training durchführen (100 Epochen wie in giraffenpuzzle)
    print(f"\n🚀 Starte Training...")
    print(f"  - Epochen: {epochs}")
    print(f"  - Modelltyp: Einfaches CNN")
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[model_checkpoint_callback]
    )
    
    # Finales Modell speichern
    final_model_path = model_output_path / f'{model_name}.keras'
    model.save(str(final_model_path))
    print(f"\n✓ Finales Modell gespeichert: {final_model_path}")
    
    return model, history


if __name__ == "__main__":
    # Beispiel-Verwendung
    print("Dieses Script sollte aus dem Notebook heraus aufgerufen werden.")
    print("Siehe semesterarbeit-training.ipynb für die Verwendung.")

