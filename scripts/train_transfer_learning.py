"""
Training-Script für Transfer Learning Modell (MobileNetV2)
"""
import tensorflow as tf
from pathlib import Path
from typing import Tuple, Optional, Dict
import sys

# Import Model
try:
    from model_transfer_learning import make_model_transfer_learning
except ImportError:
    scripts_dir = Path(__file__).parent
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from model_transfer_learning import make_model_transfer_learning


def calculate_class_weights(train_ds, class_names) -> Dict[int, float]:
    """Berechnet Class Weights für unausgewogene Datensätze."""
    class_counts = {}
    for images, labels in train_ds.unbatch():
        label_idx = labels.numpy()
        class_counts[label_idx] = class_counts.get(label_idx, 0) + 1
    
    total_samples = sum(class_counts.values())
    class_weights = {}
    for label_idx in sorted(class_counts.keys()):
        count = class_counts[label_idx]
        weight = total_samples / (len(class_names) * count)
        class_weights[label_idx] = weight
    
    return class_weights


def train_transfer_learning(
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    class_names: list,
    image_size: Tuple[int, int],
    model_name: str = "model_Werkzeuge_transfer_learning",
    epochs: int = 200,
    patience: int = 100,
    fine_tune: bool = True,
    model_output_path: Optional[Path] = None
) -> Tuple[tf.keras.Model, tf.keras.callbacks.History]:
    """
    Trainiert ein Transfer Learning Modell basierend auf MobileNetV2.
    
    Args:
        train_ds: TensorFlow Dataset mit Trainingsdaten
        val_ds: TensorFlow Dataset mit Validierungsdaten
        class_names: Liste der Klassennamen
        image_size: Tuple (height, width) der Eingabebilder
        model_name: Name des Modells (für Speicherung)
        epochs: Maximale Anzahl Epochen
        patience: Patience für EarlyStopping
        fine_tune: Wenn True, werden alle Layer trainiert
        model_output_path: Pfad zum Speichern des finalen Modells
        
    Returns:
        Tuple (trained_model, training_history)
    """
    if model_output_path is None:
        model_output_path = Path('model_output') / 'model'
    
    model_output_path.mkdir(parents=True, exist_ok=True)
    
    # Modell erstellen
    num_classes = len(class_names)
    model = make_model_transfer_learning(
        image_size=image_size,
        num_classes=num_classes,
        fine_tune=fine_tune
    )
    
    # Modell kompilieren
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy']
    )
    
    # Class Weights berechnen
    class_weights = calculate_class_weights(train_ds, class_names)
    
    # Callbacks
    checkpoint_dir = Path('checkpoints')
    checkpoint_dir.mkdir(exist_ok=True)
    
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=patience,
        restore_best_weights=True
    )
    
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(checkpoint_dir / f'{model_name}_best.keras'),
        monitor='val_accuracy',
        save_best_only=True,
        save_weights_only=False,
        mode='max',
        verbose=1
    )
    
    # Training
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
    
    return model, history

