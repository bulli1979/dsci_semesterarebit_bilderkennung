"""
Training-Script für Simple CNN Modell
"""
import tensorflow as tf
from pathlib import Path
from typing import Tuple, Optional
import sys

# Import Model
try:
    from model_simple_cnn import make_model_simple_cnn
except ImportError:
    scripts_dir = Path(__file__).parent
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from model_simple_cnn import make_model_simple_cnn


def train_simple_cnn(
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    class_names: list,
    image_size: Tuple[int, int],
    model_name: str = "model_Werkzeuge_simple_cnn",
    epochs: int = 100,
    model_output_path: Optional[Path] = None
) -> Tuple[tf.keras.Model, tf.keras.callbacks.History]:
    """
    Trainiert ein einfaches CNN-Modell.
    
    Args:
        train_ds: TensorFlow Dataset mit Trainingsdaten
        val_ds: TensorFlow Dataset mit Validierungsdaten
        class_names: Liste der Klassennamen
        image_size: Tuple (height, width) der Eingabebilder
        model_name: Name des Modells (für Speicherung)
        epochs: Anzahl Epochen
        model_output_path: Pfad zum Speichern des finalen Modells
        
    Returns:
        Tuple (trained_model, training_history)
    """
    if model_output_path is None:
        model_output_path = Path('model_output') / 'model'
    
    model_output_path.mkdir(parents=True, exist_ok=True)
    
    # Modell erstellen
    num_classes = len(class_names)
    model = make_model_simple_cnn(
        image_size=image_size,
        num_classes=num_classes
    )
    
    # Modell kompilieren
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy']
    )
    
    # Callbacks
    checkpoint_dir = Path('checkpoints')
    checkpoint_dir.mkdir(exist_ok=True)
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
        callbacks=[model_checkpoint]
    )
    
    # Finales Modell speichern
    final_model_path = model_output_path / f'{model_name}.keras'
    model.save(str(final_model_path))
    
    return model, history

