"""
Transfer Learning Modell mit MobileNetV2
EMPFOHLEN für kleine Datensätze (<100 Bilder pro Klasse)
Funktioniert deutlich besser als einfaches CNN bei wenig Daten
"""
import tensorflow as tf
from tensorflow.keras import layers


def make_model_transfer_learning(image_size, num_classes, fine_tune=True):
    """
    Erstellt ein Transfer Learning Modell basierend auf MobileNetV2.
    
    Args:
        image_size: Tuple (height, width) der Eingabebilder
        num_classes: Anzahl der Klassen
        fine_tune: Wenn True, werden alle Layer trainiert (besser, aber langsamer)
                   Wenn False, werden nur Top-Layer trainiert (schneller)
    
    Returns:
        Keras Model
    """
    color_image_size = (*image_size, 3)
    
    # Basis-Modell: MobileNetV2 (vorgefertigte Gewichte von ImageNet)
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=color_image_size,
        include_top=False,  # Entferne die letzte Klassifikationsschicht
        weights='imagenet'   # Verwende vorgefertigte Gewichte
    )
    
    # Freeze base model (optional)
    base_model.trainable = fine_tune  # True = Fine-Tuning, False = nur Top-Layer
    
    # Erweiterte Data Augmentation
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
        layers.RandomBrightness(0.2),
        layers.RandomContrast(0.2),
    ], name='data_augmentation')
    
    # Modell aufbauen
    inputs = tf.keras.Input(shape=color_image_size, name='Eingabe')
    x = layers.Rescaling(1./255)(inputs)
    x = data_augmentation(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)  # Regularisierung gegen Overfitting
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='Ausgabe')(x)
    
    model = tf.keras.Model(inputs, outputs, name='MobileNetV2-Transfer')
    return model

