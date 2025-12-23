"""
Einfaches CNN-Modell für Bildklassifikation
Verwendet für kleine Datensätze, aber funktioniert schlecht mit sehr wenig Daten (<100 Bilder)
"""
import tensorflow as tf
from tensorflow.keras import layers


def make_model_simple_cnn(image_size, num_classes):
    """
    Erstellt ein einfaches CNN-Modell mit 3 Conv2D-Schichten.
    
    Args:
        image_size: Tuple (height, width) der Eingabebilder
        num_classes: Anzahl der Klassen
    
    Returns:
        Keras Model
    """
    color_image_size = (*image_size, 3)
    
    img_inputs = tf.keras.Input(shape=color_image_size, name='Eingabe')
    
    x = layers.Rescaling(1./255, input_shape=color_image_size)(img_inputs)
    x = layers.RandomFlip("horizontal")(x)
    x = layers.RandomRotation(0.5)(x)
    x = layers.Conv2D(16, 3, padding='same', activation='relu')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(num_classes, name='Ausgabe', activation='softmax')(x)
    
    model = tf.keras.Model(inputs=img_inputs, outputs=outputs, name='Conv-Model-Standard')
    return model

