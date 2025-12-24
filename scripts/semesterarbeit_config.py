from typing import Tuple


def get_config() -> dict:
    """
    Erstellt ein vollständiges Konfigurationsobjekt für die gesamte Arbeit.
    Alle Werte sind in der Funktion gesetzt.
    
    Returns:
    --------
    dict : Konfigurationsobjekt mit 3 Bereichen:
        - extraction: Parameter für die Objektextraktion
        - training: Parameter für das Modell-Training
        - evaluation: Parameter für die Modell-Evaluation
    """
    return {
        'extraction': {
            # Auflösung: Längere Seite wird auf 400px gesetzt (bessere Qualität)
            # WICHTIG: Diese Größe bestimmt auch die image_size für Training und Evaluation!
            'pr': 400,  # Längere Seite wird auf 400px gesetzt (statt feste Höhe/Breite)
            
            # Ränder vorsichtiger abschneiden
            'fr': 0.001,  # FRACTION_OF_ROWS_TO_REMOVE
            'fc': 0.13,   # FRACTION_OF_COLS_TO_REMOVE
            
            # Schwellwert optimieren (wichtig für saubere Masken!)
            'vth': 140,  # VALUE_THRESHOLD Farbpalette alles < 150 ist Hintergrund
            # Tipp: Testen Sie verschiedene Werte (120-180) und wählen Sie den besten
            # Niedrigere Werte = mehr Pixel werden als Objekt erkannt (kann auch Hintergrund einschließen)
            # Höhere Werte = nur sehr helle Pixel werden erkannt (kann Teile des Objekts ausschließen)
            
            # Morphologische Operationen optimieren
            'es': 0,  # EROSION_SIZE (AUF 0 GESETZT - behält das gesamte Objekt!)
            'ds': 10,  # DILATION_SIZE (REDUZIERT auf 5 - minimale Transformation, da ursprüngliche Maske besser ist!)
            
            # Minimum Pixel erhöhen für größere, sauberere Objekte
            'mpx': 2000,  # Erhöht von 1000 - filtert kleine Artefakte besser
            
            # HSV-Schwellwerte (falls benötigt)
            'saturation_threshold': (0, 60),     # Weiß hat extrem wenig S
            'value_threshold': (180, 255),       # Weiß ist sehr hell
            
            # Dateiendung
            'bilddateiendung': 'JPEG',  # Dateiendung der Bilder, ohne ".", Achtung: 'jpg' ist nicht 'JPG'
            'fv': 0  # value_to_fill: Wert, mit dem die abgeschnittenen Ränder aufgefüllt werden sollen
        },
        
        'training': {
            # Bildgröße für Training (MUSS mit Evaluation übereinstimmen!)
            # WICHTIG: Sollte an die extrahierten Bilder angepasst werden!
            # Die extrahierten Bilder haben eine längere Seite von pr=400px
            # Daher verwenden wir (400, 400) für bessere Qualität und keine unnötige Reskalierung
            'image_size': (400, 400),  # (Höhe, Breite) in Pixeln - angepasst an extraction['pr']
            
            # Batch-Größe
            'batch_size': 32,
            
            # Validierung
            'validation_split': 0.2,  # 20% der Trainingsdaten für Validierung
            'seed': 123,  # Random Seed für Reproduzierbarkeit
            
            # Training-Parameter
            'epochs': 100,  # Maximale Anzahl Epochen
            'patience': 100,  # Early Stopping: Warte 100 Epochen, bevor gestoppt wird
            'learning_rate': 0.0001,  # Lernrate (kann angepasst werden)
            
            # Transfer Learning
            'fine_tune': True,  # True = alle Layer trainieren (besser), False = nur Top-Layer (schneller)
            'base_model': 'MobileNetV2',  # Basis-Modell für Transfer Learning
            
            # Callbacks
            'restore_best_weights': True,  # Stelle die besten Gewichte wieder her
            'checkpoint_dir': 'checkpoints',  # Verzeichnis für Model-Checkpoints
            'checkpoint_filename': 'best_model.keras',  # Dateiname für das beste Modell
        },
        
        'evaluation': {
            # Bildgröße für Evaluation (MUSS mit Training übereinstimmen!)
            # WICHTIG: Sollte an die extrahierten Bilder angepasst werden!
            # Die extrahierten Bilder haben eine längere Seite von pr=400px
            # Daher verwenden wir (400, 400) für bessere Qualität und keine unnötige Reskalierung
            'image_size': (400, 400),  # (Höhe, Breite) in Pixeln - angepasst an extraction['pr']
            
            # Batch-Größe
            'batch_size': 32,
            
            # Dateiendungen für Test-Bilder
            'suffixes': ['.jpg', '.jpeg', '.JPG', '.JPEG', '.png', '.PNG'],
            
            # Shuffle
            'shuffle': False,  # Test-Daten nicht shuffeln (für Reproduzierbarkeit)
        }
    }


# Standard-Konfiguration für einfachen Import
config = get_config()
