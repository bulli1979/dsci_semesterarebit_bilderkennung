"""
Funktionen zum Laden von trainierten Modellen
"""
import tensorflow as tf
from pathlib import Path
from typing import Tuple, List, Dict, Optional


def load_all_models(
    datensatzname: str = 'Werkzeuge',
    model_output_path: Optional[Path] = None,
    expected_class_names: Optional[List[str]] = None
) -> Tuple[Dict[str, tf.keras.Model], List[str]]:
    """
    Lädt alle verfügbaren trainierten Modelle.
    
    Args:
        datensatzname: Name des Datensatzes (z.B. 'Werkzeuge')
        model_output_path: Pfad zum Modell-Ordner (default: 'model_output/model')
        expected_class_names: Erwartete Klassenreihenfolge aus Training (optional)
        
    Returns:
        Tuple (models_dict, expected_class_names)
        - models_dict: Dictionary mit {model_name: model_object}
        - expected_class_names: Liste der erwarteten Klassennamen
    """
    if model_output_path is None:
        model_output_path = Path('model_output') / 'model'
    
    # Standard erwartete Klassenreihenfolge (falls nicht angegeben)
    if expected_class_names is None:
        expected_class_names = ['schraubenschluessel', 'schraubenzieher', 'seidenschneider']
    
    # Definiere Modell-Pfade
    model1_path = model_output_path / f'model_{datensatzname}_simple_cnn.keras'
    model2_path = model_output_path / f'model_{datensatzname}_transfer_learning.keras'
    model3_path = model_output_path / f'model_{datensatzname}.keras'
    
    print("="*70)
    print("LADE ALLE VERFÜGBAREN MODELLE")
    print("="*70)
    
    # Prüfe welche Modelle existieren
    models_to_load = []
    
    if model1_path.exists():
        models_to_load.append(('Modell 1 (Simple CNN)', model1_path, 'loaded_model1'))
    else:
        print(f"⚠ Modell 1 nicht gefunden: {model1_path}")
    
    if model2_path.exists():
        models_to_load.append(('Modell 2 (Transfer Learning)', model2_path, 'loaded_model2'))
    else:
        print(f"⚠ Modell 2 nicht gefunden: {model2_path}")
    
    if model3_path.exists():
        models_to_load.append(('Modell 3 (Standard)', model3_path, 'loaded_model3'))
    else:
        print(f"⚠ Modell 3 nicht gefunden: {model3_path}")
    
    if len(models_to_load) == 0:
        raise FileNotFoundError("Keine Modelle gefunden! Bitte führen Sie zuerst das Training-Notebook aus.")
    
    # Lade alle verfügbaren Modelle
    models_dict = {}
    for model_name, model_path, var_name in models_to_load:
        print(f"\n📦 Lade {model_name} von: {model_path.absolute()}")
        model = tf.keras.models.load_model(str(model_path))
        models_dict[model_name] = model
        print(f"✓ {model_name} erfolgreich geladen")
    
    print("\n" + "="*70)
    print(f"✓ {len(models_to_load)} MODELL(E) ERFOLGREICH GELADEN")
    print("="*70)
    print(f"\n📁 Modell-Ordner: {model_output_path.absolute()}")
    for model_name, model_path, var_name in models_to_load:
        print(f"  - {model_name}: {model_path.name}")
    
    # Zeige erwartete Klassenreihenfolge
    print("\n" + "="*70)
    print("ERWARTETE KLASSENREIHENFOLGE AUS TRAINING")
    print("="*70)
    print("\n📋 Erwartete Reihenfolge (aus Training-Notebook Zelle 10):")
    for i, class_name in enumerate(expected_class_names):
        print(f"  Index {i}: {class_name}")
    print(f"\n⚠ WICHTIG: Diese Reihenfolge wird in der Evaluation geprüft!")
    print("="*70)
    
    return models_dict, expected_class_names





