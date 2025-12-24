"""
Evaluation-Script für trainierte Modelle
Extrahiert aus semesterarbeit-training.ipynb und semesterarbeit-evaluation.ipynb
"""
import tensorflow as tf
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

try:
    import scikitplot as skplt
    HAS_SCIKIT_PLOT = True
except ImportError:
    HAS_SCIKIT_PLOT = False
    print("⚠ scikit-plot nicht verfügbar, verwende sklearn.metrics + seaborn")


def evaluate_model(
    model: tf.keras.Model,
    val_ds: tf.data.Dataset,
    class_names: list,
    model_name: str = "model"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluiert ein trainiertes Modell auf dem Validierungsdatensatz.
    
    Args:
        model: Trainiertes Keras-Modell
        val_ds: TensorFlow Dataset mit Validierungsdaten
        class_names: Liste der Klassennamen
        
    Returns:
        Tuple (y_true, y_pred, y_pred_proba)
    """
    # True Labels extrahieren
    # Das Dataset kann 2 oder 3 Werte zurückgeben (image, label) oder (image, label, path)
    try:
        # Versuche zuerst mit 2 Werten (image, label)
        y_true = np.asarray(list(val_ds.unbatch().map(lambda x, y: y)))
    except (TypeError, ValueError):
        # Falls das fehlschlägt, hat das Dataset 3 Werte (image, label, path)
        y_true = np.asarray(list(val_ds.unbatch().map(lambda x, y, p: y)))
    
    # Für Vorhersagen müssen wir nur die Bilder verwenden (erster Wert)
    # Erstelle ein Dataset nur mit Bildern für die Vorhersage
    val_ds_images = val_ds.map(lambda *args: args[0])  # Nimm nur das erste Element (Bild)
    
    # Vorhersagen machen
    print(f"\n🔍 Mache Vorhersagen für {len(y_true)} Bilder...")
    y_pred_proba = model.predict(val_ds_images)
    y_pred = y_pred_proba.argmax(axis=1)
    
    print(f"✓ Vorhersagen abgeschlossen")
    print(f"  - Vorhergesagte Klassen: {y_pred}")
    
    return y_true, y_pred, y_pred_proba


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list,
    normalize: bool = True,
    save_path: Optional[Path] = None
):
    """
    Plottet die Confusion Matrix.
    
    Args:
        y_true: True Labels
        y_pred: Vorhergesagte Labels
        class_names: Liste der Klassennamen
        normalize: Wenn True, wird die Matrix normalisiert
        save_path: Optional: Pfad zum Speichern der Grafik
    """
    if HAS_SCIKIT_PLOT:
        # Verwende scikit-plot wenn verfügbar
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        skplt.metrics.plot_confusion_matrix(
            y_true, y_pred, 
            normalize=normalize,
            ax=ax
        )
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
    else:
        # Fallback: sklearn + seaborn
        cm = confusion_matrix(y_true, y_pred)
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='.2f' if normalize else 'd',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax
        )
        ax.set_xlabel('Vorhergesagte Klasse')
        ax.set_ylabel('Tatsächliche Klasse')
        ax.set_title('Confusion Matrix' + (' (normalisiert)' if normalize else ''))
        plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Confusion Matrix gespeichert: {save_path}")
    
    plt.show()


def print_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list
):
    """
    Druckt einen detaillierten Classification Report.
    
    Args:
        y_true: True Labels
        y_pred: Vorhergesagte Labels
        class_names: Liste der Klassennamen
    """
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT (Ergebnisse pro Klasse)")
    print("="*60)
    
    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        output_dict=True
    )
    
    # Detaillierte Ausgabe pro Klasse
    print("\n📊 Ergebnisse pro Klasse:")
    print("-" * 60)
    for i, class_name in enumerate(class_names):
        if str(i) in report:
            metrics = report[str(i)]
            print(f"\n{class_name}:")
            print(f"  Precision: {metrics['precision']:.3f}")
            print(f"  Recall:    {metrics['recall']:.3f}")
            print(f"  F1-Score:  {metrics['f1-score']:.3f}")
            print(f"  Support:   {int(metrics['support'])}")
    
    # Gesamt-Metriken
    print("\n" + "-" * 60)
    print("Gesamt-Metriken:")
    print(f"  Accuracy:  {report['accuracy']:.3f}")
    print(f"  Macro Avg Precision: {report['macro avg']['precision']:.3f}")
    print(f"  Macro Avg Recall:    {report['macro avg']['recall']:.3f}")
    print(f"  Macro Avg F1-Score:  {report['macro avg']['f1-score']:.3f}")
    print("="*60)


def plot_training_history(
    history: tf.keras.callbacks.History,
    save_path: Optional[Path] = None
):
    """
    Plottet den Trainingsverlauf.
    
    Args:
        history: Training History von model.fit()
        save_path: Optional: Pfad zum Speichern der Grafik
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(history.history['accuracy'], label='Training Accuracy')
    ax.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax.plot(history.history['loss'], label='Training Loss')
    ax.plot(history.history['val_loss'], label='Validation Loss')
    
    ax.set_xlabel('Epoche')
    ax.set_ylabel('Metrik')
    ax.set_title('Trainingsverlauf')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Trainingsverlauf gespeichert: {save_path}")
    
    plt.show()


def full_evaluation(
    model: tf.keras.Model,
    val_ds: tf.data.Dataset,
    class_names: list,
    history: Optional[tf.keras.callbacks.History] = None,
    model_name: str = "model",
    output_dir: Optional[Path] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Führt eine vollständige Evaluation durch.
    
    Args:
        model: Trainiertes Keras-Modell
        val_ds: TensorFlow Dataset mit Validierungsdaten
        class_names: Liste der Klassennamen
        history: Optional: Training History
        model_name: Name des Modells
        output_dir: Optional: Verzeichnis für Ausgabe-Dateien
        
    Returns:
        Tuple (y_true, y_pred, y_pred_proba)
    """
    if output_dir is None:
        output_dir = Path('evaluation_results')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Evaluation - muss immer durchgeführt werden, damit wir die Werte zurückgeben können
    y_true, y_pred, y_pred_proba = evaluate_model(model, val_ds, class_names, model_name)
    
    try:
        # Classification Report
        print_classification_report(y_true, y_pred, class_names)
        
        # Confusion Matrix (normalisiert und nicht-normalisiert)
        print("\n📊 Erstelle Confusion Matrix...")
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        
        if HAS_SCIKIT_PLOT:
            skplt.metrics.plot_confusion_matrix(
                y_true, y_pred, normalize=True, ax=axes[0]
            )
            axes[0].tick_params(axis='x', rotation=45)
            axes[0].set_title('Confusion Matrix (normalisiert)')
            
            skplt.metrics.plot_confusion_matrix(
                y_true, y_pred, normalize=False, ax=axes[1]
            )
            axes[1].tick_params(axis='x', rotation=45)
            axes[1].set_title('Confusion Matrix (absolut)')
        else:
            # Fallback
            cm_norm = confusion_matrix(y_true, y_pred)
            cm_norm = cm_norm.astype('float') / cm_norm.sum(axis=1)[:, np.newaxis]
            
            sns.heatmap(
                cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[0]
            )
            axes[0].set_title('Confusion Matrix (normalisiert)')
            axes[0].set_xlabel('Vorhergesagte Klasse')
            axes[0].set_ylabel('Tatsächliche Klasse')
            
            cm = confusion_matrix(y_true, y_pred)
            sns.heatmap(
                cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[1]
            )
            axes[1].set_title('Confusion Matrix (absolut)')
            axes[1].set_xlabel('Vorhergesagte Klasse')
            axes[1].set_ylabel('Tatsächliche Klasse')
        
        plt.tight_layout()
        confusion_matrix_path = output_dir / f'{model_name}_confusion_matrix.png'
        plt.savefig(confusion_matrix_path, dpi=150, bbox_inches='tight')
        print(f"✓ Confusion Matrix gespeichert: {confusion_matrix_path}")
        plt.show()
        
        # Training History (wenn verfügbar)
        if history is not None:
            print("\n📈 Erstelle Trainingsverlauf...")
            try:
                plot_training_history(history, output_dir / f'{model_name}_training_history.png')
            except Exception as e:
                print(f"⚠ Warnung: Fehler beim Plotten der Trainingshistorie: {e}")
        
        print(f"\n✓ Evaluation abgeschlossen!")
        print(f"  - Ergebnisse gespeichert in: {output_dir}")
        
    except Exception as e:
        print(f"⚠ Fehler während der Visualisierung: {e}")
        print("  Die Vorhersagen wurden trotzdem erstellt und werden zurückgegeben.")
    
    # Stelle sicher, dass die Werte zurückgegeben werden
    return y_true, y_pred, y_pred_proba


def create_comparison_table(
    models_results: dict,
    class_names: list
) -> None:
    """
    Erstellt eine Vergleichstabelle für mehrere Modelle.
    
    Args:
        models_results: Dictionary mit {model_name: (y_true, y_pred, y_pred_proba)}
        class_names: Liste der Klassennamen
    """
    from sklearn.metrics import classification_report
    import pandas as pd
    
    print("\n" + "="*80)
    print("VERGLEICHSÜBERSICHT ALLER MODELLE")
    print("="*80)
    
    # Sammle alle Metriken
    comparison_data = []
    
    for model_name, (y_true, y_pred, y_pred_proba) in models_results.items():
        if y_true is None or y_pred is None:
            continue
            
        report = classification_report(
            y_true, y_pred,
            target_names=class_names,
            output_dict=True
        )
        
        # Gesamt-Metriken
        comparison_data.append({
            'Modell': model_name,
            'Accuracy': report['accuracy'],
            'Macro Avg Precision': report['macro avg']['precision'],
            'Macro Avg Recall': report['macro avg']['recall'],
            'Macro Avg F1-Score': report['macro avg']['f1-score'],
            'Weighted Avg Precision': report['weighted avg']['precision'],
            'Weighted Avg Recall': report['weighted avg']['recall'],
            'Weighted Avg F1-Score': report['weighted avg']['f1-score']
        })
    
    # Erstelle DataFrame und zeige Tabelle
    df = pd.DataFrame(comparison_data)
    df = df.sort_values('Accuracy', ascending=False)
    
    print("\n📊 GESAMT-METRIKEN (Vergleich aller Modelle):")
    print("-" * 80)
    print(df.to_string(index=False, float_format='%.4f'))
    
    # Detaillierte Metriken pro Klasse für jedes Modell
    print("\n" + "="*80)
    print("DETAILLIERTE METRIKEN PRO KLASSE")
    print("="*80)
    
    for model_name, (y_true, y_pred, y_pred_proba) in models_results.items():
        if y_true is None or y_pred is None:
            continue
            
        report = classification_report(
            y_true, y_pred,
            target_names=class_names,
            output_dict=True
        )
        
        print(f"\n{model_name}:")
        print("-" * 80)
        
        # Erstelle Tabelle für dieses Modell
        class_data = []
        for i, class_name in enumerate(class_names):
            if str(i) in report:
                metrics = report[str(i)]
                class_data.append({
                    'Klasse': class_name,
                    'Precision': metrics['precision'],
                    'Recall': metrics['recall'],
                    'F1-Score': metrics['f1-score'],
                    'Support': int(metrics['support'])
                })
        
        class_df = pd.DataFrame(class_data)
        print(class_df.to_string(index=False, float_format='%.4f'))
        
        # Gesamt-Metriken für dieses Modell
        print(f"\n  Gesamt-Accuracy: {report['accuracy']:.4f}")
        print(f"  Macro Avg F1-Score: {report['macro avg']['f1-score']:.4f}")
        print(f"  Weighted Avg F1-Score: {report['weighted avg']['f1-score']:.4f}")
    
    print("\n" + "="*80)
    
    return df


def check_class_order_consistency(
    training_class_names: list,
    evaluation_class_names: list
) -> bool:
    """
    Prüft ob die Klassenreihenfolge zwischen Training und Evaluation übereinstimmt.
    
    Args:
        training_class_names: Klassenliste aus dem Training-Notebook
        evaluation_class_names: Klassenliste aus dem Evaluation-Notebook
        
    Returns:
        True wenn identisch, False sonst
    """
    if len(training_class_names) != len(evaluation_class_names):
        return False
    
    return training_class_names == evaluation_class_names


def evaluate_all_models(
    models_dict: dict,
    test_ds: tf.data.Dataset,
    class_names: list,
    training_class_names: list = None
) -> dict:
    """
    Evaluiert alle Modelle in einem Dictionary.
    
    Args:
        models_dict: Dictionary mit {model_name: model_object}
        test_ds: Test-Dataset
        class_names: Klassenliste für Evaluation
        training_class_names: Klassenliste aus Training (für Prüfung)
        
    Returns:
        Dictionary mit Evaluationsergebnissen {model_name: (y_true, y_pred, y_pred_proba)}
    """
    results = {}
    
    # Prüfe Klassenreihenfolge
    if training_class_names is not None:
        print("="*70)
        print("PRÜFUNG 1: KLASSENREIHENFOLGE")
        print("="*70)
        is_consistent = check_class_order_consistency(training_class_names, class_names)
        if is_consistent:
            print("✓ Klassenreihenfolge stimmt überein!")
            print(f"  Training: {training_class_names}")
            print(f"  Evaluation: {class_names}")
        else:
            print("❌ KLASSENREIHENFOLGE STIMMT NICHT ÜBEREIN!")
            print(f"  Training: {training_class_names}")
            print(f"  Evaluation: {class_names}")
            print("\n⚠ WARNUNG: Dies führt zu falschen Vorhersagen!")
            print("  → Bitte korrigieren Sie die Reihenfolge im Evaluation-Notebook")
        print("="*70)
    
    # Evaluiere jedes Modell
    for model_name, model in models_dict.items():
        print(f"\n{'='*70}")
        print(f"EVALUATION: {model_name}")
        print("="*70)
        
        try:
            y_true, y_pred, y_pred_proba = full_evaluation(
                model=model,
                val_ds=test_ds,
                class_names=class_names,
                history=None,
                model_name=model_name.replace(" ", "_").lower()
            )
            results[model_name] = (y_true, y_pred, y_pred_proba)
            print(f"\n✓ {model_name} erfolgreich evaluiert!")
        except Exception as e:
            print(f"⚠ Fehler bei Evaluation von {model_name}: {e}")
            results[model_name] = (None, None, None)
    
    return results


if __name__ == "__main__":
    # Beispiel-Verwendung
    print("Dieses Script sollte aus dem Notebook heraus aufgerufen werden.")
    print("Siehe semesterarbeit-training.ipynb für die Verwendung.")

