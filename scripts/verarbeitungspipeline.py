"""
Haupt-Pipeline: Führt alle Verarbeitungsschritte nacheinander aus.
"""

import numpy as np
from pathlib import Path
import os

# Importiere alle Schritte
from schritt0_bild_laden import choose_an_image
from schritt1_aufloesungsreduktion import resize_image
from schritt2_raender_abschneiden import fill_borders
from schritt3_hintergrundschwellwert import create_mask_from_image
from schritt4_morphologische_transformation import apply_morphology
from schritt5_objekte_extrahieren import extract_objects
from schritt6_kleine_objekte_filtern import filter_small_objects, print_region_sizes


def process_image(im, 
                  max_resolution=(225, 400),
                  fraction_of_rows_to_remove=0.001,
                  fraction_of_cols_to_remove=0.13,
                  value_to_fill=0,
                  hue_threshold=None,
                  saturation_threshold=None,
                  value_threshold=150,
                  erosion_size=3,
                  dilation_size=7,
                  min_num_pixels=1000):
    """
    Führt die komplette Bildverarbeitungspipeline aus.
    
    Args:
        im: PIL Image
        max_resolution: Tuple (max_height, max_width)
        fraction_of_rows_to_remove: Bruchteil der Zeilen zum Entfernen
        fraction_of_cols_to_remove: Bruchteil der Spalten zum Entfernen
        value_to_fill: Wert zum Auffüllen der Ränder
        hue_threshold: Optional, Schwellwert für Hue
        saturation_threshold: Optional, Schwellwert für Saturation
        value_threshold: Schwellwert für Value
        erosion_size: Größe des Erosionskerns
        dilation_size: Größe des Dilatationskerns
        min_num_pixels: Mindestanzahl Pixel pro Objekt
        
    Returns:
        Dictionary mit allen Zwischenergebnissen
    """
    results = {}
    
    # Schritt 1: Auflösungsreduktion
    print("\n=== Schritt 1: Auflösungsreduktion ===")
    im_resized = resize_image(im, max_resolution)
    results['im_resized'] = im_resized
    
    # Schritt 2: Ränder abschneiden
    print("\n=== Schritt 2: Ränder abschneiden ===")
    im_filled = fill_borders(im_resized, value_to_fill, 
                             fraction_of_rows_to_remove, 
                             fraction_of_cols_to_remove)
    results['im_filled'] = im_filled
    
    # Schritt 3: Hintergrundschwellwert
    print("\n=== Schritt 3: Hintergrundschwellwert ===")
    mask, masked_image = create_mask_from_image(
        im_filled,
        hue_threshold=hue_threshold,
        saturation_threshold=saturation_threshold,
        value_threshold=value_threshold
    )
    results['mask'] = mask
    results['masked_image'] = masked_image
    
    # Schritt 4: Morphologische Transformationen
    print("\n=== Schritt 4: Morphologische Transformationen ===")
    morphed_mask = apply_morphology(mask, erosion_size, dilation_size)
    results['morphed_mask'] = morphed_mask
    
    # Schritt 5: Objekte extrahieren
    print("\n=== Schritt 5: Objekte extrahieren ===")
    label_img, regions = extract_objects(morphed_mask, im_filled)
    results['label_img'] = label_img
    results['regions'] = regions
    
    # Schritt 6: Kleine Objekte filtern
    print("\n=== Schritt 6: Kleine Objekte filtern ===")
    filtered_label_img, filtered_regions = filter_small_objects(
        label_img, min_num_pixels
    )
    results['filtered_label_img'] = filtered_label_img
    results['filtered_regions'] = filtered_regions
    
    print_region_sizes(filtered_regions)
    
    return results


if __name__ == "__main__":
    # Test der Pipeline
    base_path = Path(os.getcwd()) / "data"
    dataset_paths = {
        "schraubenschluessel": base_path / "schraubenschluessel",
        "schraubenzieher": base_path / "schraubenezieher",
        "seidenschneider": base_path / "seidenschneider"
    }
    
    im, klasse = choose_an_image(dataset_paths)
    results = process_image(im)
    
    print("\n=== Pipeline erfolgreich abgeschlossen ===")
    print(f"Gefilterte Regionen: {len(results['filtered_regions'])}")

