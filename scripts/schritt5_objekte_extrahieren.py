"""
Schritt 5: Objekte extrahieren
Identifiziert separate Objekte in der Maske und extrahiert sie.
"""

import numpy as np
from skimage.measure import label, regionprops
from scipy.ndimage import binary_fill_holes
from skimage.morphology import closing, square


def extract_objects(mask, im_filled):
    """
    Identifiziert separate Objekte in der Maske und extrahiert sie.
    
    Args:
        mask: Binäres NumPy Array (H, W) - Maske
        im_filled: NumPy Array (H, W, 3) - Originalbild
        
    Returns:
        label_img: Label-Array mit nummerierten Objekten
        regions: Liste von RegionProps-Objekten
    """
    # Fülle Löcher in der Maske
    mask_filled = binary_fill_holes(mask)
    
    # Schließe kleine Lücken mit closing operation (verhindert Aufteilung von Objekten)
    # Closing = Dilatation gefolgt von Erosion - verbindet nahe Objekte
    mask_closed = closing(mask_filled, square(5))  # 5x5 Kernel zum Schließen kleiner Lücken
    
    # Label die verbundenen Komponenten
    label_img = label(mask_closed)
    
    # Berechne Region-Properties mit intensity_image, damit image_intensity verfügbar ist
    # Konvertiere im_filled zu Graustufen, falls nötig, für intensity_image
    if len(im_filled.shape) == 3:
        # RGB zu Graustufen konvertieren
        intensity_image = np.mean(im_filled, axis=2).astype(np.uint8)
    else:
        intensity_image = im_filled
    
    regions = regionprops(label_img, intensity_image=intensity_image)
    
    print(f"Anzahl gefundene Bereiche: {len(regions)}")
    
    return label_img, regions


if __name__ == "__main__":
    # Test
    from schritt4_morphologische_transformation import apply_morphology
    from schritt3_hintergrundschwellwert import create_mask_from_image
    from schritt2_raender_abschneiden import fill_borders
    from schritt1_aufloesungsreduktion import resize_image
    from schritt0_bild_laden import choose_an_image
    from pathlib import Path
    import os
    
    base_path = Path(os.getcwd()) / "data"
    dataset_paths = {
        "schraubenschluessel": base_path / "schraubenschluessel",
        "schraubenzieher": base_path / "schraubenezieher",
        "seidenschneider": base_path / "seidenschneider"
    }
    
    im, _ = choose_an_image(dataset_paths)
    im_resized = resize_image(im, (225, 400))
    im_filled = fill_borders(im_resized, value_to_fill=0, 
                              fraction_of_rows_to_remove=0.001,
                              fraction_of_cols_to_remove=0.13)
    
    mask, _ = create_mask_from_image(im_filled, value_threshold=150)
    morphed_mask = apply_morphology(mask, erosion_size=3, dilation_size=7)
    
    label_img, regions = extract_objects(morphed_mask, im_filled)
    print(f"Label image shape: {label_img.shape}")
    print(f"Anzahl Regionen: {len(regions)}")

