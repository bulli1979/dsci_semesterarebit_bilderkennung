"""
Schritt 4: Morphologische Transformationen
Wendet Erosion und Dilatation auf die Maske an.
"""

import sys
import os

# Füge das scripts-Verzeichnis zum Python-Pfad hinzu
scripts_path = os.path.join(os.path.dirname(__file__), '..', 'scripts')
if scripts_path not in sys.path:
    sys.path.append(scripts_path)

from object_extraction import morphology_transform, square


def apply_morphology(mask, erosion_size=5, dilation_size=50, shape=square):
    """
    Wendet morphologische Transformationen (Erosion und Dilatation) auf die Maske an.
    
    Args:
        mask: Binäres NumPy Array (H, W)
        erosion_size: Größe des Erosionskerns
        dilation_size: Größe des Dilatationskerns
        shape: Form des morphologischen Kerns (Standard: square)
        
    Returns:
        morphed_mask: Transformierte Maske
    """
    morphed_mask = morphology_transform(
        mask,
        shape=shape,
        erosion_size=erosion_size,
        dilation_size=dilation_size
    )
    
    return morphed_mask


if __name__ == "__main__":
    # Test
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
    print(f"Morphed mask shape: {morphed_mask.shape}")

