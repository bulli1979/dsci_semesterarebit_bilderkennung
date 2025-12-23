"""
Schritt 3: Hintergrundschwellwert bestimmen
Erstellt eine Maske basierend auf HSV-Schwellwerten.
"""

import numpy as np
from matplotlib.colors import rgb_to_hsv
import sys
import os

# Füge das scripts-Verzeichnis zum Python-Pfad hinzu
scripts_path = os.path.join(os.path.dirname(__file__), '..', 'scripts')
if scripts_path not in sys.path:
    sys.path.append(scripts_path)

from object_extraction import generate_mask_with_hsv_threshold, create_masked_image


def create_mask_from_image(image, hue_threshold=None, saturation_threshold=None, 
                           value_threshold=None):
    """
    Erstellt eine Maske aus einem Bild basierend auf HSV-Schwellwerten.
    
    Args:
        im_filled: NumPy Array (H, W, 3) mit uint8 dtype
        hue_threshold: Optional, Schwellwert für Hue-Kanal
        saturation_threshold: Optional, Schwellwert für Saturation-Kanal
        value_threshold: Optional, Schwellwert für Value-Kanal
        
    Returns:
        mask: Binäres NumPy Array (H, W)
        masked_image: NumPy Array mit maskiertem Bild
    """
    # Stelle sicher, dass im_filled ein NumPy-Array ist
    if not isinstance(image, np.ndarray):
        image = np.array(image)
    
    # Stelle sicher, dass das Array die richtige Form hat (H, W, 3) für RGB
    if image.ndim == 2:
        # Graustufenbild → RGB konvertieren
        image = np.stack([image, image, image], axis=-1)
    elif image.ndim == 3 and image.shape[2] == 4:
        # RGBA → RGB konvertieren
        image = image[:, :, :3]
    elif image.ndim == 3 and image.shape[2] != 3:
        raise ValueError(f"Unerwartete Array-Form: {image.shape}. Erwartet (H, W, 3) für RGB.")
    
    # Stelle sicher, dass der Wertebereich 0-255 ist
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
    
    # Stelle sicher, dass das Array kopierbar ist (für rgb_to_hsv)
    # Konvertiere zu einem kontinuierlichen Array, falls nötig
    if not image.flags['C_CONTIGUOUS']:
        image = np.ascontiguousarray(image)
    
    # Explizite Kopie erstellen, um sicherzustellen, dass es kopierbar ist
    image = image.copy()
    
    print(f"im_filled Shape: {image.shape}, dtype: {image.dtype}")
    
    # Erstelle Maske
    mask = generate_mask_with_hsv_threshold(
        image, 
        hue_threshold=hue_threshold,
        saturation_threshold=saturation_threshold,
        value_threshold=value_threshold
    )
    
    # Erstelle maskiertes Bild
    masked_image = create_masked_image(image, mask)
    
    return mask, masked_image


if __name__ == "__main__":
    # Test
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
    
    mask, masked_image = create_mask_from_image(im_filled, value_threshold=150)
    print(f"Mask shape: {mask.shape}, dtype: {mask.dtype}")
    print(f"Masked image shape: {masked_image.shape}")

