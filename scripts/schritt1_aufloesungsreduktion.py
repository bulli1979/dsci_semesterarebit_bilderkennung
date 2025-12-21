"""
Schritt 1: Auflösungsreduktion
Reduziert die Bildauflösung unter Beibehaltung der Proportionen.
"""

import numpy as np
from PIL import Image


def resize_image(im, max_resolution):
    """
    Reduziert die Bildauflösung unter Beibehaltung der Proportionen.
    
    Args:
        im: PIL Image
        max_resolution: Tuple (max_height, max_width) - maximale Zielgröße
        
    Returns:
        PIL Image (resized)
    """
    # Maximale Zielgröße (Höhe, Breite)
    max_height, max_width = max_resolution
    
    # Aktuelle Bildgröße ermitteln
    original_width, original_height = im.size
    print(f"Ursprüngliche Bildgröße: {original_width} x {original_height}")
    
    # Berechne Skalierungsfaktor, um Proportionen beizubehalten
    # Wir skalieren so, dass das Bild in die max_width x max_height Box passt
    scale_width = max_width / original_width
    scale_height = max_height / original_height
    scale = min(scale_width, scale_height)  # Nimm den kleineren Faktor
    
    # Berechne neue Größe mit beibehaltenen Proportionen
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)
    
    print(f"Neue Bildgröße (proportional): {new_width} x {new_height}")
    print(f"Skalierungsfaktor: {scale:.3f}")
    
    # Resize mit beibehaltenen Proportionen
    im_resized = im.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    return im_resized


if __name__ == "__main__":
    # Test
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
    print(f"Resized: {im_resized.size}")

