"""
Schritt 2: Ränder abschneiden
Schneidet Ränder des Bildes ab und füllt sie mit einem Wert.
"""

import numpy as np
from PIL import Image


def fill_borders(im, value_to_fill, fraction_of_rows_to_remove, fraction_of_cols_to_remove):
    """
    Schneidet Ränder des Bildes ab und füllt sie mit einem Wert.
    
    Args:
        im: PIL Image oder NumPy Array
        value_to_fill: Wert, mit dem die Ränder aufgefüllt werden
        fraction_of_rows_to_remove: Bruchteil der Zeilen, die oben/unten entfernt werden
        fraction_of_cols_to_remove: Bruchteil der Spalten, die links/rechts entfernt werden
        
    Returns:
        NumPy Array (H, W, 3) mit uint8 dtype
    """
    # Wenn es ein PIL Image ist → nach NumPy konvertieren
    if isinstance(im, Image.Image):
        im = np.array(im)

    # Sicherheit: grayscale → RGB erzwingen
    if im.ndim == 2:
        im = np.stack([im, im, im], axis=-1)

    nRows, nCols = im.shape[0], im.shape[1]
    im_cropped = im.copy().astype('uint8')

    # Anzahl Pixel, die oben/unten abgeschnitten werden
    nrow = int(fraction_of_rows_to_remove * nRows)
    # Anzahl Pixel, die links/rechts abgeschnitten werden
    ncol = int(fraction_of_cols_to_remove * nCols)

    # Oben
    im_cropped[:nrow, :, :] = value_to_fill
    # Unten
    im_cropped[nRows - nrow:, :, :] = value_to_fill
    # Links
    im_cropped[:, :ncol, :] = value_to_fill
    # Rechts
    im_cropped[:, nCols - ncol:, :] = value_to_fill

    # Gib NumPy-Array zurück (nicht PIL Image), da die nachfolgenden Funktionen NumPy-Arrays erwarten
    return im_cropped


if __name__ == "__main__":
    # Test
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
    print(f"Filled shape: {im_filled.shape}, dtype: {im_filled.dtype}")

