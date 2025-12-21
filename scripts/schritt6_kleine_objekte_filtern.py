"""
Schritt 6: Kleine Objekte (Artefakte) ignorieren
Filtert Objekte basierend auf der Mindestanzahl von Pixeln.
"""

import numpy as np
from skimage.measure import label, regionprops


def filter_small_objects(label_img, min_num_pixels=1000):
    """
    Filtert kleine Objekte aus der Maske heraus.
    
    Args:
        label_img: Label-Array mit nummerierten Objekten
        min_num_pixels: Mindestanzahl von Pixeln pro Objekt
        
    Returns:
        filtered_label_img: Gefiltertes Label-Array
        filtered_regions: Liste von gefilterten RegionProps-Objekten
    """
    regions = regionprops(label_img)
    
    # Erstelle eine Maske für große Objekte
    large_objects_mask = np.zeros_like(label_img, dtype=bool)
    
    filtered_regions = []
    for region in regions:
        if region.area >= min_num_pixels:
            large_objects_mask[label_img == region.label] = True
            filtered_regions.append(region)
    
    # Erstelle neues Label-Array nur mit großen Objekten
    filtered_label_img = label(large_objects_mask)
    
    print(f"Ursprünglich: {len(regions)} Regionen")
    print(f"Nach Filterung: {len(filtered_regions)} Regionen (min. {min_num_pixels} Pixel)")
    
    return filtered_label_img, filtered_regions


def print_region_sizes(regions, max_display=10):
    """
    Gibt die Größen der Regionen aus.
    
    Args:
        regions: Liste von RegionProps-Objekten
        max_display: Maximale Anzahl anzuzeigender Regionen
    """
    if len(regions) < max_display:
        for ireg, reg in enumerate(regions):
            print(f"Region {ireg}: {int(reg.area):>6} Pixel")
    else:
        print(f"Anzahl Regionen: {len(regions)}")
        areas = [reg.area for reg in regions]
        print(f"Min: {int(min(areas))}, Max: {int(max(areas))}, Durchschnitt: {int(np.mean(areas))}")


if __name__ == "__main__":
    # Test
    from schritt5_objekte_extrahieren import extract_objects
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
    print_region_sizes(regions)
    
    filtered_label_img, filtered_regions = filter_small_objects(label_img, min_num_pixels=1000)
    print_region_sizes(filtered_regions)

