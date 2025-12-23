"""
Schritt 6: Kleine Objekte (Artefakte) ignorieren
Filtert Objekte basierend auf der Mindestanzahl von Pixeln.
Behält nur das größte Objekt und verbindet große Objekte.
"""

import numpy as np
from skimage.measure import label, regionprops
from skimage.morphology import dilation, square


def filter_small_objects(label_img, min_num_pixels=1000, keep_only_largest=True, connect_large_objects=True, large_object_threshold=0.3):
    """
    Filtert kleine Objekte aus der Maske heraus.
    
    Args:
        label_img: Label-Array mit nummerierten Objekten
        min_num_pixels: Mindestanzahl von Pixeln pro Objekt
        keep_only_largest: Wenn True, behält nur das größte Objekt (ignoriert min_num_pixels)
        connect_large_objects: Wenn True, verbindet große Objekte (wenn 2+ große Objekte vorhanden)
        large_object_threshold: Anteil der größten Region, ab dem ein Objekt als "groß" gilt (0.0-1.0)
        
    Returns:
        filtered_label_img: Gefiltertes Label-Array
        filtered_regions: Liste von gefilterten RegionProps-Objekten
    """
    regions = list(regionprops(label_img))
    
    if len(regions) == 0:
        print("Keine Regionen gefunden!")
        return label_img, []
    
    # Sortiere Regionen nach Größe (größte zuerst)
    regions_sorted = sorted(regions, key=lambda r: r.area, reverse=True)
    largest_area = regions_sorted[0].area
    
    print(f"Ursprünglich: {len(regions)} Regionen")
    print(f"Größte Region: {int(largest_area)} Pixel")
    
    # Erstelle eine Maske für große Objekte
    large_objects_mask = np.zeros_like(label_img, dtype=bool)
    
    if keep_only_largest:
        # Behalte nur das größte Objekt
        largest_region = regions_sorted[0]
        large_objects_mask[label_img == largest_region.label] = True
        filtered_regions = [largest_region]
        print(f"Behalte nur das größte Objekt: {int(largest_region.area)} Pixel")
    else:
        # Filtere nach min_num_pixels
        filtered_regions = []
        for region in regions_sorted:
            if region.area >= min_num_pixels:
                large_objects_mask[label_img == region.label] = True
                filtered_regions.append(region)
        print(f"Nach Filterung: {len(filtered_regions)} Regionen (min. {min_num_pixels} Pixel)")
    
    # Verbinde große Objekte, wenn gewünscht
    if connect_large_objects and len(regions_sorted) >= 2:
        # Finde große Objekte (größer als threshold * größte Region)
        large_regions = [r for r in regions_sorted if r.area >= large_object_threshold * largest_area]
        
        if len(large_regions) >= 2:
            print(f"Verbinde {len(large_regions)} große Objekte (jeweils >= {int(large_object_threshold * largest_area)} Pixel)")
            # Erstelle Maske mit allen großen Objekten
            large_objects_mask = np.zeros_like(label_img, dtype=bool)
            for region in large_regions:
                large_objects_mask[label_img == region.label] = True
            
            # Dilatiere die Maske, um nahe Objekte zu verbinden
            large_objects_mask = dilation(large_objects_mask, square(10))  # 10x10 Kernel zum Verbinden
            
            # Erstelle neues Label-Array (verbundene Objekte werden jetzt als ein Objekt erkannt)
            large_objects_mask = label(large_objects_mask)
            # Konvertiere zurück zu bool (alle verbundenen Objekte werden zu einem)
            large_objects_mask = large_objects_mask > 0
            
            # Berechne neue Regionen
            filtered_label_img = label(large_objects_mask)
            filtered_regions = list(regionprops(filtered_label_img))
            print(f"Nach Verbindung: {len(filtered_regions)} Objekt(e)")
        else:
            # Keine Verbindung nötig, erstelle Label-Array
            filtered_label_img = label(large_objects_mask)
    else:
        # Erstelle neues Label-Array nur mit großen Objekten
        filtered_label_img = label(large_objects_mask)
    
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

