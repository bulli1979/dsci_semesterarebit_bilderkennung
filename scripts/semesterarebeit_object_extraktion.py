def print_extraction_parameters(config):
    """
    Gibt die Parameter für die Objektextraktion aus dem Config-Objekt formatiert aus.
    
    Parameters:
    -----------
    config : dict
        Config-Objekt mit 'extraction' Bereich, der alle Parameter enthält
    """
    extraction_config = config['extraction']
    
    pr = extraction_config['pr']
    vth = extraction_config['vth']
    mpx = extraction_config['mpx']
    es = extraction_config['es']
    ds = extraction_config['ds']
    
    print("Parameter für Objektextraktion:")
    if isinstance(pr, int):
        print(f"  Auflösung: Längere Seite = {pr}px (proportional)")
    else:
        print(f"  Auflösung: {pr}")
    print(f"  Value Threshold: {vth}")
    print(f"  Minimum Pixel: {mpx}")
    print(f"  Erosion/Dilatation: {es}/{ds} ( Minimale Transformation, da ursprüngliche Maske besser ist!)")


def list_data(data_dir_name: str = "data"):
    """
    Zeigt den Inhalt des data-Verzeichnisses an.
    
    Parameters:
    -----------
    data_dir_name : str
        Name des Datenverzeichnisses (Standard: "data")
    """
    from pathlib import Path
    import os
    
    data_path = Path(os.getcwd()) / data_dir_name
    if data_path.exists():
        print(f"Inhalt von '{data_path}':")
        for item in sorted(data_path.iterdir()):
            print(f"  - {item.name}")
    else:
        print(f"⚠️ Verzeichnis '{data_path}' existiert nicht!")

