"""
Vorbereitungsfunktionen für die Semesterarbeit.
Enthält Funktionen zum Prüfen, Installieren von Modulen und Laden von Scripts.
"""

import sys
import os
import subprocess
import importlib
from typing import List, Optional


def add_scripts_to_path(scripts_dir: Optional[str] = None) -> str:
    """
    Fügt das übergeordnete Verzeichnis zum Python-Pfad hinzu, damit 'from scripts.xxx' funktioniert.
    
    Args:
        scripts_dir: Optional, Pfad zum scripts-Verzeichnis. 
                     Wenn None, wird 'scripts' im aktuellen Verzeichnis verwendet.
    
    Returns:
        str: Der hinzugefügte Pfad (übergeordnetes Verzeichnis)
    """
    if scripts_dir is None:
        scripts_dir = os.path.join(os.getcwd(), 'scripts')
    
    # Füge das übergeordnete Verzeichnis hinzu, nicht scripts selbst
    # Damit funktioniert 'from scripts.xxx import ...'
    parent_dir = os.path.dirname(os.path.abspath(scripts_dir))
    
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)  # Am Anfang einfügen für höhere Priorität
    
    return parent_dir


def check_required_modules(install_missing: bool = False) -> bool:
    """
    Prüft ob alle benötigten Module installiert sind.
    
    Args:
        install_missing: Wenn True, werden fehlende Module automatisch installiert
    
    Returns:
        bool: True wenn alle Module installiert sind, False wenn Installation durchgeführt wurde
    
    Raises:
        ImportError: Wenn Module fehlen und nicht installiert werden konnten
    """
    # Debug-Informationen
    print("Python-Environment-Informationen:")
    print(f"  Python-Executable: {sys.executable}")
    print(f"  Python-Version: {sys.version}")
    print(f"  Python-Pfad: {sys.path[:2]}")
    print()
    
    required_modules = {
        'numpy': 'numpy',
        'matplotlib': 'matplotlib',
        'skimage': 'scikit-image',
        'sklearn': 'scikit-learn',
        'scipy': 'scipy',
        'progressbar': 'progressbar2',  # Paket heißt progressbar2, Modul heißt progressbar
        'PIL': 'Pillow',
        'pandas': 'pandas'
    }
    
    missing_modules = []
    for module_name, package_name in required_modules.items():
        try:
            __import__(module_name)
            print(f"✓ {package_name} ist installiert")
        except ImportError as e:
            missing_modules.append(package_name)
            print(f"✗ {package_name} fehlt! (Import-Fehler: {e})")
    
    if missing_modules:
        print("\n" + "="*60)
        print("FEHLER: Folgende Module fehlen:")
        for mod in missing_modules:
            print(f"  - {mod}")
        
        if install_missing:
            print("\nVersuche fehlende Module automatisch zu installieren...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing_modules)
                print("✓ Installation erfolgreich! Bitte führen Sie diese Zelle erneut aus.")
                return False  # False zurückgeben, damit der Benutzer die Zelle nochmal ausführt
            except Exception as e:
                print(f"✗ Automatische Installation fehlgeschlagen: {e}")
        
        print("\nBitte installieren Sie die fehlenden Module mit:")
        print(f"  python -m pip install {' '.join(missing_modules)}")
        print("\nOder installieren Sie alle Module aus requirements.txt:")
        print("  python -m pip install -r requirements.txt")
        print("\nHinweis: Falls Sie Jupyter verwenden, stellen Sie sicher,")
        print("dass der Kernel das richtige Python-Environment verwendet!")
        print("="*60)
        raise ImportError(f"Fehlende Module: {', '.join(missing_modules)}")
    
    print("\n✓ Alle benötigten Module sind installiert!\n")
    return True


def reload_modules(module_names: List[str]) -> None:
    """
    Lädt angegebene Module neu (falls bereits geladen).
    
    Args:
        module_names: Liste von Modulnamen, die neu geladen werden sollen
    """
    for module_name in module_names:
        if module_name in sys.modules:
            try:
                importlib.reload(sys.modules[module_name])
            except Exception:
                pass  # Ignoriere Fehler beim Reload


def load_extraction_scripts() -> dict:
    """
    Lädt alle Scripts für die Objektextraktion.
    
    Returns:
        dict: Dictionary mit allen importierten Funktionen
    """
    # Liste der Module, die neu geladen werden sollen
    modules_to_reload = [
        'scripts.schritt0_bild_laden',
        'scripts.schritt1_aufloesungsreduktion',
        'scripts.schritt2_raender_abschneiden',
        'scripts.schritt3_hintergrundschwellwert',
        'scripts.schritt4_morphologische_transformation',
        'scripts.schritt5_objekte_extrahieren',
        'scripts.schritt6_kleine_objekte_filtern',
        'object_extraction'
    ]
    
    # Lade Module neu
    reload_modules(modules_to_reload)
    
    # Importiere alle benötigten Funktionen
    from scripts.schritt0_bild_laden import choose_an_image
    from scripts.schritt1_aufloesungsreduktion import resize_image
    from scripts.schritt2_raender_abschneiden import fill_borders
    from scripts.schritt3_hintergrundschwellwert import create_mask_from_image
    from scripts.schritt4_morphologische_transformation import apply_morphology
    from scripts.schritt5_objekte_extrahieren import extract_objects
    from scripts.schritt6_kleine_objekte_filtern import filter_small_objects, print_region_sizes
    
    # Import aus object_extraction.py (für zusätzliche Funktionen)
    # Importiere spezifische Funktionen statt import *
    import object_extraction
    
    # Erstelle Dictionary mit allen Funktionen
    functions = {
        'choose_an_image': choose_an_image,
        'resize_image': resize_image,
        'fill_borders': fill_borders,
        'create_mask_from_image': create_mask_from_image,
        'apply_morphology': apply_morphology,
        'extract_objects': extract_objects,
        'filter_small_objects': filter_small_objects,
        'print_region_sizes': print_region_sizes,
        # Füge object_extraction Modul hinzu für Zugriff auf alle Funktionen
        'object_extraction': object_extraction
    }
    
    print("✓ Alle Scripts wurden erfolgreich geladen!")
    return functions


def load_config() -> dict:
    """
    Lädt die zentrale Konfiguration und extrahiert alle Parameter.
    
    Returns:
        dict: Dictionary mit 'config', 'extraction_config' und allen einzelnen Parametern
    """
    # Verwende den absoluten Pfad basierend auf dem Speicherort dieser Datei
    import os
    # Diese Datei ist in scripts/semesterarbeit_preparation.py
    # Das übergeordnete Verzeichnis ist das Workspace-Verzeichnis
    current_file = os.path.abspath(__file__)
    scripts_dir = os.path.dirname(current_file)  # scripts-Verzeichnis
    parent_dir = os.path.dirname(scripts_dir)   # Workspace-Verzeichnis
    
    # Füge das Workspace-Verzeichnis zum Pfad hinzu
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    
    # Jetzt sollte 'from scripts.semesterarbeit_config import config' funktionieren
    from scripts.semesterarbeit_config import config
    
    # Extrahiere Parameter aus der Config
    extraction_config = config['extraction']
    
    # Erstelle Dictionary mit allen Parametern
    params = {
        'config': config,
        'extraction_config': extraction_config,
        'pr': extraction_config['pr'],
        'fr': extraction_config['fr'],
        'fc': extraction_config['fc'],
        'vth': extraction_config['vth'],
        'es': extraction_config['es'],
        'ds': extraction_config['ds'],
        'mpx': extraction_config['mpx'],
        'saturation_threshold': extraction_config['saturation_threshold'],
        'value_threshold': extraction_config['value_threshold'],
        'bilddateiendung': extraction_config['bilddateiendung'],
        'fv': extraction_config['fv']
    }
    
    return params


def setup_environment(install_missing_modules: bool = True) -> dict:
    """
    Führt alle Vorbereitungsschritte aus:
    1. Fügt übergeordnetes Verzeichnis zum Pfad hinzu (für 'from scripts.xxx')
    2. Prüft und installiert Module
    3. Lädt alle Scripts
    4. Lädt Config
    
    Args:
        install_missing_modules: Wenn True, werden fehlende Module automatisch installiert
    
    Returns:
        dict: Dictionary mit 'functions' (geladene Funktionen) und 'params' (Config-Parameter)
    """
    # Schritt 1: Füge übergeordnetes Verzeichnis zum Pfad hinzu
    # Damit funktioniert 'from scripts.xxx import ...'
    parent_dir = add_scripts_to_path()
    print(f"✓ Pfad hinzugefügt: {parent_dir}")
    
    # Stelle sicher, dass das Verzeichnis wirklich im Pfad ist
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
        print(f"✓ Pfad nochmal hinzugefügt (Sicherheit): {parent_dir}")
    
    # Schritt 2: Prüfe Module
    try:
        check_required_modules(install_missing=install_missing_modules)
    except ImportError:
        print("\nBitte installieren Sie die fehlenden Module manuell und führen Sie diese Zelle erneut aus.")
        raise
    
    # Schritt 3: Lade Scripts
    functions = load_extraction_scripts()
    
    # Schritt 4: Lade Config
    params = load_config()
    
    return {
        'functions': functions,
        'params': params
    }

