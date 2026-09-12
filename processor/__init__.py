from pathlib import Path

_legacy_path = str(Path(__file__).with_name("legacy"))
if _legacy_path not in __path__:
    __path__.append(_legacy_path)
