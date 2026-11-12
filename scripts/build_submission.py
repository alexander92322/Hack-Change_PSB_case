"""Собирает ZIP для платформы: все файлы в корне архива, без вложенных папок.

В архив попадают:
  - корневые .py-модули ядра (solution.py, config.py, assistant.py и др.);
  - requirements-submission.txt под именем requirements.txt;
  - файлы готового индекса из папки index/ (кладутся в корень архива).
Запуск:  python scripts/build_submission.py   ->  dist/submission.zip
"""

from __future__ import annotations

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "dist" / "submission.zip"
MAX_MB = 128


def main() -> None:
    OUT.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for py in sorted(ROOT.glob("*.py")):
            z.write(py, py.name)
        z.write(ROOT / "requirements-submission.txt", "requirements.txt")
        registry = ROOT / "document_registry.yaml"
        if registry.is_file():
            z.write(registry, registry.name)
        index_dir = ROOT / "index"
        if index_dir.is_dir():
            for f in sorted(index_dir.iterdir()):
                if f.is_file() and f.name != "README.md":
                    z.write(f, f.name)
    size_mb = OUT.stat().st_size / 1024 / 1024
    with zipfile.ZipFile(OUT) as z:
        names = z.namelist()
    print(f"{OUT} - {size_mb:.1f} МБ, файлов: {len(names)}")
    print("\n".join("  " + n for n in names))
    if size_mb > MAX_MB:
        raise SystemExit(f"Архив больше {MAX_MB} МБ - платформа его не примет")
    if "solution.py" not in names:
        raise SystemExit("В архиве нет solution.py")


if __name__ == "__main__":
    main()
