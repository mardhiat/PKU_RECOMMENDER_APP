"""List ingredient names from the cuisine tables.

This does not invent phenylalanine, protein, or energy values. Unmatched
ingredients stay in the nutrition tables as missing data, not as zeros that
mean "safe."
"""

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pku.paths import paths


def main():
    names = set()
    for path in sorted(paths.cuisines.glob("*.csv")):
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                ingredient = (row.get("Ingredient") or "").strip()
                if ingredient:
                    names.add(ingredient)
    destination = paths.root / "data" / "processed" / "ingredient_names.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ingredient"])
        for name in sorted(names):
            writer.writerow([name])
    print(f"Wrote {len(names)} ingredient names to {destination}")


if __name__ == "__main__":
    main()
