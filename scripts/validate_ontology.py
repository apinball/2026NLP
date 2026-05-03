"""ontology_seed.json 의 스키마·중복·연도 sanity 검증.

스키마 위반·이름 중복·alias 중복·이상한 연도(미래·1900 이전)를 잡는다.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import date

from src.config import ONTOLOGY_PATH

REQUIRED_FIELDS = {"name", "released_year", "category", "versions", "aliases"}


def main() -> int:
    raw = json.loads(ONTOLOGY_PATH.read_text(encoding="utf-8"))
    errors: list[str] = []
    name_count: Counter[str] = Counter()
    alias_owner: dict[str, str] = {}
    this_year = date.today().year

    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            errors.append(f"#{i}: not a dict")
            continue
        missing = REQUIRED_FIELDS - set(item.keys())
        if missing:
            errors.append(f"#{i} ({item.get('name','?')}): missing fields {missing}")
            continue

        name = item["name"]
        if not isinstance(name, str) or not name.strip():
            errors.append(f"#{i}: empty name")
            continue
        name_count[name.lower()] += 1

        year = item.get("released_year")
        if not isinstance(year, int) or year < 1900 or year > this_year:
            errors.append(f"{name}: bad released_year {year}")

        if not isinstance(item.get("versions"), dict):
            errors.append(f"{name}: versions must be dict")
        else:
            for v, vy in item["versions"].items():
                if not isinstance(vy, int) or vy < 1900 or vy > this_year:
                    errors.append(f"{name} v{v}: bad version year {vy}")

        if not isinstance(item.get("category"), str) or not item["category"]:
            errors.append(f"{name}: empty category")

        aliases = item.get("aliases", [])
        if not isinstance(aliases, list):
            errors.append(f"{name}: aliases must be list")
            continue
        for a in aliases:
            if not isinstance(a, str) or not a.strip():
                errors.append(f"{name}: bad alias {a!r}")
                continue
            key = a.lower()
            if key == name.lower():
                continue
            if key in alias_owner and alias_owner[key] != name:
                errors.append(
                    f"alias {a!r} claimed by both {alias_owner[key]!r} and {name!r}"
                )
            alias_owner[key] = name

    duplicate_names = [n for n, c in name_count.items() if c > 1]
    for n in duplicate_names:
        errors.append(f"duplicate name (case-insensitive): {n!r} x{name_count[n]}")

    print(f"entries: {len(raw)}")
    print(f"unique names: {len(name_count)}")
    cats = Counter(item.get("category", "") for item in raw if isinstance(item, dict))
    print("categories:")
    for c, n in cats.most_common():
        print(f"  {c:20s} {n}")

    if errors:
        print(f"\nERRORS ({len(errors)}):")
        for e in errors:
            print(f"  - {e}")
        return 1

    print("\nOK")
    return 0


if __name__ == "__main__":
    sys.exit(main())