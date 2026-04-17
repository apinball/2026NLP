from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from src.config import ONTOLOGY_PATH


@dataclass
class TechEntry:
    name: str
    released_year: int
    category: str
    versions: dict[str, int] = field(default_factory=dict)
    aliases: list[str] = field(default_factory=list)

    def version_year(self, version: str) -> int | None:
        if version in self.versions:
            return self.versions[version]
        major = version.split(".")[0]
        return self.versions.get(major)


class OntologyDB:
    """기술 이름/별칭으로 TechEntry 를 조회하는 메모리 내 인덱스."""

    def __init__(self, path: Path = ONTOLOGY_PATH) -> None:
        self.path = path
        self._by_key: dict[str, TechEntry] = {}
        self._entries: list[TechEntry] = []
        self.reload()

    def reload(self) -> None:
        self._by_key.clear()
        self._entries.clear()
        if not self.path.exists():
            return
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        for item in raw:
            entry = TechEntry(
                name=item["name"],
                released_year=int(item["released_year"]),
                category=item.get("category", ""),
                versions={k: int(v) for k, v in item.get("versions", {}).items()},
                aliases=list(item.get("aliases", [])),
            )
            self._entries.append(entry)
            self._by_key[entry.name.lower()] = entry
            for alias in entry.aliases:
                self._by_key[alias.lower()] = entry

    def lookup(self, name: str) -> TechEntry | None:
        return self._by_key.get(name.lower())

    def all_entries(self) -> list[TechEntry]:
        return list(self._entries)
