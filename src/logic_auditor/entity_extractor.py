from __future__ import annotations

import json
import re
from pathlib import Path

from src.config import DATA_DIR
from src.logic_auditor.models import DetectedEntity, TextSpan
from src.logic_auditor.ontology import OntologyDB, TechEntry


FEATURE_RULES_PATH = DATA_DIR / "logic_feature_rules.json"
CLAIM_SIGNAL_RULES_PATH = DATA_DIR / "logic_claim_signals.json"
YEAR_RE = re.compile(r"(?<!\d)(19[8-9]\d|20[0-2]\d)(?=년|[\s.\-/,)\]]|$)")
DURATION_RE = re.compile(r"(\d+)\s*(년|개월|달|주|일)")
VERSION_RE = re.compile(
    r"(?<![a-zA-Z0-9.])(?:v(?:ersion)?\s*)?(\d+(?:\.\d+){0,2})(?![a-zA-Z0-9.])",
    re.IGNORECASE,
)

_KOREAN_CHAR = r"ㄱ-㆏가-힯"
_KOREAN_PARTICLES = "은는이가을를와과의도만에로으야서"


class EntityExtractor:
    def __init__(
        self,
        ontology: OntologyDB | None = None,
        feature_rules_path: Path = FEATURE_RULES_PATH,
        signal_rules_path: Path = CLAIM_SIGNAL_RULES_PATH,
    ) -> None:
        self.ontology = ontology or OntologyDB()
        self.feature_rules_path = feature_rules_path
        self.signal_rules_path = signal_rules_path
        self.feature_rules = _load_feature_rules(feature_rules_path)
        self.signal_rules = _load_signal_rules(signal_rules_path)

    def extract(self, text: str, base_offset: int = 0) -> tuple[DetectedEntity, ...]:
        entities: list[DetectedEntity] = []
        occupied: list[range] = []

        for entry, alias in self._iter_aliases_longest_first():
            for match in _name_pattern(alias).finditer(text):
                if _overlaps(match.start(), match.end(), occupied):
                    continue
                occupied.append(range(match.start(), match.end()))
                entities.append(_tech_entity(match, text, base_offset, entry))

        entities.extend(_year_entities(text, base_offset))
        entities.extend(_duration_entities(text, base_offset))
        entities.extend(
            _keyword_entities(
                text,
                base_offset,
                self.signal_rules.get("role_terms", {}),
                "ROLE",
            )
        )
        entities.extend(
            _keyword_entities(
                text,
                base_offset,
                self.signal_rules.get("scale_terms", {}),
                "SCALE",
            )
        )
        entities.extend(_feature_entities(text, base_offset, self.feature_rules))
        entities.extend(_version_entities(text, base_offset, entities))
        return tuple(sorted(entities, key=lambda item: (item.span.start, item.span.end)))

    def _iter_aliases_longest_first(self) -> list[tuple[TechEntry, str]]:
        pairs: list[tuple[TechEntry, str]] = []
        for entry in self.ontology.all_entries():
            pairs.append((entry, entry.name))
            pairs.extend((entry, alias) for alias in entry.aliases)
        return sorted(pairs, key=lambda pair: len(pair[1]), reverse=True)


def _load_feature_rules(path: Path) -> list[dict]:
    if not path.exists():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"feature rules must be a list: {path}")
    rules: list[dict] = []
    for index, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"feature rule #{index} must be an object: {path}")
        rule = dict(item)
        for key in ("feature", "tech", "introduced_year", "introduced_version"):
            if not str(rule.get(key) or "").strip():
                raise ValueError(f"feature rule #{index} missing {key}: {path}")
        try:
            rule["introduced_year"] = int(rule["introduced_year"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"feature rule #{index} has invalid introduced_year: {path}"
            ) from exc
        keywords = rule.get("keywords")
        if not isinstance(keywords, list) or not any(str(k).strip() for k in keywords):
            raise ValueError(f"feature rule #{index} requires keywords: {path}")
        rule["keywords"] = [str(k).strip() for k in keywords if str(k).strip()]
        rules.append(rule)
    return rules


def _load_signal_rules(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {"role_terms": {}, "scale_terms": {}}
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"claim signal rules must be an object: {path}")
    role_terms = raw.get("role_terms") or {}
    scale_terms = raw.get("scale_terms") or {}
    if not isinstance(role_terms, dict) or not isinstance(scale_terms, dict):
        raise ValueError(f"claim signal terms must be objects: {path}")
    return {
        "role_terms": _clean_signal_terms(role_terms),
        "scale_terms": _clean_signal_terms(scale_terms),
    }


def _clean_signal_terms(raw: dict) -> dict[str, str]:
    return {
        str(keyword).strip(): str(normalized).strip()
        for keyword, normalized in raw.items()
        if str(keyword).strip() and str(normalized).strip()
    }


def _name_pattern(name: str) -> re.Pattern[str]:
    escaped = re.escape(name)
    if all(c.isascii() for c in name):
        return re.compile(
            rf"(?<![a-zA-Z0-9.+#-]){escaped}(?![a-zA-Z0-9.+#-])",
            re.IGNORECASE,
        )
    pattern = (
        rf"(?<![{_KOREAN_CHAR}a-zA-Z0-9])"
        rf"{escaped}"
        rf"(?:[{_KOREAN_PARTICLES}]{{1,2}}(?![{_KOREAN_CHAR}])|(?![{_KOREAN_CHAR}]))"
    )
    return re.compile(pattern, re.IGNORECASE)


def _keyword_pattern(keyword: str) -> re.Pattern[str]:
    if re.search(r"[가-힣]", keyword):
        return _name_pattern(keyword)
    return re.compile(
        rf"(?<![a-zA-Z0-9.+#-]){re.escape(keyword)}(?![a-zA-Z0-9.+#-])",
        re.IGNORECASE,
    )


def _overlaps(start: int, end: int, occupied: list[range]) -> bool:
    return any(start < item.stop and end > item.start for item in occupied)


def _tech_entity(
    match: re.Match[str],
    text: str,
    base_offset: int,
    entry: TechEntry,
) -> DetectedEntity:
    return DetectedEntity(
        entity_type="TECH",
        text=text[match.start() : match.end()],
        normalized=entry.name,
        span=TextSpan(base_offset + match.start(), base_offset + match.end()),
        metadata={
            "release_year": entry.released_year,
            "category": entry.category,
            "versions": dict(entry.versions),
        },
    )


def _year_entities(text: str, base_offset: int) -> list[DetectedEntity]:
    return [
        DetectedEntity(
            "YEAR",
            match.group(0),
            match.group(1),
            TextSpan(base_offset + match.start(), base_offset + match.end()),
        )
        for match in YEAR_RE.finditer(text)
    ]


def _duration_entities(text: str, base_offset: int) -> list[DetectedEntity]:
    entities: list[DetectedEntity] = []
    for match in DURATION_RE.finditer(text):
        amount = int(match.group(1))
        unit = match.group(2)
        months = amount * 12 if unit == "년" else amount
        if unit == "주":
            months = max(1, round(amount / 4))
        elif unit == "일":
            months = 1
        entities.append(
            DetectedEntity(
                "DURATION",
                match.group(0),
                match.group(0),
                TextSpan(base_offset + match.start(), base_offset + match.end()),
                {"amount": amount, "unit": unit, "months": months},
            )
        )
    return entities


def _keyword_entities(
    text: str,
    base_offset: int,
    keywords: dict[str, str],
    entity_type: str,
) -> list[DetectedEntity]:
    entities: list[DetectedEntity] = []
    lowered = text.lower()
    for keyword, normalized in keywords.items():
        pattern = _keyword_pattern(keyword)
        for match in pattern.finditer(lowered):
            entities.append(
                DetectedEntity(
                    entity_type,
                    text[match.start() : match.end()],
                    normalized,
                    TextSpan(base_offset + match.start(), base_offset + match.end()),
                )
            )
    return entities


def _feature_entities(
    text: str,
    base_offset: int,
    feature_rules: list[dict],
) -> list[DetectedEntity]:
    entities: list[DetectedEntity] = []
    occupied: list[range] = []
    lowered = text.lower()
    for rule in feature_rules:
        keywords = sorted(
            (str(keyword) for keyword in rule.get("keywords") or []),
            key=len,
            reverse=True,
        )
        for keyword in keywords:
            pattern = _keyword_pattern(keyword)
            for match in pattern.finditer(lowered):
                if _overlaps(match.start(), match.end(), occupied):
                    continue
                occupied.append(range(match.start(), match.end()))
                entities.append(
                    DetectedEntity(
                        "FEATURE",
                        text[match.start() : match.end()],
                        str(rule["feature"]),
                        TextSpan(base_offset + match.start(), base_offset + match.end()),
                        dict(rule),
                    )
                )
    return entities


def _version_entities(
    text: str,
    base_offset: int,
    existing_entities: list[DetectedEntity],
) -> list[DetectedEntity]:
    entities: list[DetectedEntity] = []
    tech_spans = [e.span for e in existing_entities if e.entity_type == "TECH"]
    year_spans = [e.span for e in existing_entities if e.entity_type == "YEAR"]
    duration_spans = [e.span for e in existing_entities if e.entity_type == "DURATION"]
    blocked = [*tech_spans, *year_spans, *duration_spans]
    for match in VERSION_RE.finditer(text):
        absolute_start = base_offset + match.start()
        absolute_end = base_offset + match.end()
        if any(absolute_start < span.end and absolute_end > span.start for span in blocked):
            continue
        entities.append(
            DetectedEntity(
                "VERSION",
                match.group(0),
                match.group(1),
                TextSpan(absolute_start, absolute_end),
            )
        )
    return entities
