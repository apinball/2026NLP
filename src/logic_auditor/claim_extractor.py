from __future__ import annotations

from src.logic_auditor.document_parser import sentence_spans
from src.logic_auditor.entity_extractor import EntityExtractor
from src.logic_auditor.models import Claim, DetectedEntity


class ClaimExtractor:
    def __init__(self, entity_extractor: EntityExtractor | None = None) -> None:
        self.entity_extractor = entity_extractor or EntityExtractor()

    def extract(self, text: str) -> tuple[Claim, ...]:
        claims: list[Claim] = []
        for index, span in enumerate(sentence_spans(text), start=1):
            claim_text = text[span.start : span.end]
            entities = self.entity_extractor.extract(claim_text, base_offset=span.start)
            claims.append(
                Claim(
                    claim_id=f"c{index}",
                    claim_text=claim_text,
                    span=span,
                    entities=entities,
                    claim_type=infer_claim_type(entities, claim_text),
                )
            )
        return tuple(claims)


def infer_claim_type(entities: tuple[DetectedEntity, ...], text: str) -> str:
    entity_types = {entity.entity_type for entity in entities}
    if "FEATURE" in entity_types or "VERSION" in entity_types:
        return "version_usage"
    if "TECH" in entity_types and "YEAR" in entity_types:
        return "tech_timeline"
    if {"DURATION", "ROLE", "SCALE"}.intersection(entity_types):
        return "scope_claim"
    if any(term in text for term in ("개선", "절감", "증가", "향상", "%")):
        return "result_claim"
    return "general"
