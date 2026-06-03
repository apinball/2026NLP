from __future__ import annotations

from dataclasses import dataclass, field

from src.reverse_job.arm_miner import ARMMiner
from src.reverse_job.extractor import JobExtractor, JobRequirements
from src.reverse_job.llm_reasoner import LLMReasoner


@dataclass
class ReverseJobResult:
    explicit: JobRequirements
    implicit_arm: list[str] = field(default_factory=list)
    implicit_llm: list[str] = field(default_factory=list)
    implicit_union: list[str] = field(default_factory=list)


class ReverseJobPipeline:
    """Module A — Reverse Job Engineering 오케스트레이터."""

    def __init__(
        self,
        extractor: JobExtractor | None = None,
        miner: ARMMiner | None = None,
        reasoner: LLMReasoner | None = None,
        load_ner: bool = True,
    ) -> None:
        self.extractor = extractor or JobExtractor(load_ner=load_ner)
        self.miner = miner or ARMMiner()
        self.reasoner = reasoner

    def run(
        self,
        target_job: str,
        corpus: list[str] | None = None,
        transactions: list[list[str]] | None = None,
    ) -> ReverseJobResult:
        """corpus 는 raw 채용공고 텍스트 리스트, transactions 는 이미 추출된
        스킬 리스트들. 둘 중 하나만 제공해도 되며, transactions 가 우선."""
        target_req = self.extractor.extract(target_job)

        implicit_arm: list[str] = []
        if transactions is None and corpus:
            transactions = [
                self.extractor.extract(doc).tech_stack for doc in corpus
            ]
        if transactions:
            rules = self.miner.mine(transactions)
            implicit_arm = self.miner.find_implicit(
                set(target_req.tech_stack), rules
            )

        implicit_llm: list[str] = []
        if self.reasoner is not None:
            implicit_llm = self.reasoner.infer_implicit(
                target_job,
                target_req.tech_stack,
                arm_candidates=implicit_arm,
            )

        union = sorted(set(implicit_arm) | set(implicit_llm))
        return ReverseJobResult(
            explicit=target_req,
            implicit_arm=implicit_arm,
            implicit_llm=implicit_llm,
            implicit_union=union,
        )
