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
    ) -> ReverseJobResult:
        target_req = self.extractor.extract(target_job)

        implicit_arm: list[str] = []
        if corpus:
            transactions = [
                self.extractor.extract(doc).tech_stack for doc in corpus
            ]
            rules = self.miner.mine(transactions)
            implicit_arm = self.miner.find_implicit(
                set(target_req.tech_stack), rules
            )

        implicit_llm: list[str] = []
        if self.reasoner is not None:
            implicit_llm = self.reasoner.infer_implicit(
                target_job, target_req.tech_stack
            )

        union = sorted(set(implicit_arm) | set(implicit_llm))
        return ReverseJobResult(
            explicit=target_req,
            implicit_arm=implicit_arm,
            implicit_llm=implicit_llm,
            implicit_union=union,
        )
