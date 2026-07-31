"""
Evaluation framework for topic discovery and local RAG pipelines.
Includes LLM-as-judge scoring for Faithfulness, Answer Relevancy, and Ground-Truth Correctness.
"""

from .llm_as_judge import LLMAsJudge
from .metrics import FaithfulnessMetric, AnswerRelevancyMetric, CorrectnessMetric

__all__ = [
    "LLMAsJudge",
    "FaithfulnessMetric",
    "AnswerRelevancyMetric",
    "CorrectnessMetric",
]
