"""
Individual Evaluation Metrics for Local RAG Assessment:
- FaithfulnessMetric: Checks if answer contains only facts present in context.
- AnswerRelevancyMetric: Checks if answer directly addresses the user question.
- CorrectnessMetric: Checks if answer matches reference ground truth.
"""

from typing import Dict, Any, Optional
from langchain_ollama import ChatOllama


class FaithfulnessMetric:
    """
    Evaluates whether the generated response contains ONLY information present in retrieved context.
    """

    PROMPT_TEMPLATE = (
        "You are an evaluator assessing RAG system faithfulness.\n"
        "Question: {question}\n"
        "Retrieved Context: {context}\n"
        "Generated Answer: {answer}\n\n"
        "Does the generated answer contain ONLY information supported by the retrieved context? "
        "Answer with SCORE: 1.0 (fully faithful) or SCORE: 0.0 (unfaithful / contains hallucinations). "
        "Provide a short explanation."
    )

    def __init__(self, model_name: str = "mistral:latest"):
        self.llm = ChatOllama(model=model_name, temperature=0.0)

    def score(self, question: str, context: str, answer: str) -> Dict[str, Any]:
        prompt = self.PROMPT_TEMPLATE.format(
            question=question, context=context, answer=answer
        )
        response = self.llm.invoke(prompt).content
        score_val = 1.0 if "1.0" in response or "SCORE: 1" in response else 0.0
        return {"score": score_val, "reasoning": response}


class AnswerRelevancyMetric:
    """
    Evaluates whether the generated response directly addresses the user question.
    """

    PROMPT_TEMPLATE = (
        "You are an evaluator assessing answer relevancy.\n"
        "Question: {question}\n"
        "Generated Answer: {answer}\n\n"
        "Does the answer directly and concisely address the question? "
        "Answer with SCORE: 1.0 (relevant) or SCORE: 0.0 (irrelevant / off-topic). "
        "Provide a short explanation."
    )

    def __init__(self, model_name: str = "mistral:latest"):
        self.llm = ChatOllama(model=model_name, temperature=0.0)

    def score(self, question: str, answer: str) -> Dict[str, Any]:
        prompt = self.PROMPT_TEMPLATE.format(question=question, answer=answer)
        response = self.llm.invoke(prompt).content
        score_val = 1.0 if "1.0" in response or "SCORE: 1" in response else 0.0
        return {"score": score_val, "reasoning": response}


class CorrectnessMetric:
    """
    Evaluates whether the generated response matches reference ground-truth answer.
    """

    PROMPT_TEMPLATE = (
        "You are an evaluator assessing factual correctness against ground truth.\n"
        "Question: {question}\n"
        "Ground Truth Reference: {ground_truth}\n"
        "Generated Answer: {answer}\n\n"
        "Is the generated answer factually correct according to the ground truth? "
        "Answer with SCORE: 1.0 (correct) or SCORE: 0.0 (incorrect). "
        "Provide a short explanation."
    )

    def __init__(self, model_name: str = "mistral:latest"):
        self.llm = ChatOllama(model=model_name, temperature=0.0)

    def score(self, question: str, ground_truth: str, answer: str) -> Dict[str, Any]:
        prompt = self.PROMPT_TEMPLATE.format(
            question=question, ground_truth=ground_truth, answer=answer
        )
        response = self.llm.invoke(prompt).content
        score_val = 1.0 if "1.0" in response or "SCORE: 1" in response else 0.0
        return {"score": score_val, "reasoning": response}
