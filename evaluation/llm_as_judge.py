"""
LLM-as-Judge Evaluation Orchestrator.
Runs batch evaluation across Faithfulness, Answer Relevancy, and Correctness metrics.
"""

from typing import List, Dict, Any, Optional
from tqdm import tqdm
from .metrics import FaithfulnessMetric, AnswerRelevancyMetric, CorrectnessMetric


class LLMAsJudge:
    """
    Orchestrates RAG evaluation across multiple test samples using local Ollama model as judge.

    Parameters
    ----------
    judge_model_name : str, default="mistral:latest"
        Name of local LLM used as evaluator.
    """

    def __init__(self, judge_model_name: str = "mistral:latest"):
        self.faithfulness_metric = FaithfulnessMetric(model_name=judge_model_name)
        self.relevancy_metric = AnswerRelevancyMetric(model_name=judge_model_name)
        self.correctness_metric = CorrectnessMetric(model_name=judge_model_name)

    def evaluate_sample(
        self,
        question: str,
        answer: str,
        context: Optional[str] = None,
        ground_truth: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a single (question, answer, context, ground_truth) sample.
        """
        results = {"question": question, "answer": answer}

        if context:
            results["faithfulness"] = self.faithfulness_metric.score(question, context, answer)

        results["relevancy"] = self.relevancy_metric.score(question, answer)

        if ground_truth:
            results["correctness"] = self.correctness_metric.score(question, ground_truth, answer)

        return results

    def evaluate_dataset(
        self,
        samples: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """
        Evaluate a batch of test samples.
        Each sample dict should contain: 'question', 'answer', and optionally 'context', 'ground_truth'.
        """
        eval_results = []
        faithfulness_scores = []
        relevancy_scores = []
        correctness_scores = []

        for sample in tqdm(samples, desc="Evaluating RAG Samples"):
            res = self.evaluate_sample(
                question=sample["question"],
                answer=sample["answer"],
                context=sample.get("context"),
                ground_truth=sample.get("ground_truth")
            )
            eval_results.append(res)

            if "faithfulness" in res:
                faithfulness_scores.append(res["faithfulness"]["score"])
            if "relevancy" in res:
                relevancy_scores.append(res["relevancy"]["score"])
            if "correctness" in res:
                correctness_scores.append(res["correctness"]["score"])

        summary = {
            "total_samples": len(samples),
            "mean_faithfulness": sum(faithfulness_scores) / len(faithfulness_scores) if faithfulness_scores else None,
            "mean_relevancy": sum(relevancy_scores) / len(relevancy_scores) if relevancy_scores else None,
            "mean_correctness": sum(correctness_scores) / len(correctness_scores) if correctness_scores else None,
            "sample_details": eval_results
        }
        return summary
