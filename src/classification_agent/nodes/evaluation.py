"""
Evaluation node - compares prediction with ground truth and computes accuracy metrics.
Only runs when ground truth is provided in the state.
"""
from typing import Dict, Any
from classification_agent.graph.state import ClassificationState
from classification_agent.types.schemas import EvaluationResult, SingleEvaluationResult
from classification_agent.llm.base import BaseLLM
from classification_agent.utils.logging import get_logger
from .base_node import BaseNode

logger = get_logger(__name__)


class EvaluationNode(BaseNode):
    """Evaluate prediction against ground truth, compute accuracy metrics.

    This node only executes when ground_truth_data_items is provided in the state.
    It compares the final predicted labels with ground truth and calculates:
    - exact match accuracy (all labels match exactly)
    - macro precision / recall / F1
    """

    def __init__(self, llm: BaseLLM):
        super().__init__(llm)

    def _evaluate_single(
        self,
        predicted: list[str],
        ground_truth: list[str],
    ) -> SingleEvaluationResult:
        """Evaluate a single sample"""
        predicted_set = set(predicted)
        gt_set = set(ground_truth)

        correct = list(predicted_set & gt_set)
        wrong = list(predicted_set - gt_set)
        missing = list(gt_set - predicted_set)

        exact_match = predicted_set == gt_set

        return SingleEvaluationResult(
            predicted_data_items=predicted,
            ground_truth_data_items=ground_truth,
            correct_predictions=correct,
            wrong_predictions=wrong,
            missing_predictions=missing,
            exact_match=exact_match,
        )

    def process(self, state: ClassificationState) -> Dict[str, Any]:
        """Process evaluation: compare final prediction with ground truth"""
        ground_truth = state.get("ground_truth_data_items")
        final_labels = state.get("_final_labels")

        # If no ground truth or no final prediction, skip
        if ground_truth is None or final_labels is None:
            logger.debug("No ground truth or final prediction, skipping evaluation")
            return {"evaluation": None}

        # Evaluate single sample (graph processes one sample at a time)
        single_result = self._evaluate_single(final_labels, ground_truth)

        # Aggregate metrics (for single sample, macro equals sample)
        total_samples = 1
        exact_match_count = 1 if single_result["exact_match"] else 0
        exact_match_accuracy = exact_match_count / total_samples

        # Calculate TP, FP, FN for macro averaging
        # For each label, it's either in prediction or not
        tp = len(single_result["correct_predictions"])
        fp = len(single_result["wrong_predictions"])
        fn = len(single_result["missing_predictions"])

        total_tp = tp
        total_fp = fp
        total_fn = fn

        # Macro precision = TP / (TP + FP), avoid division by zero
        if (tp + fp) == 0:
            macro_precision = 0.0
        else:
            macro_precision = tp / (tp + fp)

        # Macro recall = TP / (TP + FN)
        if (tp + fn) == 0:
            macro_recall = 0.0
        else:
            macro_recall = tp / (tp + fn)

        # Macro F1
        if (macro_precision + macro_recall) == 0:
            macro_f1 = 0.0
        else:
            macro_f1 = 2 * (macro_precision * macro_recall) / (macro_precision + macro_recall)

        # Build full evaluation result
        evaluation_result = EvaluationResult(
            total_samples=total_samples,
            exact_match_count=exact_match_count,
            exact_match_accuracy=exact_match_accuracy,
            total_true_positives=total_tp,
            total_false_positives=total_fp,
            total_false_negatives=total_fn,
            macro_precision=macro_precision,
            macro_recall=macro_recall,
            macro_f1=macro_f1,
            per_sample_results=[single_result],
        )

        # Log result
        logger.info(
            "Evaluation completed: exact_match_accuracy=%.4f, macro_precision=%.4f, macro_recall=%.4f, macro_f1=%.4f",
            exact_match_accuracy,
            macro_precision,
            macro_recall,
            macro_f1,
        )

        return {"evaluation": evaluation_result}
