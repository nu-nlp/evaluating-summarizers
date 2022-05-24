from typing import List, Dict

from evaluation.metrics.base import AbstractMetric

def avg_gen_length(pred_summary: str) -> int:
    """
    Computes the number of tokens in the given predicted summary.
    Args:
        pred_summary: The predicted summary text string.

    Returns:
        The number of tokens in the predicted summary.
    """
    pred_tokens = pred_summary.split(' ')
    return len(pred_tokens)


class AvgGenLengthMetric(AbstractMetric):
    def __init__(self):
        super().__init__()
        self.metric_name = "avg_gen_length"

    @staticmethod
    def evaluate(predictions: List[str], references: List[str]) -> Dict:
        """Computes the average generated length of the prediction summaries.

        Args:
            predictions (List[str]): list of summaries
            references (List[str]): list of target summaries

        Returns:
            results: Dictionary of metric score output
        """

        # Calculate the number of tokens for every prediction summary
        lens = [avg_gen_length(pred_sum) for pred_sum in predictions]

        # Average the lengths to produce a single, final score
        return float(sum(lens)) / float(len(lens))
