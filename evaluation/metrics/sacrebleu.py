from typing import List, Dict
from datasets import load_metric

from evaluation.metrics.base import AbstractMetric


class SacrebleuMetric(AbstractMetric):
    def __init__(self):
        super().__init__()
        self.metric_name = "sacrebleu"

    @staticmethod
    def evaluate(predictions: List[str], references: List[str]) -> Dict:
        """Runs sacrebleu: https://github.com/huggingface/datasets/tree/master/metrics/sacrebleu
        Args:
            predictions (List[str]): list of summaries
            references (List[str]): list of target summaries

        Returns:
            results: Dictionary of metric score output
        """
        # make sure to format inputs as described in metric documentation
        references = [[reference] for reference in references]

        # load metric
        metric = load_metric("sacrebleu")

        # compute metric
        scores = metric.compute(predictions=predictions, references=references)

        # select the key/score we are interested in
        sacrebleu_score = scores["score"]

        return sacrebleu_score
