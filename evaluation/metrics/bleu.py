from typing import List, Dict
from nltk.tokenize import word_tokenize
from datasets import load_metric

from evaluation.metrics.base import AbstractMetric


class BleuMetric(AbstractMetric):
    def __init__(self):
        super().__init__()
        self.metric_name = "bleu"

    @staticmethod
    def evaluate(predictions: List[str], references: List[str]) -> Dict:
        """Runs bleu: https://github.com/huggingface/datasets/tree/master/metrics/bleu

        Args:
            predictions (List[str]): list of summaries
            references (List[str]): list of target summaries

        Returns:
            results: Dictionary of metric score output
        """
        # tokenize predictions and references
        # make sure to format inputs as described in metric documentation
        # predictions = [word_tokenize(prediction) for prediction in predictions]
        predictions = [prediction.split() for prediction in predictions]
        # references = [[word_tokenize(reference)] for reference in references]
        references = [[reference.split()] for reference in references]

        # load metric
        metric = load_metric("bleu")

        # compute metric
        scores = metric.compute(predictions=predictions, references=references)

        return 100 * scores["bleu"]
