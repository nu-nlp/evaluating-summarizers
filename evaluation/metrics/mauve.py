from typing import List, Dict
from datasets import load_metric

from evaluation.metrics.base import AbstractMetric


class MauveMetric(AbstractMetric):
    def __init__(self):
        super().__init__()
        self.metric_name = "mauve"

    @staticmethod
    def evaluate(predictions: List[str], references: List[str]) -> Dict:
        """Runs mauve: https://github.com/huggingface/datasets/tree/master/metrics/mauve

        Args:
            predictions (List[str]): list of summaries
            references (List[str]): list of target summaries

        Returns:
            results: Dictionary of metric score output
        """
        # make sure to format inputs as described in metric documentation
        # not required here

        # predictions = ["hello world", "goodnight moon"]
        # references = ["hello world",  "goodnight moon"]
        
        # load metric
        metric = load_metric("mauve")

        
        #TODO: Debug this metric.
        # Don't share this metric if not understood. I think it uses GPU so run on colab
        # Look at metric card for details: https://github.com/huggingface/datasets/tree/master/metrics/mauve
        
        # compute metric
        scores = metric.compute(predictions=predictions, references=references)

        return scores