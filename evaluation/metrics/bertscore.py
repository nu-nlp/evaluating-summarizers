from typing import List, Dict
import numpy as np
from datasets import load_metric

from evaluation.metrics.base import AbstractMetric


class BertscoreMetric(AbstractMetric):
    def __init__(self):
        super().__init__()
        self.metric_name = "bertscore"

    @staticmethod
    def evaluate(predictions: List[str], references: List[str]) -> Dict:
        """Runs bertscore: https://github.com/huggingface/datasets/tree/master/metrics/bertscore

        Args:
            predictions (List[str]): list of summaries
            references (List[str]): list of target summaries

        Returns:
            results: Dictionary of metric score output
        """
        # make sure to format inputs as described in metric documentation
        # not required here
        
        # load metric
        metric = load_metric("bertscore")
        
        # compute metric
        scores = metric.compute(predictions=predictions, references=references, lang="en")
        # scores = metric.compute(predictions=predictions, references=references, model_type="distilbert-base-uncased") 
        bertscore = np.mean(scores['f1'])

        return 100 * bertscore
