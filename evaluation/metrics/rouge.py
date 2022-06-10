from typing import List, Dict
from datasets import load_metric

from evaluation.metrics.base import AbstractMetric


class RougeMetric(AbstractMetric):
    def __init__(self):
        super().__init__()
        self.metric_name = "rouge"

    @staticmethod
    def evaluate(predictions: List[str], references: List[str]) -> Dict:
        """Runs rouge: https://github.com/huggingface/datasets/tree/master/metrics/rouge
        Tokenizer: Google research tokenizer
        Args:
            predictions (List[str]): list of summaries
            references (List[str]): list of target summaries

        Returns:
            results: Dictionary of metric score output
        """
        # make sure to format inputs as described in metric documentation
        # not required here

        # load metric
        metric = load_metric("rouge")

        # preprocessing: each prediction or reference should be a string separated by spaces
        # predictions = [' '.join(prediction.split()) for prediction in predictions]
        # references = [' '.join(reference.split()) for reference in references]
        
        # compute metric
        scores = metric.compute(predictions=predictions, references=references)

        rouge_scores = {}
        rouge_scores['rouge1'] = 100 * scores["rouge1"].mid.fmeasure
        rouge_scores['rouge2'] = 100 * scores["rouge2"].mid.fmeasure
        rouge_scores['rougeL'] = 100 * scores["rougeL"].mid.fmeasure

        return rouge_scores
