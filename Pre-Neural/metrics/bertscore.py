from typing import List, Dict
from metrics.base import AbstractMetric
import numpy as np
from datasets import load_metric


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

        
        #TODO: Explore best params here. You can pick batch size, lang, model_type. Ask David for model type and batch size
        # Increasing batch size will make this run quicker so I recommend increasing to whatever fits on your machine
        # You can also use roberta as default with lang="en" or you can select a model type
        # Also the outout of bertscore will give f1, precision, recall for each output.
        # So I think you need to average those outputs to get "overall bert score"
        # e.g. {'summarizer': 'TextRank', 'dataset': 'cnn_dailymail', 'bertscore': 
        # {'f1': [0.8529757261276245, 0.8617121577262878, 0.0, 0.8795979022979736, 0.8753492832183838]}}
        # bert score should be np.mean(bertscore['f1'])
        # Look at metric card for details: https://github.com/huggingface/datasets/tree/master/metrics/bertscore
        
        # compute metric
        scores = metric.compute(predictions=predictions, references=references, lang="en")
        # scores = metric.compute(predictions=predictions, references=references, model_type="distilbert-base-uncased") 
        bertscore = np.mean(scores['f1'])

        return 100 * bertscore
