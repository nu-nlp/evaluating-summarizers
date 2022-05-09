import numpy as np
from typing import List, Dict
from scipy.spatial import distance
from sklearn.feature_extraction.text import CountVectorizer

from evaluation.metrics.base import AbstractMetric

def compute_token_probabilities(token_counts: List[int]) -> List[float]:
    """
    Computes the probability of each token occuring relative to the others.
    Args:
        token_counts: A list of frequency counts of tokens.

    Returns:
        A list of probabilities for each token.
    """
    num_tokens = sum(token_counts)
    return [c / float(num_tokens) for c in token_counts]

def jensen_shannon_distance(pred_summary: str,
                            reference: str) -> float:
    """
    Computes the Jensen-Shannon distance of two texts using the unigram probabilities in the two texts.
    Args:
        pred_summary: The predicted summary text string.
        reference: The reference text string.

    Returns:
        The Jensen-Shannon distance as a float.
    """

    # Get unigram token frequencies
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform([pred_summary, reference])
    # feature_names = vectorizer.get_feature_names_out()

    # Get the unigram counts for the summary text and reference text
    summary_counts = X.toarray()[0]
    reference_counts = X.toarray()[1]

    # If there are no tokens for both summaries, the distance is the same
    if sum(summary_counts) == 0 and sum(reference_counts) == 0:
        return 0.0

    # If there is no tokens for either summary, the distance is the max value of 1.0
    if sum(summary_counts) == 0 or sum(reference_counts) == 0:
        return 1.0

    # Get the unigram probabilities of the summary text and reference text
    summary_probs = compute_token_probabilities(summary_counts)
    reference_probs = compute_token_probabilities(reference_counts)

    # Use SciPy to calculate the Jensen-Shannon distance
    return distance.jensenshannon(summary_probs, reference_probs)


class JensenShannonMetric(AbstractMetric):
    def __init__(self):
        super().__init__()
        self.metric_name = "jensen_shannon"

    @staticmethod
    def evaluate(predictions: List[str], references: List[str]) -> Dict:
        """Computes the average Jensen-Shannon distance between the prediction and reference summaries.

        Args:
            predictions (List[str]): list of summaries
            references (List[str]): list of target summaries

        Returns:
            results: Dictionary of metric score output
        """

        # Calculate the Jensen-Shannon distance for every prediction-reference pair
        scores = [jensen_shannon_distance(pair[0], pair[1]) for pair in zip(predictions, references)]

        # Average the distances to produce a single, final score
        jensen_shannon_avg = np.mean(scores)

        return jensen_shannon_avg*100