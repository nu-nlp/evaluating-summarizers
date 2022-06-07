from summa.summarizer import summarize
# from .summa_local.summarizer import summarize
from optimization_based.summarizers.base import AbstractSummarizer


class TextRankSummarizer(AbstractSummarizer):
    def __init__(self):
        super().__init__()
        self.model_name = "TextRank"


    def get_summary(self, text: str, length: int) -> str:
        """Summarizes the input text.
        Args:
            text (str): Input text to summarize.
            length (int): Summary length as number of words.
        Returns:
            summary (str): Summary of input text.
        """
        summary = summarize(text, words=length)

        return summary.strip()
