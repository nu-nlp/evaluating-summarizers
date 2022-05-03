from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer as SumyLexRank

from summarizers.base import AbstractSummarizer


class LexRankSummarizer(AbstractSummarizer):
    def __init__(self):
        super().__init__()
        self.model_name = "LexRank"

    @staticmethod
    def get_summary(text: str, length: int) -> str:
        """Summarizes the input text
        Args:
            text (str): Input text to summarize.
            length (int): Summary length as number of sentences.
        Returns:
            summary (str): Summary of input text.
        """
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        summarizer = SumyLexRank()
        summary = summarizer(parser.document, length)
        # summary = " ".join([str(sentence) for sentence in summary])
        summary = " ".join([sentence._text for sentence in summary])

        return summary
