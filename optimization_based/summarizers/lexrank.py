from typing import List
from lexrank import LexRank, STOPWORDS
from nltk import sent_tokenize, word_tokenize
# from sumy.parsers.plaintext import PlaintextParser
# from sumy.nlp.tokenizers import Tokenizer
# from sumy.summarizers.lex_rank import LexRankSummarizer as SumyLexRank

from optimization_based.summarizers.base import AbstractSummarizer


class LexRankSummarizer(AbstractSummarizer):
    def __init__(self):
        super().__init__()
        self.model_name = "LexRank"
        self.lxr = None

    def fit(self, documents: List[str]):
        self.documents = documents
        self.lxr = LexRank(documents, stopwords=STOPWORDS['en'])


    def get_summary(self, text: str, length: int) -> str:
        """Summarizes the input text
        Args:
            text (str): Input text to summarize.
            length (int): Summary length as number of sentences.
        Returns:
            summary (str): Summary of input text.
        """
        # parser = PlaintextParser.from_string(text, Tokenizer("english"))
        # summarizer = SumyLexRank()
        # summary = summarizer(parser.document, length)
        # summary = " ".join([sentence._text for sentence in summary])
        # get summary with classical LexRank algorithm

        sentences = sent_tokenize(text)
        summary_sentences = self.lxr.get_summary(sentences, summary_size=length, threshold=.1)
        summary = ' '.join(summary_sentences).strip()

        return summary
