from typing import List
import numpy as np
from lexrank import LexRank, STOPWORDS
from nltk import sent_tokenize, word_tokenize

from optimization_based.summarizers.base import AbstractSummarizer


class LexRankSummarizer(AbstractSummarizer):
    def __init__(self):
        super().__init__()
        self.model_name = "LexRank"
        self.lxr = None

    def fit(self, documents: List[str]):
        self.documents = documents
        self.lxr = LexRank(documents, stopwords=STOPWORDS["en"])

    def get_summary(self, text: str, length: int) -> str:
        """Summarizes the input text
        Args:
            text (str): Input text to summarize.
            length (int): Summary length as number of sentences.
        Returns:
            summary (str): Summary of input text.
        """
        sentences = sent_tokenize(text)

        # get LexRank scores for sentences
        lex_scores = self.lxr.rank_sentences(
            sentences,
            threshold=None, # 0.1 also used in default
            fast_power_method=False, # Set to True for faster computation. Requires more RAM.
        )

        # order sentence indexes by descending sentence scores
        ranked_sentence_indexes = np.argsort(lex_scores)[::-1]

        # summarize and keep track of summary word count
        summary_word_count = 0
        selected_sentences = []
        
        for sentence_index in ranked_sentence_indexes:

            sentence = sentences[sentence_index]
            sentence_word_count = len(word_tokenize(sentence))

            # if adding sentence leads to less accurate word count, stop adding sentences
            if abs(length - summary_word_count - sentence_word_count) > abs(
                length - summary_word_count
            ):
                break
            
            selected_sentences.append(sentence)
            summary_word_count += sentence_word_count
        
        summary = ' '.join(selected_sentences)
        
        return summary.strip()
