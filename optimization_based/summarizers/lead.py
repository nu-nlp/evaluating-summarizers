from nltk.tokenize import word_tokenize, sent_tokenize
from optimization_based.summarizers.base import AbstractSummarizer


class LeadSummarizer(AbstractSummarizer):
    def __init__(self):
        super().__init__()
        self.model_name = "Lead"

    def get_summary(self, text: str, length: int) -> str:
        """Summarizes the input text
        Args:
            text (str): Input text to summarize.
            length (int): Summary length as number of words.
        Returns:
            summary (str): Summary of input text.
        """
        # split text into sentences
        sentences = sent_tokenize(text)

        # summarize and keep track of summary word count
        summary_word_count = 0
        selected_sentences = []

        for sentence in sentences:

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
