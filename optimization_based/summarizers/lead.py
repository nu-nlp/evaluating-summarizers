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
        summary = ""
        sentence_index, summary_word_count = 0, 0
        
        while summary_word_count <= length and sentence_index < len(sentences):
            sentence = sentences[sentence_index]
            summary += sentence + " "
            sentence_index += 1
            summary_word_count += len(word_tokenize(sentence))

        return summary.strip()
        