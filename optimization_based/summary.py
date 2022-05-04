from typing import Dict
import math
from datetime import datetime
import numpy as np
from nltk import word_tokenize, sent_tokenize

from optimization_based.extensions import (
    textrank_summarizer,
    lexrank_summarizer,
    lead_summarizer,
    random_summarizer,
)

summarization_length_mapping = {
    "TextRank": "word count",
    "LexRank": "sentence count",
    "Lead": "word count",
    "Random": "word count",
}

def summarize(text: str, length: int, model_name: str) -> Dict:
    """Summarizes the input text
    Args:
        text (str): Input text to summarize
        length (int): Summary length as number of words.
    Returns:
        summary_output (Dict): Summary of input text and some data
    """
    if model_name == "TextRank":
        summarizer = textrank_summarizer

    elif model_name == "LexRank":
        summarizer = lexrank_summarizer
        # summarizer.fit(documents=['test'])
        # print(summarizer.documents)

    elif model_name == "Random":
        summarizer = random_summarizer

    else:
        summarizer = lead_summarizer
    # TODO: add other summarizers

    # convert length from word count to target number of sentences if model requires it
    words_or_sentences = summarization_length_mapping.get(
        summarizer.model_name, "word count"
    )
    if words_or_sentences == "sentence count":
        sentence_lengths = [
            len(word_tokenize(sentence)) for sentence in sent_tokenize(text)
        ]
        length = math.ceil(length // np.median(sentence_lengths))

    # start timer
    start = datetime.now()

    # summarize
    summary = summarizer.get_summary(text=text, length=length)

    # end timer
    end = datetime.now()

    # record document and summary lengths in words
    document_word_count = len(word_tokenize(text))
    summary_word_count = len(word_tokenize(summary))

    summary_output = {
        "model": summarizer.model_name,
        "document": text,
        "document_word_count": document_word_count,
        "summary": summary,
        "summary_word_count": summary_word_count,
        "summarization_time": end - start,
    }

    return summary_output

if __name__ == "__main__":
    text = (
        "This is a test for chicken nuggets. "
        "Did this work? "
        "Chicken nuggets should be in the summary. "
        "Do we have the technology to contact aliens?"
    )
    length = 10
    model_name = "TextRank"
    summary_output = summarize(text, length, model_name)
    print(summary_output)

    model_name = "LexRank"
    summary_output = summarize(text, length, model_name)
    print(summary_output)

    model_name = "Random"
    summary_output = summarize(text, length, model_name)
    print(summary_output)

    model_name = "Lead"
    summary_output = summarize(text, length, model_name)
    print(summary_output)
