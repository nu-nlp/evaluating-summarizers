from pathlib import Path
import argparse
import pandas as pd
from tqdm import tqdm

from data_utils.data import load_dataset_huggingface

# from summary import summarize


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
    abstract_summarizer,
)

summarization_length_mapping = {
    "TextRank": "word count",
    "LexRank": "sentence count",
    "Lead": "word count",
    "Random": "word count",
}


def summarize_document(text: str, length: int, summarizer: abstract_summarizer) -> Dict:
    """Summarizes the input text
    Args:
        text (str): Input text to summarize
        length (int): Summary length as number of words.
    Returns:
        summary_output (Dict): Summary of input text and some data
    """
    # convert length from word count to target number of sentences if model requires it
    words_or_sentences = summarization_length_mapping.get(
        summarizer.model_name, "word count"
    )
    if words_or_sentences == "sentence count":
        sentence_lengths = [
            len(word_tokenize(sentence)) for sentence in sent_tokenize(text)
        ]
        length = math.ceil(length // np.mean(sentence_lengths))

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


def main(
    dataset: str,
    summarizer: str,
    summarizations_dir: Path,
    debug: bool = False,
    # save_output: bool = False
):
    # Make the data
    documents, summaries, length = load_dataset_huggingface(dataset=dataset)

    # Set up/train summarizer
    if summarizer == "TextRank":
        model = textrank_summarizer

    elif summarizer == "LexRank":
        model = lexrank_summarizer
        # start timer
        start = datetime.now()        
        print(f"Start training LexRank")
        model.fit(documents=documents)
        # end timer
        end = datetime.now()
        print(f"LexRank training time on {dataset}= {end-start}")

    elif summarizer == "Random":
        model = random_summarizer

    else:
        model = lead_summarizer
    
    # TODO: add other summarizers

    if debug:
        documents, summaries = documents[:5], summaries[:5]

    summarization_outputs = []

    for document, target_summary in tqdm(zip(documents, summaries)):
        summary_output = summarize_document(
            text=document, length=length, summarizer=model
        )
        # summary_output = summarize(text=document, length=length, model_name=summarizer)
        summary_output["target"] = target_summary
        summary_output["target_word_count"] = length
        summarization_outputs.append(summary_output)

    df = pd.DataFrame(summarization_outputs)

    column_names = [
        "model",
        "document",
        "summary",
        "target",
        "document_word_count",
        "summary_word_count",
        "target_word_count",
        "summarization_time",
    ]
    df = df.reindex(columns=column_names)

    output_directory = summarizations_dir / dataset / summarizer
    output_directory.mkdir(parents=True, exist_ok=True)
    filename = "summarization_test_debug" if debug else "summarization_test"
    df.to_csv(f"{output_directory}/{filename}.csv", index=False)

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run summarization models on datasets."
    )

    parser.add_argument("--dataset", type=str, default="cnn_dailymail")
    parser.add_argument("--summarizer", type=str, default="TextRank")
    parser.add_argument(
        "--summarizations-dir",
        type=Path,
        default=Path("optimization_based/summarization_outputs"),
    )
    parser.add_argument("--debug", action="store_true")
    # parser.add_argument("--save-output", action="store_true")

    args = parser.parse_args()

    main(**dict(args._get_kwargs()))
