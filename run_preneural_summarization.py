from typing import Dict
from pathlib import Path
from datetime import datetime
import argparse

import pandas as pd
from tqdm import tqdm
from nltk import word_tokenize, sent_tokenize
import nltk
nltk.download('punkt')
from data_utils.data import load_dataset_huggingface


from optimization_based.extensions import (
    textrank_summarizer,
    lexrank_summarizer,
    lead_summarizer,
    random_summarizer,
    occams_summarizer,
    abstract_summarizer,
)

summarization_length_mapping = {
    "TextRank": "word count",
    "LexRank": "sentence count",
    "Lead": "word count",
    "Random": "word count",
    "Occams": "word count"
}

def summarize_document(text: str,
                       length: int,
                       summarizer: abstract_summarizer) -> Dict:
    """Summarizes the input text
    Args:
        text (str): Input text to summarize
        length (int): Summary length as number of words.
    Returns:
        summary_output (Dict): Summary of input text and some data
    """
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


def main(training_dataset: str,
         evaluation_dataset: str,
         summarizer: str,
         target_length: int,
         summarizations_dir: Path,
         debug: bool = False) -> None:

    # Load in the evaluation dataset
    documents, summaries, length = load_dataset_huggingface(dataset=evaluation_dataset)

    # Set the training dataset, if none was specified by the user
    if training_dataset is None:
        training_dataset = evaluation_dataset

    # Use the target_length specified by the user
    length = target_length

    if debug:
        documents, summaries = documents[:5], summaries[:5]

    # Set up/train summarizer
    if summarizer == "TextRank":
        model = textrank_summarizer

    elif summarizer == "LexRank":
        documents_as_sentences = [sent_tokenize(document) for document in documents]
        model = lexrank_summarizer

        start = datetime.now() # start timer
        print(f"Start training LexRank")
        model.fit(documents=documents_as_sentences)
        end = datetime.now() # end timer
        print(f"LexRank training time on {training_dataset}= {end-start}")

    elif summarizer == "Random":
        model = random_summarizer

    elif summarizer == "Occams":
        model = occams_summarizer

    else:
        model = lead_summarizer

    # Generate the summaries
    print(f"Generating summaries with {summarizer} (\"trained\" on {training_dataset}) with target length of {target_length} on evaluation dataset {evaluation_dataset}")
    summarization_outputs = []
    for document, target_summary in tqdm(zip(documents, summaries)):
        summary_output = summarize_document(text=document, length=length, summarizer=model)
        # summary_output = summarize(text=document, length=length, model_name=summarizer)

        # Add "columns" for target summary and the word count of that summary
        summary_output["target"] = target_summary
        summary_output["target_word_count"] = length
        summarization_outputs.append(summary_output)

    # Create a Pandas dataframe to store the generated summaries
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

    # Get the directory/path where the generated summaries will be saved
    summary_filename = f"{training_dataset}_{evaluation_dataset}_{summarizer}_{target_length}"
    if debug:
        summary_filename += "_debug"

    # Save the generated summaries
    summarizations_dir.mkdir(parents=True, exist_ok=True)
    final_output_path = f"{summarizations_dir}/{summary_filename}.csv"
    df.to_csv(final_output_path, index=False)
    print(f"Finished. Wrote generated summaries to: {final_output_path}")

    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate summaries using the specified pre-neural models and datasets.")

    parser.add_argument("--training_dataset", type=str, default=None)
    parser.add_argument("--evaluation_dataset", type=str, default="arxiv")
    parser.add_argument("--summarizer", type=str, default="TextRank")
    parser.add_argument("--target_length", type=int, default=200)

    # Directory arguments
    parser.add_argument("--summarizations_dir", type=Path, default=Path("evaluation/generated_summaries"))

    # Debug arguments
    parser.add_argument("--debug", action="store_true")
    # parser.add_argument("--save-output", action="store_true")

    # Parse the arguments and run the script
    args = parser.parse_args()
    main(**dict(args._get_kwargs()))
