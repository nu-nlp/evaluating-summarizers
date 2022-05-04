import json
from typing import List
from pathlib import Path
import argparse
import pandas as pd
import nltk
from data_utils.data import datasets_mapping

nltk.download("punkt")

from evaluation.extensions import (
    bleu_metric,
    sacrebleu_metric,
    rouge_metric,
    bertscore_metric,
    # mauve_metric,
)

david_datasets_mapping = {
    "cnn_dailymail": "cnn",
    "arxiv": "arxiv",
    # "pubmed": "pubmed",
    # "reddit_tifu": "reddit_tifu",
    "billsum": "billsum",
    "govreport": "gov",
}


def load_summarization_outputs(
    dataset: str,
    summarizer: str,
    summary_column: str,
    target_column: str,
    summarizations_dir: Path,
    debug: bool,
):
    if summarizer in ["TextRank", "LexRank", "Lead", "Random"]:
        # optimization based models outputs will be found with this code
        filename = "summarization_test_debug.csv" if debug else "summarization_test.csv"
        file = summarizations_dir / dataset / summarizer / filename
    else:
        # To run this on David's CSVs: e.g. billsum_bartbase_197.csv
        filename = f"{david_datasets_mapping[dataset]}_{summarizer}_{datasets_mapping[dataset][4]}.csv"
        file = summarizations_dir / filename
    
    # Read the summarization outputs into  dataframe
    df = pd.read_csv(file)

    # TextRank returns empty summaries if the document does not have enough sentences
    # It seems to occur with bad sentence tokenization and is probably fixable in the future
    # https://github.com/summanlp/textrank/issues/28
    # For now we replace empty summaries with empty strings.
    
    predictions = df[summary_column].fillna("").tolist()
    references = df[target_column].fillna("").tolist()
    
    # unit_summary_time = df['summarization_time']
    # summarization_time = unit_summary_time.mean()

    # if you want to debug, use debug flag to restrict to 5 samples
    if debug:
        predictions, references = predictions[:5], references[:5]

    return predictions, references


def evaluate(
    summarizations_dir: Path,
    scores_dir: Path,
    dataset: str,
    summary_column: str,
    target_column: str,
    summarizer: str,
    metrics: List[str],
    debug: bool,
):
    predictions, references = load_summarization_outputs(
        dataset, summarizer, summary_column, target_column, summarizations_dir, debug
    )

    # create a dictionary for the metric results
    results = {"summarizer": summarizer, "dataset": dataset}

    for metric_name in metrics:

        # load metric
        if metric_name == "bleu":
            metric = bleu_metric

        elif metric_name == "sacrebleu":
            metric = sacrebleu_metric

        elif metric_name == "rouge":
            metric = rouge_metric

        elif metric_name == "bertscore":
            metric = bertscore_metric

        # TODO: NOT PRIORITY. Implement these two successfully
        # Mauve does not work yet
        # elif metric_name == "mauve":
        #   metric = mauve_metric

        # elif metric_name == "jensen-shannon":
        #   metric = jensen_shannon_metric

        # evaluate summaries
        scores = metric.evaluate(predictions=predictions, references=references)

        results[metric_name] = scores

    output_directory = scores_dir / dataset / summarizer
    output_directory.mkdir(parents=True, exist_ok=True)
    
    filename = "evaluation_test_debug" if debug else "evaluation_test"
    if target_column == "label":
        filename = "evaluation_test_debug_label" if debug else "evaluation_test_label"
    
    with open(output_directory / f"{filename}.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run evaluation metrics on summarization outputs."
    )

    parser.add_argument("--dataset", type=str, default="cnn_dailymail")
    parser.add_argument("--summarizer", type=str, default="TextRank")
    parser.add_argument(
        "--summarizations-dir",
        type=Path,
        default=Path("optimization_based/summarization_outputs"),
    )
    parser.add_argument("--scores-dir", type=Path, default=Path("evaluation/evaluation_outputs"))
    parser.add_argument("--summary-column", type=str, default="summary")
    parser.add_argument("--target-column", type=str, default="target")
    parser.add_argument(
        "--metrics", type=str, nargs="*", default=["rouge", "sacrebleu", "bleu"]
    )
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    results = evaluate(**dict(args._get_kwargs()))
    """
    scores_dir = args.scores_dir
    dataset = args.dataset
    summarizer = args.summarizer

    output_directory = scores_dir / dataset / summarizer
    output_directory.mkdir(parents=True, exist_ok=True)
    filename = "evaluation_test_debug" if args.debug else "evaluation_test"
    if args.target_column == 'label':
        filename = "evaluation_test_debug_label" if args.debug else "evaluation_test_label"
    with open(output_directory / f"{filename}.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    """
