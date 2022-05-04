from pathlib import Path
import argparse
import pandas as pd
from tqdm import tqdm

from data_utils.data import load_dataset_huggingface
from summary import summarize


def main(
    dataset: str,
    summarizer: str,
    summarizations_dir: Path,
    debug: bool = False,
    # save_output: bool = False
):
    # Make the data
    documents, summaries, length = load_dataset_huggingface(dataset=dataset)

    if debug:
        documents, summaries = documents[:5], summaries[:5]

    summarization_outputs = []



    for document, target_summary in tqdm(zip(documents, summaries)):
        summary_output = summarize(text=document, length=length, model_name=summarizer)
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
    # filename = "summarization_test"
    df.to_csv(f"{output_directory}/{filename}.csv", index=False)

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run summarization models on datasets."
    )

    parser.add_argument("--dataset", type=str, default="cnn_dailymail")
    parser.add_argument("--summarizer", type=str, default="TextRank")
    parser.add_argument(
        "--summarizations-dir", type=Path, default=Path("optimization_based/summarization_outputs")
    )
    parser.add_argument("--debug", action="store_true")
    # parser.add_argument("--save-output", action="store_true")

    args = parser.parse_args()

    main(**dict(args._get_kwargs()))
