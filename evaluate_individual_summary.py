import os
import pandas as pd
from data_utils.data import datasets_mapping
import nltk
nltk.download("punkt")

from evaluation.extensions import (
    bleu_metric,
    sacrebleu_metric,
    rouge_metric,
    bertscore_metric,
    jensen_shannon_metric
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

DATASETS = "cnn_dailymail"
MODELS = ["TextRank", "LexRank", "Lead", "Random", "bartbase", "bartlarge", "t5small"]
METRICS = ["sacrebleu", "bleu", "rouge", "bertscore", "jensen_shannon"]

summarizations_dir = '/content/drive/MyDrive/NU-NLP/summarization_outputs'


def load_summarization_outputs(
    summarizations_dir,
        dataset=DATASETS,
        summarizers=MODELS,
        summary_column='summary',
        target_column='target',
        debug=False
):

    for summarizer in summarizers:
        if summarizer in ["TextRank", "LexRank", "Lead", "Random"]:
            # optimization based models outputs will be found with this code
            filename = "summarization_test_debug.csv" if debug else "summarization_test.csv"
            file = os.path.join(summarizations_dir, dataset, summarizer, filename)
        else:
            # To run this on David's CSVs: e.g. billsum_bartbase_197.csv
            # print(datasets_mapping[dataset])
            filename = f"{david_datasets_mapping[dataset]}_{summarizer}_{datasets_mapping[dataset][4]}.csv"
            file = os.path.join(summarizations_dir, filename)

        # Read the summarization outputs into  dataframe
        df = pd.read_csv(file, nrows=100)
        df = df[[target_column, summary_column]]
        print(df.shape)
        predictions = df[summary_column].fillna("").tolist()
        references = df[target_column].fillna("").tolist()

        print(f"Processing summaries for model {summarizer}")
        # print(len(predictions), len(references))
        for metric_name in METRICS:
            # load metric
            results = []
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

            elif metric_name == "jensen_shannon":
                metric = jensen_shannon_metric


            for id in range(len(predictions)):
                    if predictions[id] != '':
                        results.append(metric.evaluate([predictions[id]], [references[id]]))
                    else:
                        results.append(0.0)
            df[metric_name] = results
            # df['bleu'] = df.apply(lambda x: evaluate(predictions, references), axis=1)
            # df = pd.DataFrame(summarization_outputs)

        print(f"Model {summarizer} processed")
        output_directory = os.path.join(summarizations_dir, dataset, summarizer)
        os.makedirs(output_directory, exist_ok=True)
        filename = "summarization_scores_per_summary"
        df.to_csv(f"{output_directory}/{filename}.csv", index=False)
    return

if __name__ == "__main__":

    results = load_summarization_outputs(summarizations_dir)

