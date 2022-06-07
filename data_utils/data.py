from datasets import load_dataset
from sklearn.model_selection import train_test_split

# Note that we are using this for now because it uses average summary length
# These numbers were on the slides and that's what David used.
datasets_mapping_whitespace = {
    # uses mean number of tokens from whitespace tokenization for all documents. 
    # "big_patent": ("description", "abstract"),
    "arxiv": ("ccdv/arxiv-summarization", "article", "abstract", "", 279),

    "billsum": ("billsum", "text", "summary", "", 179),

    "cnn": ("cnn_dailymail", "article", "highlights", "3.0.0", 51),
    "cnn_dailymail": ("cnn_dailymail", "article", "highlights", "3.0.0", 51),

    "gov": ("ccdv/govreport-summarization", "report", "summary", "", 477),
    "govreport": ("ccdv/govreport-summarization", "report", "summary", "", 477),

    "pubmed": ("ccdv/pubmed-summarization", "article", "abstract", "", 202),

    "reddit_tifu": ("reddit_tifu", "documents", "tldr", "long", 20),
    "tifu": ("reddit_tifu", "documents", "tldr", "long", 20),
}

datasets_mapping_nltk = {
    # uses mean number of tokens from nltk word tokenize for all documents. 
    # "big_patent": ("description", "abstract"),
    "arxiv": ("ccdv/arxiv-summarization", "article", "abstract", "", 293),

    "billsum": ("billsum", "text", "summary", "", 204),

    "cnn": ("cnn_dailymail", "article", "highlights", "3.0.0", 56),
    "cnn_dailymail": ("cnn_dailymail", "article", "highlights", "3.0.0", 56),

    "gov": ("ccdv/govreport-summarization", "report", "summary", "", 546),
    "govreport": ("ccdv/govreport-summarization", "report", "summary", "", 546),

    "pubmed": ("ccdv/pubmed-summarization", "article", "abstract", "", 204),

    "reddit_tifu": ("reddit_tifu", "documents", "tldr", "long", 23),
    "tifu": ("reddit_tifu", "documents", "tldr", "long", 23),
}

datasets_mapping = {
    # uses mean number of tokens from nltk word tokenize of a sample of 1000 documents. 
    # this sample uses a length cutoff for some datasets
    # "big_patent": ("description", "abstract"),
    "arxiv": ("ccdv/arxiv-summarization", "article", "abstract", "", 200),

    "billsum": ("billsum", "text", "summary", "", 197),

    "cnn": ("cnn_dailymail", "article", "highlights", "3.0.0", 56),
    "cnn_dailymail": ("cnn_dailymail", "article", "highlights", "3.0.0", 56),

    "gov": ("ccdv/govreport-summarization", "report", "summary", "", 542),
    "govreport": ("ccdv/govreport-summarization", "report", "summary", "", 542),

    "pubmed": ("ccdv/pubmed-summarization", "article", "abstract", "", 215),

    "reddit_tifu": ("reddit_tifu", "documents", "tldr", "long", 23),
    "tifu": ("reddit_tifu", "documents", "tldr", "long", 23),
}

def load_dataset_huggingface(dataset: str):
    (
        dataset_name,
        document_key,
        summary_key,
        config,
        length,
    ) = datasets_mapping_whitespace[dataset] # datasets_mapping[dataset]

    # TODO: Don't worry about this now, but we want to download a specific split.
    # I had some weird stuff happen so just keep this like so for now. NOT A PRIORITY
    if config:
        # data = load_dataset(dataset_name, config, download_mode="force_redownload")
        data = load_dataset(dataset_name, config)

    else:
        # data = load_dataset(dataset_name, download_mode="force_redownload")
        data = load_dataset(dataset_name)

    if dataset_name == "reddit_tifu":
        (
            train_doc,
            test_doc,
            train_sum,
            test_sum,
        ) = train_test_split(
            data["train"][document_key],
            data["train"][summary_key],
            #TODO: PRIORITY. Make sure this runs and pick a value for test_size
            # based on test_size you see in slides
            test_size=0.1,
            random_state=42,
        )
        return test_doc, test_sum, length

    return data["test"][document_key], data["test"][summary_key], length


