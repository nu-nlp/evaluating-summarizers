# evaluating-summarizers

A project to compare neural abstractive and non-neural/optimization_based extractive summarizers.

## 1. Setup

###  Setup your python environment for neural models summarization
1. **Navigate into the directory `neural`**
2. Create a miniconda environment and install the requirements.txt
```
conda create -n huggingface python=3.7
conda activate huggingface
pip3 install -r requirements.txt
```

###  Setup for optmization based summarization and evaluation
1. **Navigate into the directory `optimization_based`**
2. Create a miniconda environment and install the requirements.txt

```
conda create -n extractive python=3.7
conda activate extractive
pip3 install -r requirements.txt
```

###  Connect to [the shared google drive](https://drive.google.com/drive/folders/1DEPi12LsAozAQeNym5UVVj3i_6_AL5_X?usp=sharing) (*Optional*)

## 2. Summarize with Pre-Neural Extractive Models

### Run the summarization code
You can run a summarizer on a dataset and save the outputs to a csv with the following command:
```python3
python run_preneural_summarization.py --training_dataset arxiv --evaluation_dataset arxiv --target_length 200 --summarizer TextRank --summarizations_dir evaluation/generated_summaries

# Use the debug flag to restrict dataset to 5 samples.
python run_preneural_summarization.py --training_dataset arxiv --evaluation_dataset arxiv --target_length 200 --summarizer TextRank --summarizations_dir evaluation/generated_summaries --debug
```
Note that by default, the generated summaries are saved to: `evaluating-summarizers/evaluation/generated_summaries`.

### Summarization arguments

* `--evaluation_dataset`: The dataset for which summaries will be generated. Values currently supported: `cnn`, `arxiv`, `pubmed`, `tifu`, `billsum`, `gov`
* `--training_dataset`: The dataset which a method could be "trained" against. Will almost always be set to the same dataset as `--evaluation_dataset`
* `--summarizer`: The summarization model. Values currently supported: `LexRank`, `TextRank`, `Random`, `Lead`, `Occams`
* `--target_length`: The number of tokens the summarizer should aim to generate for each summary.
* `--summarizations_dir`: The directory path for the summarization output csv files. Default argument is `evaluation/generated_summaries`
* `--debug`: Include this boolean flag to limit the dataset to the first 5 documents. This will keep the summarization short and help you iterate quicker when developing.

## 3. Run Evaluation Scripts on any Summarizer Output *(pre-neural and neural)*
### Run the evaluation code
Run evaluation on a summarizer's outputs for a given dataset and save the outputs to a json with the following commands:  
```python3
python evaluate.py --summarizations_dir evaluation/generated_summaries --results_dir evaluation/evaluation_results --training_dataset arxiv --evaluation_dataset arxiv --summarizer TextRank --target_length 200 --metrics bleu sacrebleu rouge bertscore jensen_shannon avg_gen_length

# Use the debug flag to restrict evaluation to 5 samples.
python evaluate.py --summarizations_dir evaluation/generated_summaries --results_dir evaluation/evaluation_results --training_dataset arxiv --evaluation_dataset arxiv --summarizer TextRank --target_length 200 --metrics bleu sacrebleu rouge bertscore jensen_shannon avg_gen_length--debug
```

### Understanding the evaluation script arguments

* `--training_dataset`: The dataset on which the model/method was trained. Values currently supported: `arxiv`, `billsum`, `cnn`, `gov`, `pubmed`, `tifu` 
* `--evaluation_dataset`: The dataset on for which the summaries were generated. Values currently supported: `arxiv`, `billsum`, `cnn`, `gov`, `pubmed`, `tifu`
* `--summarizer`: The summarization model. Values currently supported: `LexRank`, `TextRank`, `Random`, `Lead`, `Occams`, `bartbase`, `bartlarge`, `t5small`, `t5base`, `pegasuslarge`, `pegasusxsum` *(note that any summarizer name will work, as long as the corresponding csv file exists)*
* `--summarizations_dir`: The directory path for the summarization output csv files. Default is `evaluation/generated_summaries`
* `--results_dir`: The directory path for the evaluation results output json files. Default is `evaluation/evaluation_results`
* `--summary-column`: in your summarization output csv file, what is the column header for output summaries. Default value is `"summary"`
* `--target-column`: in your summarization output csv file, what is the column header for target summaries. Default value is `"target"`  
* `--metrics`: The list of metrics you want to compute. Metrics supported are `bleu`, `sacrebleu`, `rouge`, `bertscore`, `jensen_shannon`, `avg_gen_length`. Default is `bleu`, `sacrebleu`, `rouge`, `jensen_shannon`, `avg_gen_length` because `bertscore` can be slow without GPU.
* `--debug`: include this boolean flag to limit the output summaries to the first 5 documents. This will keep the evaluation short and help you iterate 


### Some steps to consider before you run the code:
1. Ensure that the generated summary files are in the same place, as this will make evaluation much simpler. We recommend using the default directory `evaluation/generated_summaries`
2. Generated summaries will have a standardized naming convention: `<training_dataset>_<evaluation_dataset>_<summarizer>_<target_length>.csv`. For example, generating summaries with a `bartbase` model fine-tuned on `arxiv` and evaluated on `cnn` with a target length of `56` would be saved as follows: `arxiv_cnn_bartbase_56.csv`
3. Clone this repo and run the scripts on a machine with a GPU (such as Google Colab or a remote server). Make sure you save the results json somewhere where it will stay! So, if you mount your Google Drive make sure the output path is [the shared NU-NLP Google Drive](https://drive.google.com/drive/folders/1DEPi12LsAozAQeNym5UVVj3i_6_AL5_X). If you run this on a remote server, save the json locally and upload it to the shared Google Drive.
