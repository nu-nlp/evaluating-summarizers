# evaluating-summarizers

A project to compare neural abstractive and non-neural/optimization_based extractive summarizers.

## 1. Setup

###  Connect to [the shared google drive](https://drive.google.com/drive/folders/1DEPi12LsAozAQeNym5UVVj3i_6_AL5_X?usp=sharing)

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

## 2. Summarize with optimization based extractive models

### Run the summarization code
1. Run a summarizer on a dataset and save the outputs to a csv with the following command:
```python3
python run_summarization.py --dataset cnn_dailymail --summarizer TextRank --summarizations-dir summarization_outputs

# Use the debug flag to restrict dataset to 5 samples.
python run_summarization.py --dataset cnn_dailymail --summarizer TextRank --summarizations-dir summarization_outputs --debug
```
2. Pay attention to where the outputs are saved. The default path is `evaluating-summarizers/optimization_based/summarization_outputs`. In this directory, you will find a the directory structure: `dataset/summarizer`. If an output csv was created in `--debug` mode, you will see a 5 line csv called `summarization_test_debug`. If an output csv is created, normally, the file is called `summarization_test.csv`. This is not that practical but it will do for now.

3. Put your output CSVs in the same place. In other words, copy `evaluating-summarizers/optimization_based/summarization_outputs` to Google Drive or something. I recommend you use: [the shared NU-NLP/summarization_outputs Google Drive](https://drive.google.com/drive/folders/1yDzktsBUhMsS8vzWREKk54XO34ljQGin?usp=sharing) so that others can access the files. We don't want people running the same experiment twice.

### Summarization arguments

* `--dataset`: The dataset. Values currently supported: `cnn_dailymail`, `arxiv`, `pubmed`, `reddit_tifu`, `billsum`, `govreport`
* `--summarizer`: The summarization model. Values currently supported: `LexRank`, `TextRank`, `Random`, `Lead`, `Occams`
* `--summarizations-dir`: The directory path for the summarization output csv files. Default is `summarization_outputs` and that is why we have `--dataset` and `--summarizer` as input. In practice this will probably be the relative path of the [shared NU-NLP/summarization_outputs directory](https://drive.google.com/drive/folders/1yDzktsBUhMsS8vzWREKk54XO34ljQGin?usp=sharing)
* `--debug`: include this boolean flag to limit the dataset to the first 5 documents. This will keep the summarization short and help you iterate quicker when developing.
```python3
python run_summarization.py --dataset cnn_dailymail --summarizer TextRank --debug
```

## 3. Run evaluation scripts on any summarizer output (neural and optimziation based)
### Run the evaluation code
Run evaluation on a summarizer's outputs for a given dataset and save the outputs to a json with the following commands:  
```python3
# For the extractive models:
python evaluate.py --summarizations-dir optimization_based/summarization_outputs --scores-dir evaluation/evaluation_outputs --dataset billsum --summarizer TextRank --metrics bleu sacrebleu rouge bertscore jensen_shannon avg_gen_length

# For the neural models:
python evaluate.py --summarizations-dir optimization_based/summarization_outputs --scores-dir evaluation/evaluation_outputs --dataset billsum --summarizer bartbase --metrics bleu sacrebleu rouge bertscore jensen_shannon avg_gen_length
```
To test if the code works, use the `--debug` flag to save time. It will run the code on 5 samples:
```python3
python evaluate.py --summarizations-dir optimization_based/summarization_outputs --scores-dir evaluation/evaluation_outputs --dataset billsum --summarizer TextRank --metrics bleu sacrebleu rouge bertscore jensen_shannon avg_gen_length --debug
```

### Understanding the evaluation script arguments

* `--dataset`: The dataset. Values currently supported: `cnn_dailymail`, `arxiv`, `pubmed`, `reddit_tifu`, `billsum`, `govreport`
```python3
python evaluate.py --dataset cnn_dailymail --summarizer TextRank --metrics bleu
```
* `--summarizer`: The summarization model. Values currently supported: `LexRank`, `TextRank`, `Random`, `Lead`, `Occams` and the neural models. For the summarizers, dataset and summarizer are used to find the output files. These arguments also help name the output json. Your code won't break if you put a neural model here, so long as you make sure the output csv for your model can be read. (see description of `--summarizations-dir`) 
Here is an example for `billsum_bartbase_197.csv`:
```python3
python evaluate.py --dataset billsum --summarizer bartbase
```
* `--summarizations-dir`: The directory path for the summarization output csv files. Default is `evaluating-summarizers/optimization_based/summarization_outputs` and that is why we have `--dataset` and `--summarizer` as input. In practice this will probably be the relative path of the [shared NU-NLP/summarization_outputs directory](https://drive.google.com/drive/folders/1yDzktsBUhMsS8vzWREKk54XO34ljQGin?usp=sharing)
* `--scores-dir`: The directory path for the scores output json files. Default is `evaluating-summarizers/evaluation/evaluation_outputs`.  You should probably make sure this gets saved in [the shared NU-NLP/evaluation_outputs directory](https://drive.google.com/drive/folders/1thiUxz5DbP2-3SIWegHBJcEHsJ6WTpRl?usp=sharing)
* `--summary-column`: in your summarization output csv file, what is the column header for output summaries. Default value is `"summary"`.
* `--target-column`: in your summarization output csv file, what is the column header for target summaries. Default value is `"target"`. For David's `"label"` column, set this to `"label"`  
* `--metrics`: The list of metrics you want to compute. Metrics supported are `bleu`, `sacrebleu`, `rouge`, `bertscore`, `jensen_shannon`, `avg_gen_length`. Default is `bleu`, `sacrebleu`, `rouge`, `jensen_shannon`, `avg_gen_length` because `bertscore` can be slow without GPU. Here is an example:
* `--debug`: include this boolean flag to limit the output summaries to the first 5 documents. This will keep the evaluation short and help you iterate 


### Some steps to consider before you run the code:
1. Pay attention to where the summarization outputs are saved. The default path is `evaluating-summarizers/optimization_based/summarization_outputs`. In this directory, you will find a the directory structure: `dataset/summarizer`. If an output csv was created in `--debug` mode, you will see a 5 line csv called `summarization_test_debug`. If an output csv is created, normally, the file is called `summarization_test.csv`. This is not that practical but it will do for now.
2. Put yout output CSVs in the same place. In other words, copy `evaluating-summarizers/optimization_based/summarization_outputs` to Google Drive or something. I recommend you use: [shared NU-NLP/summarization_outputs directory](https://drive.google.com/drive/folders/1yDzktsBUhMsS8vzWREKk54XO34ljQGin?usp=sharing) so that others can access the files. We don't want people running the same experiment twice. 
3. Find your `--summarizations-dir`. This should be the path of the summarization outputs relative to the evaluate.py script. Default is `evaluating-summarizers/optimization_based/summarization_outputs`.
4. Choose your `--scores-dir`. This should be the path of the scores relative to the evaluate.py script. Default is `evaluating-summarizers/evaluation/evaluation_outputs`. Scores for an experiment will be saved in a directory `evaluation_outputs/dataset/summarizer`. An output json will be created. If you run in `--debug` mode, you will see a json called `evaluation_test_debug.json`. In regular mode, the file is called `evaluation_test.json`. 
5. Clone this repo and run this script on google colab or a GPU machine. Make sure you save the results json somewhere where it will stay! So if you mount your google drive make sure the output path is [the shared NU-NLP Google Drive](https://drive.google.com/drive/folders/1DEPi12LsAozAQeNym5UVVj3i_6_AL5_X). If you run this on a remote David machine, save the json locally and push it to github. But you will need to move the summarization outputs onto David's remote machine.
