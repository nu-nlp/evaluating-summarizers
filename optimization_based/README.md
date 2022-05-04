##  Setup

1. Navigate into the directory `optimization_based`
2. Create a miniconda environment and install requirements.txt

```
conda create -n extractive python=3.7
conda activate extractive
pip3 install -r requirements.txt
```

## Run the summarization code
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
* `--summarizer`: The summarization model. Values currently supported: `LexRank`, `TextRank`, `Random`, `Lead`
* `--summarizations-dir`: The directory path for the summarization output csv files. Default is `summarization_outputs` and that is why we have `--dataset` and `--summarizer` as input. In practice this will probably be the relative path of the [shared NU-NLP/summarization_outputs directory](https://drive.google.com/drive/folders/1yDzktsBUhMsS8vzWREKk54XO34ljQGin?usp=sharing)
* `--debug`: include this boolean flag to limit the dataset to the first 5 documents. This will keep the summarization short and help you iterate quicker when developing.
```python3
python run_summarization.py --dataset cnn_dailymail --summarizer TextRank --debug
```

