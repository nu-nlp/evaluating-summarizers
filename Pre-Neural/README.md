Put pre-neural stuff here.

##  Setup

1. Navigate into the directory `Pre-Neural`
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
2. Pay attention to where the outputs are saved. The default path is `summarization-olympics/Pre-neural/summarization_outputs`. In this directory, you will find a the directory structure: `dataset/summarizer`. If an output csv was created in `--debug` mode, you will see a 5 line csv called `summarization_test_debug`. If an output csv is created, normally, the file is called `summarization_test.csv`. This is not that practical but it will do for now.

3. Put your output CSVs in the same place. In other words, copy `summarization-olympics/Pre-neural/summarization_outputs` to Google Drive or something. I recommend you use: [the shared NU-NLP/summarization_outputs Google Drive](https://drive.google.com/drive/folders/1yDzktsBUhMsS8vzWREKk54XO34ljQGin?usp=sharing) so that others can access the files. We don't want people running the same experiment twice.

### Summarization arguments

* `--dataset`: The dataset. Values currently supported: `cnn_dailymail`, `arxiv`, `pubmed`, `reddit_tifu`, `billsum`, `govreport`
* `--summarizer`: The summarization model. Values currently supported: `LexRank`, `TextRank`, `Random`, `Lead`
* `--summarizations-dir`: The directory path for the summarization output csv files. Default is `summarization_outputs` and that is why we have `--dataset` and `--summarizer` as input. In practice this will probably be the relative path of the [shared NU-NLP/summarization_outputs directory](https://drive.google.com/drive/folders/1yDzktsBUhMsS8vzWREKk54XO34ljQGin?usp=sharing)
* `--debug`: include this boolean flag to limit the dataset to the first 5 documents. This will keep the summarization short and help you iterate quicker when developing.
```python3
python run_summarization.py --dataset cnn_dailymail --summarizer TextRank --debug
```

## Run the evaluation code
Run a evaluation on a summarizer's outputs for a given dataset and save the outputs to a json with the following commands:  
```python3
# For the extractive models:
python evaluate.py --summarizations-dir summarization_outputs --scores-dir evaluation_outputs --dataset billsum --summarizer TextRank --metrics bleu sacrebleu rouge bertscore

# For the neural models:
python evaluate.py --summarizations-dir summarization_outputs --scores-dir evaluation_outputs --dataset billsum --summarizer bartbase --metrics bleu sacrebleu rouge bertscore 
```
To test if the code works, use the `--debug` flag to save time. It will run the code on 5 samples:
```python3
python evaluate.py --summarizations-dir summarization_outputs --scores-dir evaluation_outputs --dataset billsum --summarizer TextRank --metrics bleu sacrebleu rouge bertscore --debug
```

### Understanding the evaluation script arguments

* `--dataset`: The dataset. Values currently supported: `cnn_dailymail`, `arxiv`, `pubmed`, `reddit_tifu`, `billsum`, `govreport`
```python3
python evaluate.py --dataset cnn_dailymail --summarizer TextRank --metrics bleu
```
* `--summarizer`: The summarization model. Values currently supported: `LexRank`, `TextRank`, `Random`, `Lead` and the neural models. For the summarizers, dataset and summarizer are used to find the output files. These arguments also help name the output json. Your code won't break if you put a neural model here, so long as you make sure the output csv for your model can be read. (see description of `--summarizations-dir`) 
Here is an example for `billsum_bartbase_197.csv`:
```python3
python evaluate.py --dataset billsum --summarizer bartbase
```
* `--summarizations-dir`: The directory path for the summarization output csv files. Default is `summarization_outputs` and that is why we have `--dataset` and `--summarizer` as input. In practice this will probably be the relative path of the [shared NU-NLP/summarization_outputs directory](https://drive.google.com/drive/folders/1yDzktsBUhMsS8vzWREKk54XO34ljQGin?usp=sharing)
* `--scores-dir`: The directory path for the scores output json files. Default is `evaluation_outputs`.  You should probably make sure this gets saved in [the shared NU-NLP/evaluation_outputs directory](https://drive.google.com/drive/folders/1thiUxz5DbP2-3SIWegHBJcEHsJ6WTpRl?usp=sharing)
* `--summary-column`: in your summarization output csv file, what is the column header for output summaries. Default value is `"summary"`.
* `--target-column`: in your summarization output csv file, what is the column header for target summaries. Default value is `"target"`. For David's `"label"` column, set this to `"label"`  
* `--metrics`: The list of metrics you want to compute. Metrics supported are `bleu`, `sacrebleu`, `rouge`, `bertscore`. Default is `bleu`, `sacrebleu`, `rouge` because `bertscore` can be slow without GPU. Here is an example:
* `--debug`: include this boolean flag to limit the output summaries to the first 5 documents. This will keep the evaluation short and help you iterate 


### Some steps before you run the code:
1. Pay attention to where the summarization outputs are saved. The default path is `summarization-olympics/Pre-neural/summarization_outputs`. In this directory, you will find a the directory structure: `dataset/summarizer`. If an output csv was created in `--debug` mode, you will see a 5 line csv called `summarization_test_debug`. If an output csv is created, normally, the file is called `summarization_test.csv`. This is not that practical but it will do for now.
2. Put yout output CSVs in the same place. In other words, copy `summarization-olympics/Pre-neural/summarization_outputs` to Google Drive or something. I recommend you use: [shared NU-NLP/summarization_outputs directory](https://drive.google.com/drive/folders/1yDzktsBUhMsS8vzWREKk54XO34ljQGin?usp=sharing) so that others can access the files. We don't want people running the same experiment twice. 
3. Find your `--summarizations-dir`. This should be the path of the summarization outputs relative to the evaluate.py script. Default is `summarization_outputs`.
4. Choose your `--scores-dir`. This should be the path of the scores relative to the evaluate.py script. Default is `evaluation_outputs`. Scores for an experiment will be saved in a directory `evaluation_outputs/dataset/summarizer`. An output json will be created. If you run in `--debug` mode, you will see a json called `evaluation_test_debug.json`. In regular mode, the file is called `evaluation_test.json`. 
5. Look at the #TODO in the code. I think some of them tell you to decide on details for metric computation. sometimes you need to compute an average. Sometimes you need to pick your key in the results dict. Just do control F TODO and you will see what I mean. If you modify the results json.
    - e.g. bleu is a percentage. sacrebleu is out of 100. So these need to be on the same scale.
    ```json
    {
        "summarizer": "TextRank",
        "dataset": "cnn_dailymail",
        "bleu": {
            "bleu": 0.06235886749706198
        },
        "sacrebleu": {
            "score": 6.364158646509533
        },
    }
    ```
    - e.g. bertscore is computed for each pair (summary, target) but we have no aggregate score. So maybe compute an average. Also pick the keys you care about for bert score. this is just f1. theres precision and recall too.
    ```json
    {
        "summarizer": "TextRank",
        "dataset": "cnn_dailymail",
        "bertscore": {
            "f1": [
                0.8529757261276245,
                0.8617121577262878,
                0.0,
                0.8795979022979736,
                0.8753492832183838
            ]
        }
    }
    ```
    - e.g. rouge similar issue to bertscore.
    ```json
    {
        "summarizer": "TextRank",
        "dataset": "cnn_dailymail",
        "rouge": {
            "rouge1": [
                [
                    0.08155844155844157,
                    0.1626769626769627,
                    0.10848236730589673
                ],
                [
                    0.16482539682539685,
                    0.3371943371943372,
                    0.2202696049470243
                ],
                [
                    0.2244502164502165,
                    0.4545688545688546,
                    0.2988975077210371
                ]
            ],
            "rouge2": [
                [
                    0.00759493670886076,
                    0.016666666666666666,
                    0.010434782608695653
                ],
                [
                    0.050340313664764026,
                    0.10081300813008129,
                    0.06709367926759231
                ],
                [
                    0.09308569062066731,
                    0.18495934959349597,
                    0.12375257592648899
                ]
            ],
            "rougeL": [
                [
                    0.04933333333333333,
                    0.10270270270270272,
                    0.06663614163614165
                ],
                [
                    0.10734199134199134,
                    0.21413127413127414,
                    0.14286276131627176
                ],
                [
                    0.1565887445887446,
                    0.305019305019305,
                    0.20676219205630972
                ]
            ]
        },
    }
    ```
6. Clone this repo and run this script on google colab or a GPU machine. Make sure you save the results json somewhere where it will stay! So if you mount your google drive make sure the output path is [the shared NU-NLP Google Drive](https://drive.google.com/drive/folders/1DEPi12LsAozAQeNym5UVVj3i_6_AL5_X). If you run this on a remote David machine, save the json locally and push it to github. But you will need to move the summarization outputs onto David's remote machine.

