import os.path
import json
import pandas as pd


def json_to_csv(evaluation_output_dir):
    metric_names = ['sacrebleu', 'bleu', 'bertscore', 'rouge']

    # Stores: {dataset: {model : result, }, dataset2 :
    results = {}

    for dataset in sorted(os.listdir(evaluation_output_dir)):
        path_to_model = os.path.join(evaluation_output_dir, dataset)
        results[dataset] = {}
        for model in sorted(os.listdir(path_to_model)):
            path_to_output = os.path.join(evaluation_output_dir, dataset, model, 'evaluation_test.json')
            results[dataset][model] = json.load(open(path_to_output))

    dataset_names = list(results.keys())
    model_names = list(results[dataset_names[0]].keys())
    # print(model_names)
    result_list = []

    for metric in metric_names:
        for model in sorted(model_names):
            row = [metric, model]
            for dataset in sorted(dataset_names):
                print(dataset, model)
                if metric == 'rouge':
                    values = []
                    for variants in results[dataset][model][metric]:
                        values.append(results[dataset][model][metric][variants])
                    row.append(values)
                else:
                    if metric in results[dataset][model].keys():
                        row.append(results[dataset][model][metric])
                    else:
                        row.append(0)
            result_list.append(row)
    print(result_list)

    result_df = pd.DataFrame(result_list)
    # result_df.columns = ['Metric', 'Model'] + dataset_names
    result_df.to_csv('Summarization_scores.csv', index=False, header=True)


if __name__ == '__main__':
    output_dir = '/content/drive/MyDrive/NU-NLP/evaluation_outputs/'
    json_to_csv(output_dir)