from evaluation.metrics import (
    BleuMetric,
    SacrebleuMetric,
    RougeMetric,
    BertscoreMetric,
    JensenShannonMetric,
    AvgGenLengthMetric
    # MauveMetric
)

bleu_metric = BleuMetric()
sacrebleu_metric = SacrebleuMetric()
rouge_metric = RougeMetric()
bertscore_metric = BertscoreMetric()
jensen_shannon_metric = JensenShannonMetric()
avg_gen_length_metric = AvgGenLengthMetric()
# mauve_metric = MauveMetric()
