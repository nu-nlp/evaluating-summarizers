from evaluation.metrics import (
    BleuMetric,
    SacrebleuMetric,
    RougeMetric,
    BertscoreMetric,
    JensenShannonMetric
    # MauveMetric
)

bleu_metric = BleuMetric()
sacrebleu_metric = SacrebleuMetric()
rouge_metric = RougeMetric()
bertscore_metric = BertscoreMetric()
jensen_shannon_metric = JensenShannonMetric()
# mauve_metric = MauveMetric()
