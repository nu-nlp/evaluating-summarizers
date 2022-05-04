from evaluation.metrics import (
    BleuMetric,
    SacrebleuMetric,
    RougeMetric,
    BertscoreMetric,
    # MauveMetric
)

bleu_metric = BleuMetric()
sacrebleu_metric = SacrebleuMetric()
rouge_metric = RougeMetric()
bertscore_metric = BertscoreMetric()
# mauve_metric = MauveMetric()
