from optimization_based.summarizers import (
    LeadSummarizer,
    RandomSummarizer,
    TextRankSummarizer,
    LexRankSummarizer,
    AbstractSummarizer,
)

lead_summarizer = LeadSummarizer()
random_summarizer = RandomSummarizer()
textrank_summarizer = TextRankSummarizer()
lexrank_summarizer = LexRankSummarizer()
abstract_summarizer = AbstractSummarizer