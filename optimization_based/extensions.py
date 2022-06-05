from optimization_based.summarizers import (
    LeadSummarizer,
    RandomSummarizer,
    TextRankSummarizer,
    LexRankSummarizer,
    AbstractSummarizer,
    OccamsSummarizer
)

lead_summarizer = LeadSummarizer()
random_summarizer = RandomSummarizer()
textrank_summarizer = TextRankSummarizer()
lexrank_summarizer = LexRankSummarizer()
occams_summarizer = OccamsSummarizer()
abstract_summarizer = AbstractSummarizer