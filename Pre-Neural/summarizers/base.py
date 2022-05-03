from abc import ABCMeta, abstractmethod


class AbstractSummarizer(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self):
        self.model_name = None

    @abstractmethod
    def get_summary(text: str, length: int):
        return NotImplementedError
