from abc import ABCMeta, abstractmethod
from typing import List


class AbstractMetric(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self):
        self.metric_name = None

    @abstractmethod
    def evaluate(predictions: List[str], references: List[str]):
        return NotImplementedError
