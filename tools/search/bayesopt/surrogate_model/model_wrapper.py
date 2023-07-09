from abc import abstractmethod
import torch

class BaseModel:
    @abstractmethod
    def fit(self, indices: list, scores: list):
        raise NotImplementedError

    @abstractmethod
    def predict(self, eval_X: torch.Tensor, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def acquisition(self, eval_X: torch.Tensor, acq_func='expected_improvement', bias=None, **kwargs):
        raise NotImplementedError
