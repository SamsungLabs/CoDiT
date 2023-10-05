from abc import *

import torch


class BaseModel(torch.nn.Module):
    @abstractmethod
    def forward(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def set_data_statistics(self):
        raise NotImplementedError

    @abstractmethod
    def train_one_step(self, data, optimizer):
        raise NotImplementedError

    @abstractmethod
    def val_one_step(self, data):
        raise NotImplementedError
