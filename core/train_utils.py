import copy
from typing import Dict

import numpy as np
import torch
from core.data_format import DataStatistics


class EpochManager:
    def __init__(self, log_path: str = None):
        self.log_path = log_path

    def __enter__(self):
        self._step = 0
        self._last_log = ''
        self._loss_dict = dict()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.log_path:
            self.write_log(self._last_log + "\n")
        self._step = 0
        self._last_log = ''
        self._loss_dict = dict()

    def update(self, one_step_loss_dict: Dict[str, float]):
        for key, value in one_step_loss_dict.items():
            if key in self._loss_dict:
                i = self._step
                p_loss = self._loss_dict[key]
                c_loss = value.detach().cpu().numpy()
                loss = (p_loss * (i + 1) + c_loss) / (i + 2)
            else:
                loss = value.detach().cpu().numpy()
            self._loss_dict.update({key: loss})
        self._step += 1

    def get_log(self, prefix: str = ''):
        log = copy.copy(prefix)
        for key, value in self._loss_dict.items():
            log += f'{key}: {value:.6f} '
        self._last_log = copy.copy(log)
        return log

    def write_log(self, log: str):
        with open(self.log_path, 'a') as f:
            f.write(log)

    def get_avg_losses(self):
        return self._loss_dict


class CheckpointManager:
    def __init__(self, ckpt_path, ckpt_last_path):
        self.ckpt_path = ckpt_path
        self.ckpt_last_path = ckpt_last_path
        self._last_loss = np.inf

    def save(self, model, optimizer, epoch, loss_dict):
        loss = loss_dict['total_loss']
        print("======= Loss ======")
        print("now: ", loss, " / pre_best: ", self._last_loss)
        print("===================")
        if loss < self._last_loss:
            self._last_loss = loss
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'model_data_statistics': model._data_statistics,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss_dict,
                },
                self.ckpt_path,
            )

        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'model_data_statistics': model._data_statistics,
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss_dict,
            },
            self.ckpt_last_path,
        )

    def load(self, model, optimizer=None, device="cuda"):
        checkpoint = torch.load(self.ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        device = torch.device(device)
        model.to(device)
        stats = DataStatistics(
            is_sim=True,
            action_max=checkpoint['model_data_statistics']['action_max'].detach().cpu().numpy(),
            action_min=checkpoint['model_data_statistics']['action_min'].detach().cpu().numpy(),
        )
        model.set_data_statistics(stats)

        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epoch']
            loss_dict = checkpoint['loss']
            self._last_loss = loss_dict['total_loss']
            return model, optimizer, epoch, loss_dict
        else:
            return model
