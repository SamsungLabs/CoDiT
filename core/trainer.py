import os
from typing import Optional

import gin
import numpy as np
import torch

from tqdm import tqdm
from torch.utils.data import DataLoader

from optimizers import TorchOptimizerBuilder
from common import nested_tensor_utils
from core.train_utils import EpochManager

from core import Task

from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from common.distributed_utils import is_main_process


@gin.configurable(denylist=['task'])
class Trainer:
    def __init__(
        self,
        task: Task,
        gpu,
        num_workers,
        train_batch_size: int = 8,
        val_batch_size: int = 1,
        train_epoch: int = 1992,
        val_epoch_interval: int = 3,
        seed: Optional[int] = None,
    ):
        self.task = task
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.train_epoch = train_epoch
        self.val_epoch_interval = val_epoch_interval
        self.seed = seed
        self.gpu = gpu
        self.num_workers = num_workers
        if self.seed is not None:
            self.set_seed(self.seed)

    def build_optimizer(self, model):
        builder = TorchOptimizerBuilder()
        optimizer = builder.build(model.parameters())
        return optimizer

    def set_seed(self, seed):
        torch.manual_seed(seed)
        np.random.seed(seed)

    def train(self):
        if not torch.cuda.is_available():
            raise Exception("Only cuda device is supported for training phase")
        else:
            device = torch.device("cuda")

        exp_dir = self.task.exp_dir
        log_file = os.path.join(exp_dir, 'loss.log')

        train_ds, val_ds = self.task.get_dataset()

        model = self.task.get_model()

        data_statistics = train_ds.get_statistics()
        model.set_data_statistics(data_statistics)

        # model = model.to(device)
        model.cuda(self.gpu)
        optimizer = self.build_optimizer(model)

        train_sampler = DistributedSampler(train_ds)
        val_sampler = DistributedSampler(val_ds)

        train_dataloader = DataLoader(
            train_ds,
            batch_size=self.train_batch_size,
            pin_memory=True,
            num_workers=self.num_workers,
            prefetch_factor=2,
            sampler=train_sampler,
        )

        val_dataloader = DataLoader(
            val_ds,
            batch_size=self.val_batch_size,
            pin_memory=True,
            num_workers=self.num_workers,
            prefetch_factor=2,
            sampler=val_sampler,
        )

        if self.task.is_ckpt_exist():
            model, optimizer, saved_epoch, losses = self.task.load_training_ckpt(
                model, optimizer, self.gpu
            )
            init_loss = losses['total_loss']
            print(f'ckpt epoch: {saved_epoch}, ckpt total_loss: {init_loss:.6f}')
            start_epoch = saved_epoch + 1
        else:
            start_epoch = 0

        model = DistributedDataParallel(model, device_ids=[self.gpu])

        train_total = 0
        val_total = 0
        epoch_manager = EpochManager(log_path=log_file)
        for epoch in range(start_epoch, self.train_epoch):
            model.train()
            with epoch_manager as em:
                if is_main_process():
                    if epoch == start_epoch:
                        pbar = tqdm(train_dataloader)
                    else:
                        pbar = tqdm(train_dataloader, total=train_total)
                else:
                    pbar = train_dataloader

                for data in pbar:
                    data = nested_tensor_utils.to_device(data, device=device)
                    one_step_losses = model.module.train_one_step(data, optimizer)
                    em.update(one_step_losses)

                    if is_main_process():
                        prefix = f'[Train epoch {epoch}] '
                        log = em.get_log(prefix)
                        pbar.set_description(log)

                        if epoch == start_epoch:
                            train_total += 1

                losses = em.get_avg_losses()

            if epoch % self.val_epoch_interval == 0:
                with torch.inference_mode():
                    model.eval()
                    with epoch_manager as em:
                        if is_main_process():
                            if epoch == start_epoch:
                                pbar = tqdm(val_dataloader)
                            else:
                                pbar = tqdm(val_dataloader, total=val_total)
                        else:
                            pbar = train_dataloader

                        for data in pbar:
                            data = nested_tensor_utils.to_device(data, device=device)
                            one_step_losses = model.module.val_one_step(data)
                            em.update(one_step_losses)

                            if is_main_process():
                                prefix = f'[Val epoch {epoch}] '
                                log = em.get_log(prefix)
                                pbar.set_description(log)

                                if epoch == start_epoch:
                                    val_total += 1

                        losses = em.get_avg_losses()

                if is_main_process():
                    self.task.save_training_ckpt(model.module, optimizer, epoch, losses)
