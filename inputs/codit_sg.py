import glob
import os
import random
from typing import List, Union

import gin
import h5py
import numpy as np
import torch
from .base import BaseInput
from .registry import register

from core.data_format import DataStatistics


@register('codit_sg')
@gin.configurable(denylist=['data_root', 'split', 'shuffle'])
class CoDiTSGInput(BaseInput):
    def __init__(
        self,
        data_root: str,
        data_rel_dir: str,
        split: str = 'train',
        task_list: Union[str, List[str]] = 'all',
        num_train_demo: Union[str, int] = 'all',
        num_val_demo: Union[str, int] = 'all',
        shuffle: bool = True,
        chunk_size: int = 50,
        sample_full_episode: bool = False,
        duplicate_num: int = 1,
    ):
        self.data_dir = os.path.join(data_root, data_rel_dir)
        self.split = split
        self.task_list = task_list
        self.num_train_demo = num_train_demo
        self.num_val_demo = num_val_demo
        self.shuffle = shuffle
        self.chunk_size = chunk_size
        self.sample_full_episode = sample_full_episode
        self.h5df_file_list = self._get_h5df_list() * duplicate_num
        if self.shuffle:
            random.shuffle(self.h5df_file_list)

    def __len__(self):
        return len(self.h5df_file_list)

    def __getitem__(self, idx):
        data_output = self._read_h5df(self.h5df_file_list[idx])
        return data_output

    def _get_h5df_list(self):
        file_list = []
        if self.task_list == 'all':
            task_list = os.listdir(self.data_dir)
        else:
            task_list = self.task_list

        for task in task_list:
            h5path = os.path.join(self.data_dir, task, self.split, '*.hdf5')
            h5files = sorted(glob.glob(h5path))

            if self.split == 'train' and self.num_train_demo != 'all':
                h5files = h5files[: self.num_train_demo]
            elif self.split == 'val' and self.num_val_demo != 'all':
                h5files = h5files[: self.num_val_demo]
            else:
                h5files = h5files

            if len(h5files) == 0:
                raise Exception(f"Data not exist: {h5path}")
            file_list += h5files
        return file_list

    def _read_h5df(self, hdf5_file):
        with h5py.File(hdf5_file, 'r') as root:
            episode_len = root.attrs['episode_len']
            sentence_embedding = root['sentence_embedding'][:]

            if self.sample_full_episode:
                start_ts = 0
            else:
                start_ts = np.random.choice(episode_len)

            qpos = []
            qpos_keys = sorted(list(root['qpos'].keys()))
            for key in qpos_keys:
                qpos.append(root['qpos'][key][start_ts])
            qpos = np.concatenate(qpos, axis=0)

            qvel = []
            qvel_keys = sorted(list(root['qvel'].keys()))
            for key in qvel_keys:
                qvel.append(root['qvel'][key][start_ts])
            qvel = np.concatenate(qvel, axis=0)

            images = []
            images_keys = sorted(list(root['images'].keys()))
            for key in images_keys:
                images.append(root['images'][key][start_ts])
            images = np.stack(images, axis=0)

            actions_full = []
            actions_keys = sorted(list(root['actions'].keys()))
            for key in actions_keys:
                actions_full.append(root['actions'][key][start_ts : start_ts + self.chunk_size])
            actions_full = np.concatenate(actions_full, axis=1)
            action_len = actions_full.shape[0]

            original_action_shape = (self.chunk_size,) + actions_full.shape[1:]
            padded_action = np.zeros(original_action_shape, dtype=np.float32)
            padded_action[:action_len] = actions_full
            padded_action[action_len:] = actions_full[-1]
            is_pad = np.zeros(self.chunk_size)
            is_pad[action_len:] = 1

            image_data = torch.from_numpy(images)
            qpos_data = torch.from_numpy(qpos).float()
            action_data = torch.from_numpy(padded_action).float()
            is_pad = torch.from_numpy(is_pad).bool()
            context_data = torch.from_numpy(sentence_embedding).float()

            image_data = image_data / 255.0
            image_data = torch.einsum('k h w c -> k c h w', image_data)

            episode_num = int(hdf5_file.split('/')[-1].split('.')[0].split('_')[1])
            file_name = 'data_' + 'ep' + str(episode_num) + '_s' + str(start_ts) + '.npz'
            file_path = os.path.join('control_factor_data', file_name)
            data = np.load(file_path)
            control_factor = data['control_factors_norm']
            actions_gen = data['actions_gen']
            ind = np.random.choice(control_factor.shape[0])
            control_factor_data = torch.from_numpy(control_factor[ind : ind + 1, 0]).float()
            actions_gen_data = torch.from_numpy(actions_gen[ind, :, :]).float()

            output_data = {
                'images': image_data,
                'contexts': context_data,
                'qpos': qpos_data,
                'actions': actions_gen_data,
                'is_pad': is_pad,
                'control_factor': control_factor_data,
            }

        return output_data

    def get_statistics(self):
        h5df_file_list = self._get_h5df_list()
        all_qpos = []
        all_action = []
        for h5df_file in h5df_file_list:
            with h5py.File(h5df_file, 'r') as root:
                qpos = []
                for key in root['qpos'].keys():
                    qpos.append(root['qpos'][key][:])
                qpos = np.concatenate(qpos, axis=1)

                action = []
                for key in root['actions'].keys():
                    action.append(root['actions'][key][:])
                action = np.concatenate(action, axis=1)

                all_qpos.append(qpos)
                all_action.append(action)

        all_qpos = np.concatenate(all_qpos, axis=0)
        all_action = np.concatenate(all_action, axis=0)

        action_mean = np.mean(all_action, axis=0, keepdims=True)
        action_std = np.std(all_action, axis=0, keepdims=True)
        action_std = np.clip(action_std, 1e-2, 10)  # clipping
        action_max = np.max(all_action, axis=0, keepdims=True)
        action_min = np.min(all_action, axis=0, keepdims=True)

        # normalize qpos data
        qpos_mean = np.mean(all_qpos, axis=0, keepdims=True)
        qpos_std = np.std(all_qpos, axis=0, keepdims=True)
        qpos_std = np.clip(qpos_std, 1e-2, 10)  # clipping
        qpos_max = np.max(all_qpos, axis=0, keepdims=True)
        qpos_min = np.min(all_qpos, axis=0, keepdims=True)

        stats = DataStatistics(
            is_sim=True,
            action_mean=action_mean.squeeze(),
            action_std=action_std.squeeze(),
            action_max=action_max.squeeze(),
            action_min=action_min.squeeze(),
            qpos_mean=qpos_mean.squeeze(),
            qpos_std=qpos_std.squeeze(),
            qpos_max=qpos_max.squeeze(),
            qpos_min=qpos_min.squeeze(),
        )
        return stats
