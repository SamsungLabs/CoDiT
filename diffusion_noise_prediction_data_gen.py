import os
from absl import app, flags

import torch
import gin
from core import Task
import inputs
import h5py
import numpy as np
import time

import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from sklearn.preprocessing import MinMaxScaler
from numba import jit

FLAGS = flags.FLAGS
flags.DEFINE_string('exp_dir', None, 'Path of exp dir')
flags.DEFINE_string(
    'data_root',
    '/home/sr5/dlvr/dataset/srrc',
    'Data root dir will be concatenated with data path in data_config.gin',
)
flags.DEFINE_multi_string('gin_param', None, 'List of Gin parameter bindings')
flags.DEFINE_string('port', '23456', 'TCP Port')


def main(argv):
    os.makedirs('control_factor_data', exist_ok=True)

    exp_dir = FLAGS.exp_dir
    data_root = FLAGS.data_root
    port = FLAGS.port
    data_config = os.path.join(exp_dir, 'data.gin')
    model_config = os.path.join(exp_dir, 'model.gin')
    task_config = os.path.join(exp_dir, 'task.gin')

    gin_files = [data_config, model_config, task_config]
    gin_params = FLAGS.gin_param
    ngpus_per_node = torch.cuda.device_count()

    gin.parse_config_files_and_bindings(gin_files, gin_params)
    task = Task(exp_dir=exp_dir, data_root=data_root)

    train_ds = inputs.make('codit')(
        task.data_root, split='train', shuffle=False, sample_full_episode=True, duplicate_num=1
    )
    train_h5df_file_list = train_ds._get_h5df_list()

    val_ds = inputs.make('codit')(
        task.data_root, split='val', shuffle=False, sample_full_episode=True, duplicate_num=1
    )
    val_h5df_file_list = val_ds._get_h5df_list()
    h5df_file_list = train_h5df_file_list + val_h5df_file_list

    episode_lens = []
    for episode_num in range(len(h5df_file_list)):
        with h5py.File(h5df_file_list[episode_num], 'r') as root:
            episode_len = root.attrs['episode_len']
            episode_lens.append(episode_len)
    interval = sum(episode_lens) // (ngpus_per_node)

    print("==================")
    print("Total Episode Length:", sum(episode_lens))
    print("Episode Length per gpu:", interval)

    ind_set = np.arange(0, len(episode_lens), 1)
    start_ind = []
    start_ind.append(0)
    end_ind = []
    episode_len_check = 0
    for i in range(len(episode_lens)):
        episode_len_check += episode_lens[i]
        if episode_len_check >= interval:
            episode_len_check = 0
            end_ind.append(i + 1)
            start_ind.append(i + 1)
            if len(start_ind) == ngpus_per_node:
                break

    end_ind.append(len(episode_lens))
    start_ind = np.array(start_ind)
    end_ind = np.array(end_ind)

    print("Start Index:", start_ind)
    print("End Index:", end_ind)
    print("Number of episode per gpu:", end_ind - start_ind)

    for i in range(start_ind.shape[0]):
        print("GPU", i, " Epi Length", sum(episode_lens[start_ind[i] : end_ind[i]]))

    mp.spawn(
        main_worker,
        nprocs=ngpus_per_node,
        args=(
            ngpus_per_node,
            exp_dir,
            data_root,
            gin_files,
            gin_params,
            port,
            ind_set,
            start_ind,
            end_ind,
        ),
    )


def main_worker(
    gpu,
    ngpus_per_node,
    exp_dir,
    data_root,
    gin_files,
    gin_params,
    port,
    ind_set,
    start_ind,
    end_ind,
):
    torch.cuda.set_device(gpu)
    if not torch.cuda.is_available():
        raise Exception("Only cuda device is supported.")
    else:
        _device = torch.device("cuda")

    print("Use GPU: {} for training".format(gpu))
    dist.init_process_group(
        backend='nccl',
        init_method='tcp://127.0.0.1:' + port,
        world_size=ngpus_per_node,
        rank=gpu,
    )

    gin.parse_config_files_and_bindings(gin_files, gin_params)

    task = Task(exp_dir=exp_dir, data_root=data_root)
    task.restore_model()

    train_ds = inputs.make('codit')(
        task.data_root, split='train', shuffle=False, sample_full_episode=True, duplicate_num=1
    )
    train_h5df_file_list = train_ds._get_h5df_list()

    val_ds = inputs.make('codit')(
        task.data_root, split='val', shuffle=False, sample_full_episode=True, duplicate_num=1
    )
    val_h5df_file_list = val_ds._get_h5df_list()
    h5df_file_list = train_h5df_file_list + val_h5df_file_list

    model = task.get_model()
    model = DistributedDataParallel(model, device_ids=[gpu])

    scaler = MinMaxScaler()

    ind_set = ind_set[start_ind[gpu] : end_ind[gpu]]

    batch_size = 100
    num_test = 1
    for episode_num in ind_set:
        with h5py.File(h5df_file_list[episode_num], 'r') as root:
            episode_len = root.attrs['episode_len']
        for start_ts in range(episode_len):
            episode_num_true = int(
                h5df_file_list[episode_num].split('/')[-1].split('.')[0].split('_')[1]
            )
            file_name = (
                'control_factor_data/data_'
                + 'ep'
                + str(episode_num_true)
                + '_s'
                + str(start_ts)
                + '.npz'
            )

            if not os.path.isfile(file_name):
                ts = time.time()
                output_data = load_output_data(
                    h5df_file_list, episode_num, start_ts, model.module.chunk_size, _device
                )
                qpos = output_data['qpos'][None]  # (1, 16)
                images = output_data['images'][None]  # (1, 4, 3, 480, 640)

                results = []
                control_factors = []
                with torch.inference_mode():
                    for i in range(num_test):
                        noise = torch.randn(
                            size=(batch_size, model.module.chunk_size, model.module.action_dim),
                            dtype=torch.float,
                            device=_device,
                            generator=None,
                        )
                        result = (
                            model.module(
                                actions=noise,
                                prop=qpos.repeat(batch_size, 1),
                                images=images.repeat(batch_size, 1, 1, 1, 1),
                            )['a_hat']
                            .detach()
                            .cpu()
                            .numpy()
                        )  # (batch_size, chunk_size, action_dim)

                        control_factor = calc_control_factor(result)  # (batch_size,)
                        control_factors.append(control_factor)
                        results.append(result)
                        print(
                            "/ epi:",
                            episode_num_true,
                            "/ start:",
                            start_ts,
                            "/ num_test:",
                            i,
                            "/",
                            num_test,
                        )

                results = np.array(results).reshape(
                    -1, model.module.chunk_size, model.module.action_dim
                )  # (N, chunk_size, action_dim)
                control_factors = np.array(control_factors).reshape(-1)  # (N, )
                control_factors_norm = scaler.fit_transform(control_factors.reshape(-1, 1)) * 2 - 1

                te = time.time()
                print("Time: ", te - ts)

                np.savez_compressed(
                    file_name,
                    actions_gen=results,
                    control_factors=control_factors,
                    control_factors_norm=control_factors_norm,
                )


def calc_control_factor(actions):
    #### actions (bs, chunk_size, action_dim)
    bs = actions.shape[0]
    control_factors = np.zeros((bs,))
    for i in range(bs):
        positions_left = []
        positions_right = []
        for j in range(25):
            position = Franka_forward_kinematics(actions[i, j, :7])[:3, 3]
            positions_left.append(position)
            position = Franka_forward_kinematics(actions[i, j, 8:15])[:3, 3]
            positions_right.append(position)
        positions_left = np.array(positions_left)  # (chunk_size, 3)
        positions_right = np.array(positions_right)  # (chunk_size, 3)

        positions_left_next = np.roll(positions_left, -1, axis=0)
        positions_left_next[-1, :] = positions_left[-1, :]

        positions_right_next = np.roll(positions_right, -1, axis=0)
        positions_right_next[-1, :] = positions_right[-1, :]

        control_factors[i] = np.sum(
            np.linalg.norm(positions_left_next - positions_left, axis=1)
        ) + np.sum(np.linalg.norm(positions_right_next - positions_right, axis=1))

    return control_factors


def load_output_data(h5df_file_list, episode_num, start_ts, chunk_size, _device):
    with h5py.File(h5df_file_list[episode_num], 'r') as root:
        episode_len = root.attrs['episode_len']
        sentence_embedding = root['sentence_embedding'][:]

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
            actions_full.append(root['actions'][key][start_ts : start_ts + chunk_size])
        actions_full = np.concatenate(actions_full, axis=1)
        action_len = actions_full.shape[0]

        original_action_shape = (chunk_size,) + actions_full.shape[1:]
        padded_action = np.zeros(original_action_shape, dtype=np.float32)
        padded_action[:action_len] = actions_full
        padded_action[action_len:] = actions_full[-1]
        is_pad = np.zeros(chunk_size)
        is_pad[action_len:] = 1

        image_data = torch.from_numpy(images)
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()
        context_data = torch.from_numpy(sentence_embedding).float()

        image_data = image_data / 255.0
        image_data = torch.einsum('k h w c -> k c h w', image_data)

        output_data = {
            'images': image_data.to(_device),
            'contexts': context_data.to(_device),
            'qpos': qpos_data.to(_device),
            'actions': action_data.to(_device),
            'is_pad': is_pad.to(_device),
        }

    return output_data


@jit(nopython=True)
def DH_fast(a_vec, d_vec, R1, q):
    R2 = np.array([[np.cos(q), -np.sin(q), 0.0], [np.sin(q), np.cos(q), 0.0], [0.0, 0.0, 1.0]])

    R_tot = R1 @ R2
    p_tot = R1 @ R2 @ d_vec + a_vec

    T = np.eye(4)
    T[:3, :3] = R_tot
    T[:3, 3] = p_tot
    return T


@jit(nopython=True)
def Franka_forward_kinematics(q):
    a_vecs = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0825, 0.0, 0.0],
            [-0.0825, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.088, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    d_vecs = np.array(
        [
            [0.0, 0.0, 0.333],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.316],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.384],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.107],
        ]
    )
    R1s = np.array(
        [
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, -0.0],
                [0.0, 0.0, 1.0],
            ],
            [
                [1.000000e00, 0.000000e00, 0.000000e00],
                [0.000000e00, 6.123234e-17, 1.000000e00],
                [0.000000e00, -1.000000e00, 6.123234e-17],
            ],
            [
                [1.000000e00, 0.000000e00, 0.000000e00],
                [0.000000e00, 6.123234e-17, -1.000000e00],
                [0.000000e00, 1.000000e00, 6.123234e-17],
            ],
            [
                [1.000000e00, 0.000000e00, 0.000000e00],
                [0.000000e00, 6.123234e-17, -1.000000e00],
                [0.000000e00, 1.000000e00, 6.123234e-17],
            ],
            [
                [1.000000e00, 0.000000e00, 0.000000e00],
                [0.000000e00, 6.123234e-17, 1.000000e00],
                [0.000000e00, -1.000000e00, 6.123234e-17],
            ],
            [
                [1.000000e00, 0.000000e00, 0.000000e00],
                [0.000000e00, 6.123234e-17, -1.000000e00],
                [0.000000e00, 1.000000e00, 6.123234e-17],
            ],
            [
                [1.000000e00, 0.000000e00, 0.000000e00],
                [0.000000e00, 6.123234e-17, -1.000000e00],
                [0.000000e00, 1.000000e00, 6.123234e-17],
            ],
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, -0.0],
                [0.0, 0.0, 1.0],
            ],
        ]
    )

    T = np.eye(4)
    N = len(a_vecs)
    for i in range(N):
        if i == N - 1:
            T = T @ DH_fast(a_vecs[i], d_vecs[i], R1s[i], 0.0)
        else:
            T = T @ DH_fast(a_vecs[i], d_vecs[i], R1s[i], q[i])

    return T


if __name__ == '__main__':
    app.run(main)
