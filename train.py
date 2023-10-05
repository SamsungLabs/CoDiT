import os
import shutil

from absl import app, flags

import gin
from core import Task, Trainer
import torch
import torch.multiprocessing as mp
import torch.distributed as dist

FLAGS = flags.FLAGS
flags.DEFINE_string('exp_dir', None, 'Path of exp dir')
flags.DEFINE_string(
    'data_root',
    '/storage/dataset',
    'Data root dir will be concatenated with data path in data_config.gin',
)
flags.DEFINE_string('data_config', 'none', 'Gin config for dataset')
flags.DEFINE_string('model_config', 'none', 'Gin config for model')
flags.DEFINE_string('task_config', 'none', 'Gin config for task')
flags.DEFINE_string('task', 'none', 'Task')
flags.DEFINE_string('model', 'none', 'Model')
flags.DEFINE_multi_string('gin_param', None, 'List of Gin parameter bindings')


def main(argv):
    exp_dir = FLAGS.exp_dir
    data_root = FLAGS.data_root
    data_config = FLAGS.data_config
    model_config = FLAGS.model_config
    task_config = FLAGS.task_config

    import numpy as np

    port = str(np.random.randint(23456, 24456))

    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    shutil.copyfile(data_config, os.path.join(exp_dir, 'data.gin'))
    shutil.copyfile(model_config, os.path.join(exp_dir, 'model.gin'))
    shutil.copyfile(task_config, os.path.join(exp_dir, 'task.gin'))

    gin_files = [data_config, model_config, task_config]
    gin_params = FLAGS.gin_param

    ngpus_per_node = torch.cuda.device_count()
    mp.spawn(
        main_worker,
        nprocs=ngpus_per_node,
        args=(ngpus_per_node, exp_dir, data_root, gin_files, gin_params, port),
    )


def main_worker(gpu, ngpus_per_node, exp_dir, data_root, gin_files, gin_params, port):
    torch.cuda.set_device(gpu)

    print("Use GPU: {} for training".format(gpu))
    dist.init_process_group(
        backend='nccl',
        init_method='tcp://127.0.0.1:' + port,
        world_size=ngpus_per_node,
        rank=gpu,
    )
    if ngpus_per_node == 1:
        num_workers = 1
    else:
        num_workers = 8

    gin.parse_config_files_and_bindings(gin_files, gin_params)

    task = Task(exp_dir=exp_dir, data_root=data_root)
    trainer = Trainer(task, gpu, num_workers)
    trainer.train()


if __name__ == '__main__':
    app.run(main)
