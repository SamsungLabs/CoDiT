# CoDiT

Jaemin Yoon, Rakjoon Chung, Changsu Ha, Rijeong Kang, Woosub Lee, Heungwoo Han, Sungchul Kang, "**CoDiT: Controllable Diffusion Transformer for Visuomotor Policy Learning**.", IEEE International Conference on Robotics and Automation (ICRA), 2024.

<img alt="Badge_Python" src ="https://img.shields.io/badge/Python-3.8-3776AB.svg?&style=flat-square"/>

Pytorch-based robot action learning framework
<br />

## List of maintainers
- Jaemin Yoon (jae_min.yoon@samsung.com)
- Rakjoon Chung (rock.chung@samsung.com)


## Installation
```
conda create -n CoDiT python=3.8
conda activate CoDiT
pip install -r requirements.txt
```
<br />

## Prerequisite: Prepare robot datasets
Create datasets under ```./data_root/raw``` directory. Please following instrucions in each repo to create datasets. After creating datasets, ```data_root``` directory looks like:
```bash
data_root
├── raw
│   └── srrc_dual_frankas
│       ├── task
│       │   ├── episode_0000
│       │   │   ├── 0000.npy
│       │   │   ├── 0001.npy
│       │   │   ├── ...
│       │   │   └── meta_data.json
│       │   ├── episode_0001
│       │   │   ├── 0000.npy
│       │   │   ├── 0001.npy
│       │   │   ├── ...
│       │   │   └── meta_data.json
│       └── ...
          
```

### 1. Process datasets
For efficient training, the robot datasets are processed to a training dataset as a predefined format (e.g. episodic_data). The following command processes a robot dataset (format: **srrc_dual_frankas**) in ```source_dir``` and creates a training dataset (format: **episodic_data**) in ```target_dir```. Please use your own path for ```source_dir``` and ```target_dir```.

```
python ./utils/data_process/process_data.py \
    --data_process srrc_dual_frankas \
    --data_type episodic_data \
    --source_dir ./data_root/raw/srrc_dual_frankas/task \
    --target_dir ./data_root/episodic_data/srrc_dual_frankas \
    --skip_param 1
```

After finishing the processing,  ```data_root``` directory looks like

```bash
data_root
├── raw
│   └── srrc_dual_frankas
│       ├── task
│       │   ├── episode_0000
│       │   │   ├── 0000.npy
│       │   │   ├── 0001.npy
│       │   │   ├── ...
│       │   │   └── meta_data.json
│       │   ├── episode_0001
│       │   │   ├── 0000.npy
│       │   │   ├── 0001.npy
│       │   │   ├── ...
│       │   │   └── meta_data.json
│       └── ...
├── episodic_data
│   └── srrc_dual_frankas
│       ├── task
│       │   ├── train
│       │   │   ├── episode_0000.hdf5
│       │   │   └── ...
│       │   └── val
│       └── ...
```

The structure of ```*.npy file``` and ```meta_data.json``` can be checked from sample data in ./data_root/raw/srrc_dual_frankas/task/episode_000*/*.

### 2. Training diffusion transformer model
Following example is to train  **diffusion transformer** model from **srrc_dual_franka** dataset. Please check ```data_config```, ```model_config```, ```task_config``` for details. Please use your own path for ```source_dir``` and ```target_dir```.

The parameters are in each *.gin file in ```dataset, model, task```.

```
python ./train.py \
    --exp_dir ~/exp/my_exp \
    --data_root ./data_root \
    --data_config ./configs/dataset/srrc_dual_frankas/task.gin \
    --model_config ./configs/model/diffusion_transformer/task.gin \
    --task_config ./configs/task/diffusion_transformer/task.gin
```

After training, ```loss.log```, ```model.pt```, ```model_last.pt``` and config files are created under ```exp_dir``` path.

<br />

### 3. Generate Samples and Calculate Control Factor.

Generate samples based on trained diffusion transformer model.
```exp_dir``` is path of trained diffusion transformer model.
It generates samples in ```control_factor_data``` folder in the execution path.

```
python diffusion_noise_prediction_data_gen.py --exp_dir ~/exp/my_exp \
                                            --data_root ./data_root
```
### 4. Training Sample Generator

Following example is to train **sample generator** utilizing diffusion transformer model from **srrc_dual_franka** dataset. Please check ```data_config```, ```model_config```, ```task_config``` for details. Please use your own path for ```source_dir``` and ```target_dir```.

```
python ./train.py \
    --exp_dir ~/exp/my_exp_cf \
    --data_root ./data_root \
    --data_config ./configs/dataset/srrc_dual_frankas/task_cf.gin \
    --model_config ./configs/model/diffusion_transformer/task_cf.gin \
    --task_config ./configs/task/diffusion_transformer/task_cf.gin
```
## Citation

```
@article{yoon2024codit,
  title={RGBD Fusion Grasp Network with Large-Scale Tableware Grasp Dataset},
  author={Yoon, Jaemin and Chung, Rakjoon and Ha, Changsu and Kang, Rijeong and Lee, Woosub and Han, Heungwoo and Kang, Sungchul},
  booktitle={2024 IEEE International Conference on Robotics and Automation (ICRA)},
  year={2024}
}
```
# License 
MIT License. 

We referenced some code in [Diffusion Policy](https://github.com/real-stanford/diffusion_policy), [Action Chunking with Transformer](https://github.com/tonyzhaozh/act).

# Thanks to 
Jaehyun Park, Wonhyuk Choi, Joonmo Ahn, Hosang Lee, Rijeong Kang, Sunpyo Hong, Hoseong Seo, Dongwoo Park, Changsu Ha in Samsung Research, Robot Center.
