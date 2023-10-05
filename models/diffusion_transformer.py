import gin
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .registry import register
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from models.module.diffusion_submodule_network import TransformerForDiffusion
from models.module.diffusion_submodule_network import PositionEmbeddingSine
from einops import reduce
from typing import Optional

from models.module.cnn_backbone import ResnetBackbone


@register('diffusion_transformer')
@gin.configurable()
class DiffusionTransformer(nn.Module):
    def __init__(
        self,
        noise_scheduler_name='DDIM',
        num_train_timesteps=100,
        num_inference_timesteps=10,
        model_type='diffusion_model',  # 'noise_generator' / 'diffusion_model_with_guided_noise' / 'diffusion_model'
        action_dim=16,
        chunk_size=50,
        hidden_dim=512,
        transformer_dropout=0.1,
        transformer_nhea=8,
        dim_feedforward=2048,
        num_encoder_layers=2,
        num_decoder_layers=2,
        is_included_images=False,
        cnn_backbone_type='resnet18',
        is_included_proprioception=False,
        proprioception_dim=16,
        is_included_control_factor=False,
        is_included_diffusion_x=False,
        diffusion_x_steps=5,
        use_mask=False,
        freeze_imagenet=False,
        use_only_external_images=False,
        use_whole_features=True,
        **kwargs,
    ):
        super(DiffusionTransformer, self).__init__()

        if noise_scheduler_name == 'DDIM':
            noise_scheduler = DDIMScheduler(
                num_train_timesteps=num_train_timesteps,
                beta_start=0.0001,
                beta_end=0.02,
                beta_schedule='squaredcos_cap_v2',
                clip_sample=True,
                set_alpha_to_one=True,
                steps_offset=0,
                prediction_type='epsilon',
            )
        elif noise_scheduler_name == 'DDPM':
            noise_scheduler = DDPMScheduler(
                num_train_timesteps=num_train_timesteps,
                beta_start=0.0001,
                beta_end=0.02,
                beta_schedule='squaredcos_cap_v2',
                variance_type='fixed_small',
                clip_sample=True,
                prediction_type='epsilon',
            )
        else:
            raise ValueError(f"Unsupported Noise Scheduler {noise_scheduler_name}")

        self.num_train_timesteps = num_train_timesteps
        self.num_inference_timesteps = num_inference_timesteps
        self.model_type = model_type
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.hidden_dim = hidden_dim
        self.transformer_dropout = transformer_dropout
        self.transformer_nhea = transformer_nhea
        self.dim_feedforward = dim_feedforward
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.is_included_images = is_included_images
        self.cnn_backbone_type = cnn_backbone_type
        self.is_included_proprioception = is_included_proprioception
        self.proprioception_dim = proprioception_dim
        self.is_included_control_factor = is_included_control_factor
        self.is_included_diffusion_x = is_included_diffusion_x
        self.diffusion_x_steps = diffusion_x_steps
        self.use_mask = use_mask
        self.freeze_imagenet = freeze_imagenet
        self.use_only_external_images = use_only_external_images
        self.use_whole_features = use_whole_features

        if self.num_inference_timesteps is None:
            self.num_inference_timesteps = self.num_train_timesteps

        self.model = TransformerForDiffusion(
            self.action_dim,
            self.chunk_size,
            self.hidden_dim,
            self.transformer_dropout,
            self.transformer_nhea,
            self.dim_feedforward,
            self.num_encoder_layers,
            self.num_decoder_layers,
            self.is_included_images,
            self.cnn_backbone_type,
            self.is_included_proprioception,
            self.proprioception_dim,
            self.is_included_control_factor,
        )

        if self.is_included_images:
            self.cnn_backbone = ResnetBackbone(
                name=self.cnn_backbone_type,
                train_backbone=True,
                return_interm_layers=False,
                dilation=False,
                use_whole_features=self.use_whole_features,
            )
            if self.freeze_imagenet:
                for param in self.cnn_backbone.parameters():
                    param.requires_grad = False

            if self.use_whole_features:
                self.input_proj_image = nn.Conv2d(
                    self.cnn_backbone.num_channels, self.hidden_dim, kernel_size=1
                )

                self.position_embedding = PositionEmbeddingSine(
                    num_pos_feats=self.hidden_dim // 2,
                    temperature=10000,
                    normalize=False,
                    scale=None,
                )
            else:
                self.input_proj_image = nn.Linear(self.cnn_backbone.num_channels, self.hidden_dim)
                if self.use_only_external_images:
                    self.position_embedding = nn.Embedding(2, self.hidden_dim)
                else:
                    self.position_embedding = nn.Embedding(4, self.hidden_dim)

        if self.is_included_proprioception:
            self.input_proj_prop = nn.Linear(self.proprioception_dim, self.hidden_dim)

        if self.is_included_control_factor:
            self.input_proj_cf = nn.Linear(1, self.hidden_dim)

        self.noise_scheduler = noise_scheduler
        self.kwargs = kwargs

        if (
            self.model_type == 'noise_generator'
            or self.model_type == 'diffusion_model_with_guided_noise'
            or self.model_type == 'diffusion_model'
        ):
            pass
        else:
            raise ValueError("Invalid Model Type")

    def set_data_statistics(self, data_statistics):
        self._data_statistics = {
            'action_max': torch.from_numpy(data_statistics.action_max.astype(np.float32)),
            'action_min': torch.from_numpy(data_statistics.action_min.astype(np.float32)),
        }

    def images_to_feature_and_pos(self, images):
        if self.use_whole_features:
            bs = images.shape[0]
            # fold camera dimension into width dimension
            image_input, image_pos_emb = self.get_image_features(
                images
            )  # (bs, hidden_dim, 15, 80), (1, hidden_dim, 15, 80)

            # flatten NxCxHxW to HWxNxC
            image_input = image_input.flatten(2).permute(0, 2, 1)  # (bs, 15*20, hidden_dim)
            image_pos_emb = image_pos_emb.flatten(2).permute(0, 2, 1).repeat(bs, 1, 1)
        else:
            bs = images.shape[0]
            image_input = self.get_image_features(images)  # (bs, 4, hidden_dim)
            image_pos_emb = self.position_embedding.weight.unsqueeze(0).repeat(
                bs, 1, 1
            )  # (bs, 4, hidden_dim)

        return image_input, image_pos_emb

    def get_image_features(self, images):
        if self.use_whole_features:
            # Image observation features and position embeddings
            all_cam_features = []
            all_cam_pos = []
            num_camera = images.shape[1]

            for i in range(num_camera):
                features = self.cnn_backbone(images[:, i, :, :, :])

                feature_list = []
                pos_list = []
                for key, feature in features.items():
                    # print(feature.shape)
                    # feature: (bs, 512, 15, 20)
                    pos = self.position_embedding(feature)
                    feature_list.append(feature)
                    pos_list.append(pos)
                picked_features = feature_list[0]  # (bs, 512, 15, 20)
                picked_pos = pos_list[0]  # (bs, hidden_dim, 15, 20)
                all_cam_features.append(self.input_proj_image(picked_features))
                all_cam_pos.append(picked_pos)
            return torch.cat(all_cam_features, dim=3), torch.cat(all_cam_pos, dim=3)
        else:
            # Image observation features and position embeddings
            all_cam_features = []
            num_camera = images.shape[1]

            for i in range(num_camera):
                feature = self.cnn_backbone(images[:, i, :, :, :])  # (bs, 512)
                all_cam_features.append(self.input_proj_image(feature))
            return torch.stack(all_cam_features, dim=1)  # (bs, 4, 512)

    def forward(
        self,
        actions,
        prop: Optional[torch.Tensor] = None,
        control_factor: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        if self.model_type == 'noise_generator':
            pass
        elif self.model_type == 'diffusion_model_with_guided_noise':
            actions = self.normalization(actions)
        elif self.model_type == 'diffusion_model':
            pass

        prop_input = None
        cf_input = None
        image_input = None
        image_pos_emb = None

        if self.is_included_proprioception:
            prop_input = self.input_proj_prop(prop)  # (bs, hidden_dim)

        if self.is_included_control_factor:
            cf_input = self.input_proj_cf(control_factor)  # (bs, hidden_dim)

        if self.is_included_images:
            if self.use_only_external_images:
                images = torch.cat([images[:, 1:2, :, :, :], images[:, 3:4, :, :, :]], dim=1)
            image_input, image_pos_emb = self.images_to_feature_and_pos(images)

        self.noise_scheduler.set_timesteps(self.num_inference_timesteps)
        for t in self.noise_scheduler.timesteps:
            model_output = self.model(
                actions=actions,
                timestep=t,
                prop_input=prop_input,
                cf_input=cf_input,
                image_input=image_input,
                image_pos_emb=image_pos_emb,
                use_mask=self.use_mask,
            )
            actions = self.noise_scheduler.step(
                model_output, t, actions, generator=None, **kwargs
            ).prev_sample

        if self.is_included_diffusion_x:
            for i in range(self.diffusion_x_steps):
                t = self.noise_scheduler.timesteps[-1]
                model_output = self.model(
                    actions=actions,
                    timestep=t,
                    prop_input=prop_input,
                    cf_input=cf_input,
                    image_input=image_input,
                    image_pos_emb=image_pos_emb,
                    use_mask=self.use_mask,
                )
                actions = self.noise_scheduler.step(
                    model_output, t, actions, generator=None, **kwargs
                ).prev_sample

        action_pred = self.unnormalization(actions)
        result = {'a_hat': action_pred}
        return result

    def compute_loss(
        self,
        actions,
        prop: Optional[torch.Tensor] = None,
        control_factor: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
    ):
        actions = self.normalization(actions)

        prop_input = None
        cf_input = None
        image_input = None
        image_pos_emb = None

        if self.is_included_proprioception:
            prop_input = self.input_proj_prop(prop)  # (bs, hidden_dim)

        if self.is_included_control_factor:
            cf_input = self.input_proj_cf(control_factor)  # (bs, hidden_dim)

        if self.is_included_images:
            image_input, image_pos_emb = self.images_to_feature_and_pos(images)

        noise = torch.randn(actions.shape, device=actions.device)
        bs = actions.shape[0]
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, (bs,), device=actions.device
        ).long()
        noisy_trajectory = self.noise_scheduler.add_noise(actions, noise, timesteps)

        pred = self.model(
            actions=noisy_trajectory,
            timestep=timesteps,
            prop_input=prop_input,
            cf_input=cf_input,
            image_input=image_input,
            image_pos_emb=image_pos_emb,
            use_mask=self.use_mask,
        )

        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = actions
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()

        losses = {'total_loss': loss}
        return losses

    def normalization(self, actions):
        action_max = self._data_statistics['action_max']
        action_min = self._data_statistics['action_min']
        action_range = action_max - action_min
        ignore_dim = torch.abs(action_range) < 1e-6
        action_max[ignore_dim] = action_min[ignore_dim] + 1
        action_min[ignore_dim] = action_min[ignore_dim] - 1
        action_max = action_max.to(actions.device)[None, None, :].repeat(
            actions.shape[0], actions.shape[1], 1
        )
        action_min = action_min.to(actions.device)[None, None, :].repeat(
            actions.shape[0], actions.shape[1], 1
        )
        actions_nor = (actions - action_min) / (action_max - action_min) * 2 - 1
        return actions_nor

    def unnormalization(self, actions):
        action_max = self._data_statistics['action_max']
        action_min = self._data_statistics['action_min']

        action_range = action_max - action_min
        ignore_dim = torch.abs(action_range) < 1e-6
        action_max[ignore_dim] = action_min[ignore_dim] + 1
        action_min[ignore_dim] = action_min[ignore_dim] - 1
        action_max = action_max.to(actions.device)[None, None, :].repeat(
            actions.shape[0], actions.shape[1], 1
        )
        action_min = action_min.to(actions.device)[None, None, :].repeat(
            actions.shape[0], actions.shape[1], 1
        )
        actions_unnor = (action_max - action_min) / 2 * actions + (action_max + action_min) / 2

        return actions_unnor

    @torch.jit.unused
    def train_one_step(self, data, optimizer):
        actions = data['actions']  # (bs, chunk_size, 16)
        prop = None
        control_factor = None
        images = None

        if self.is_included_proprioception:
            prop = data['qpos']  # (bs, 16)
        if self.is_included_control_factor:
            control_factor = data['control_factor']
        if self.is_included_images:
            if self.use_only_external_images:
                images = torch.cat(
                    [data['images'][:, 1:2, :, :, :], data['images'][:, 3:4, :, :, :]], dim=1
                )  # (bs, 2, 3, 480, 640)
            else:
                images = data[
                    'images'
                ]  # (bs, 4, 3, 480, 640) (0 to 1) left hand, left head, right hand, right head)

        optimizer.zero_grad()
        losses = self.compute_loss(actions, prop, control_factor, images)
        total_loss = losses['total_loss']
        total_loss.backward()
        optimizer.step()
        return losses

    @torch.jit.unused
    def val_one_step(self, data):
        actions = data['actions']  # (bs, chunk_size, 16)
        prop = None
        control_factor = None
        images = None

        if self.is_included_proprioception:
            prop = data['qpos']  # (bs, 16)
        if self.is_included_control_factor:
            control_factor = data['control_factor']  # (bs, 1)
        if self.is_included_images:
            if self.use_only_external_images:
                images = torch.cat(
                    [data['images'][:, 1:2, :, :, :], data['images'][:, 3:4, :, :, :]], dim=1
                )  # (bs, 2, 3, 480, 640)
            else:
                images = data[
                    'images'
                ]  # (bs, 4, 3, 480, 640) (0 to 1) left hand, left head, right hand, right head)

        with torch.inference_mode():
            losses = self.compute_loss(actions, prop, control_factor, images)
        return losses
