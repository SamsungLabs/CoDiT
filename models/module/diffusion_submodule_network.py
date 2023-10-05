import torch
import torch.nn as nn
import math
import logging
from typing import Optional
from torch import nn

logger = logging.getLogger(__name__)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class TransformerForDiffusion(nn.Module):
    def __init__(
        self,
        action_dim,
        chunk_size,
        hidden_dim,
        transformer_dropout,
        transformer_nhea,
        dim_feedforward,
        num_encoder_layers,
        num_decoder_layers,
        is_included_images=False,
        cnn_backbone_type=False,
        is_included_proprioception=False,
        proprioception_dim=16,
        is_included_control_factor=False,
    ):
        super(TransformerForDiffusion, self).__init__()

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

        additional_obs_num = 0

        self.input_proj_action = nn.Linear(self.action_dim, self.hidden_dim)

        if self.is_included_proprioception:
            additional_obs_num += 1

        if self.is_included_control_factor:
            additional_obs_num += 1

        self.time_encoder = nn.Sequential(
            SinusoidalPosEmb(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim * 4),
            nn.Mish(),
            nn.Linear(self.hidden_dim * 4, self.hidden_dim),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.transformer_nhea,
            dim_feedforward=self.dim_feedforward,
            dropout=self.transformer_dropout,
            activation='gelu',
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=num_encoder_layers
        )

        self.additional_pos_embed = nn.Embedding(
            self.chunk_size + additional_obs_num + 1, self.hidden_dim
        )  # learned position embedding for actions and addtional obs like proprio, + time

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.hidden_dim,
            nhead=self.transformer_nhea,
            dim_feedforward=self.dim_feedforward,
            dropout=self.transformer_dropout,
            activation='gelu',
            batch_first=True,
            norm_first=False,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer, num_layers=num_decoder_layers
        )

        tgt_mask = self.make_tgt_mask().cuda()
        self.register_buffer("tgt_mask", tgt_mask)

        self.action_head = nn.Linear(self.hidden_dim, self.action_dim)

    def forward(
        self,
        actions,
        timestep,
        prop_input: Optional[torch.Tensor] = None,
        cf_input: Optional[torch.Tensor] = None,
        image_input: Optional[torch.Tensor] = None,
        image_pos_emb: Optional[torch.Tensor] = None,
        use_mask=False,
    ):
        bs = actions.shape[0]

        # action
        action_input = self.input_proj_action(actions)  # (bs, chunk_size, hidden_dim)

        # time
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], dtype=torch.long, device=actions.device)
        elif torch.is_tensor(timestep) and len(timestep.shape) == 0:
            timestep = timestep[None].to(actions.device)
        timestep = timestep.expand(actions.shape[0])
        time_input = self.time_encoder(timestep)  # (bs, hidden_dim)
        src = []
        src.append(time_input)

        if self.is_included_proprioception:
            src.append(prop_input)

        if self.is_included_control_factor:
            src.append(cf_input)

        src = torch.stack(src, dim=1)  # (bs, N, hidden_dim)
        src = torch.cat(
            [action_input, src], dim=1
        )  # [action, time, prop, control_factor], (bs, N, hidden_dim)

        pos = self.additional_pos_embed.weight.unsqueeze(0).repeat(bs, 1, 1)

        if self.is_included_images:
            src = torch.cat([src, image_input], dim=1)
            pos = torch.cat([pos, image_pos_emb], dim=1)

        src = src + pos
        tgt = src[:, : self.chunk_size, :]

        memory = self.encoder(src)[:, self.chunk_size :, :]

        if use_mask:
            hs = self.decoder(tgt, memory, tgt_mask=self.tgt_mask)  # (bs, chunk_size, hidden_dim)
        else:
            hs = self.decoder(tgt, memory)  # (bs, chunk_size, hidden_dim)
        a_hat = self.action_head(hs)  # (bs, chunk_size, action_dim)

        return a_hat

    def make_tgt_mask(self):
        tgt_mask = (torch.triu(torch.ones(self.chunk_size, self.chunk_size)) == 1).transpose(0, 1)
        tgt_mask = (
            tgt_mask.masked_fill(tgt_mask == 1, float(0.0))
            .masked_fill(tgt_mask == 0, float('-inf'))
            .float()
        )  # (seq_len, seq_len)

        return tgt_mask


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor):
        x = tensor

        not_mask = torch.ones_like(x[0, [0]])
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos
