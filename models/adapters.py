import math
from typing import Tuple
import torch
from torch import Tensor
from torch import nn

from models.model_utils import sandwich, pe_encode_float


class FourierImageInputAdapter(nn.Module):
    """
    A module to convert 2D image coordinates into a set of vectors represented as a matrix, with fourier position codes.
    """

    def __init__(
        self,
        input_channels: int = 3,
        input_shape: Tuple[int, int] = (32, 32),
        n_freq_bands: int = 64,
        output_height: int = 32,
        value_res: int = -1,
        mask_res: int = -1,
        add_pos_feats: bool = True,
        add_mask: bool = True,
        learn_pos_feats: bool = False,
        pos_embed_size: int = 32,
        init_scale: float = 0.02,
    ):
        super().__init__()
        self.input_shape = input_shape
        self.n_freq_bands = n_freq_bands
        self.value_res = value_res
        self.mask_res = mask_res
        self.add_pos_feats = add_pos_feats
        self.add_mask = add_mask
        self.input_channels = input_channels
        if learn_pos_feats:
            pos_feats = nn.Parameter(
                init_scale
                * torch.randn(1, input_shape[0] * input_shape[1], pos_embed_size)
            )
            self.register_parameter("pos_feats", pos_feats)
        else:
            x = torch.linspace(-1.0, 1.0, steps=input_shape[0])
            y = torch.linspace(-1.0, 1.0, steps=input_shape[1])
            x_pos, y_pos = torch.meshgrid(x, y, indexing="ij")
            pos = torch.stack((x_pos, y_pos), dim=-1)
            pos = pos.reshape(-1, 2)
            x_bands = torch.linspace(1.0, input_shape[0] / 2, steps=n_freq_bands)
            y_bands = torch.linspace(1.0, input_shape[1] / 2, steps=n_freq_bands)
            bands = torch.stack((x_bands, y_bands), dim=0)
            vals = pos[:, :, None] * bands[None, :, :]
            vals = math.pi * vals.reshape(vals.shape[0], -1)
            pos_feats = torch.cat([vals.sin(), vals.cos()], dim=-1)
            pos_feats = torch.cat([pos_feats, pos], dim=-1)
            self.register_buffer("pos_feats", pos_feats)
        img_feat_height = input_channels
        pos_feat_height = pos_feats.size(-1)
        if self.mask_res > 0:
            mask_feat_height = (n_freq_bands * 2) + 1
        else:
            mask_feat_height = 1
        all_feat_height = img_feat_height
        if add_mask:
            all_feat_height += mask_feat_height
        if add_pos_feats:
            all_feat_height += pos_feat_height
        self.output_projection = None
        if output_height != all_feat_height:
            self.output_projection = nn.Linear(all_feat_height, output_height)

    def forward(self, img: Tensor, t: Tensor) -> Tensor:
        #print(f'FourierIIA img.shape {img.shape}')
        img = img.reshape(-1, *self.input_shape, self.input_channels)
        #print(f'FourierIIA img.shape {img.shape}')
        #print(f'FourierIIA t.shape {t.shape}')
        flat_img = sandwich(img)
        flat_t = sandwich(t)
        #print(f'flat_img.shape {flat_img.shape}')

        # broadcast t to the same shape as the image
        flat_t = flat_t.expand(-1, flat_img.size(1), 1)
        #print(f'flat_t.shape {flat_t.shape}')
        t_feats = (flat_t.float()[..., :1] * 2) - 1
        #print(f't_feats.shape {t_feats.shape}')

        if self.mask_res > 0:
            t_feats = torch.cat(
                [
                    t_feats,
                    pe_encode_float(
                        t_feats, self.mask_res, self.n_freq_bands * 2
                    ).flatten(start_dim=2),
                ],
                -1,
            )
        fourier_feats = self.pos_feats.expand(img.size(0), -1, -1)
        all_feat_list = [flat_img]
        if self.add_mask:
            all_feat_list.append(t_feats)
        if self.add_pos_feats:
            all_feat_list.append(fourier_feats)
        #print(f'all_feat_list[0].shape {all_feat_list[0].shape}')
        #print(f'all_feat_list[1].shape {all_feat_list[1].shape}')
        all_feats = torch.cat(all_feat_list, dim=-1)
        if self.output_projection is None:
            output = all_feats
        else:
            output = self.output_projection(all_feats)
        return output


class OutputAdapter(nn.Module):
    """
        input_height: 131
        output_channels: 3 # (r,g,b)
        output_height: 2 # mean, std
    """
    def __init__(self, network_height: int = 160, output_channels: int = 3, n_outputs: int = 2):
        super().__init__()
        self.output_channels = output_channels
        self.n_outputs = n_outputs
        self.output_projection = nn.Linear(
            network_height, output_channels*n_outputs
        )

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        output = self.output_projection(inp)
        # output is of shape Tensor[B, D*channels, n_outputs]
        return output.reshape(output.size(0), -1, self.n_outputs)   