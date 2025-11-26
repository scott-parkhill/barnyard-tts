from typing import Optional

import torch
from torch import nn
from torch.nn.utils import weight_norm

from vocos.modules import ConvNeXtBlock, ResBlock1, AdaLayerNorm


class Backbone(nn.Module):
    """Base class for the generator's backbone. It preserves the same temporal resolution across all layers."""

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            x (Tensor): Input tensor of shape (B, C, L), where B is the batch size,
                        C denotes output features, and L is the sequence length.

        Returns:
            Tensor: Output of shape (B, L, H), where B is the batch size, L is the sequence length,
                    and H denotes the model dimension.
        """
        raise NotImplementedError("Subclasses must implement the forward method.")


class VocosBackbone(Backbone):
    """
    Vocos backbone module built with ConvNeXt blocks. Supports additional conditioning with Adaptive Layer Normalization

    Args:
        input_channels (int): Number of input features channels.
        hidden_dimension (int): Hidden dimension of the model.
        intermediate_dimension (int): Intermediate dimension used in ConvNeXtBlock.
        number_of_layers (int): Number of ConvNeXtBlock layers.
        layer_scale_init_value (float, optional): Initial value for layer scaling. Defaults to `1 / num_layers`.
        adanorm_embeddings (int, optional): Number of embeddings for AdaLayerNorm.
                                                None means non-conditional model. Defaults to None.
    """

    def __init__(
        self,
        input_channels: int,
        hidden_dimension: int,
        intermediate_dimension: int,
        number_of_layers: int,
        layer_scale_init_value: Optional[float] = None,
        adanorm_embeddings: Optional[int] = None,
    ):
        # TODO Magic numbers.
        super().__init__()
        self.input_channels = input_channels
        self.embed = nn.Conv1d(input_channels, hidden_dimension, kernel_size=7, padding=3)
        self.adanorm = adanorm_embeddings is not None
        if adanorm_embeddings:
            self.norm = AdaLayerNorm(adanorm_embeddings, hidden_dimension, eps=1e-6)
        else:
            self.norm = nn.LayerNorm(hidden_dimension, eps=1e-6)
        layer_scale_init_value = layer_scale_init_value or 1 / number_of_layers

        # TODO Make this a normal for block setting a list. This is crazy initialization.
        self.convnext = nn.ModuleList(
            [
                ConvNeXtBlock(
                    dim=hidden_dimension,
                    intermediate_dim=intermediate_dimension,
                    layer_scale_init_value=layer_scale_init_value,
                    adanorm_num_embeddings=adanorm_embeddings,
                )
                for _ in range(number_of_layers)
            ]
        )
        self.final_layer_norm = nn.LayerNorm(hidden_dimension, eps=1e-6)
        self.apply(self._init_weights)

    # TODO What is "m"?
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    # TODO What is "x"?
    # TODO Why kwargs when there is only a single access to bandwidth_id and nothing else?
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        bandwidth_id = kwargs.get('bandwidth_id', None)
        x = self.embed(x)
        if self.adanorm:
            assert bandwidth_id is not None
            x = self.norm(x.transpose(1, 2), cond_embedding_id=bandwidth_id)
        else:
            x = self.norm(x.transpose(1, 2))
        x = x.transpose(1, 2)
        for conv_block in self.convnext:
            x = conv_block(x, cond_embedding_id=bandwidth_id)
        x = self.final_layer_norm(x.transpose(1, 2))
        return x


class VocosResNetBackbone(Backbone):
    """
    Vocos backbone module built with ResBlocks.

    Args:
        input_channels (int): Number of input features channels.
        hidden_dimension (int): Hidden dimension of the model.
        blocks (int): Number of ResBlock1 blocks.
        layer_scale_init_value (float, optional): Initial value for layer scaling. Defaults to None.
    """

    def __init__(
        self, input_channels, hidden_dimension, blocks, layer_scale_init_value=None,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.embed = weight_norm(nn.Conv1d(input_channels, hidden_dimension, kernel_size=3, padding=1))
        layer_scale_init_value = layer_scale_init_value or 1 / blocks / 3

        # TODO Remove this cursed initialization.
        self.resnet = nn.Sequential(
            *[ResBlock1(dim=hidden_dimension, layer_scale_init_value=layer_scale_init_value) for _ in range(blocks)]
        )

    # TODO What is x? Why is kwargs here when it is unused?
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        x = self.embed(x)
        x = self.resnet(x)
        x = x.transpose(1, 2)
        return x
