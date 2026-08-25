# Reference: https://github.com/ML4SCI/DeepLense/blob/main/DeepLense_Diffusion_Rishi/models/variable_encoder.py
"""MLP encoder for continuous physical parameter conditioning."""

from torch import Tensor, nn


class ParameterEncoder(nn.Module):
    """Encode continuous physical parameters into a diffusers class embedding.

    A 3-layer MLP, following the DeepLense reference's ``variableencoder``,
    that projects a batch of continuous physical parameters (e.g. lens mass,
    orientation, redshift) to ``embedding_dim``. Pairs with a U-Net built with
    ``class_embed_type="identity"``, whose ``time_embed_dim`` (``block_out_channels[0]
    * 4`` by default) ``embedding_dim`` must match; see :class:`astrogen.models.DDPM`'s
    ``parameter_encoder`` argument.
    """

    def __init__(self, num_parameters: int, embedding_dim: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        hidden_dim = hidden_dim or embedding_dim
        self.mlp = nn.Sequential(
            nn.Linear(num_parameters, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.ReLU(),
            nn.Linear(2 * hidden_dim, embedding_dim),
        )

    def forward(self, parameters: Tensor) -> Tensor:
        return self.mlp(parameters)
