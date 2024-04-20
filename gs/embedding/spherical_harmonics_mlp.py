from typing import List, Union
import torch
from torch import nn
from gs.core.View import KnownView

class SphericalHarmonicsMLP(nn.Module):
    def __init__(self, num_sh_coefficients: int, num_sh_channels: int, cameras: List[KnownView], embedding_dim: int=128):
        # Num SH coefficients: Number of spherical harmonics coefficients to process
        # Num SH channels: Number of channels in each spherical harmonics coefficient (3 for RGB)
        # Num positional channels: Number of channels for positional encoding, we default to 10
        super().__init__()
        self.num_sh_coefficients = num_sh_coefficients
        self.num_sh_channels = num_sh_channels
        self.embedding_dim = embedding_dim

        # Create a lookup table from camera.id to embedding index
        self.camera_id_to_embedding_index = {camera.id: i for i, camera in enumerate(cameras)}
        self.embedding_layer = nn.Embedding(len(cameras), embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(num_sh_coefficients * num_sh_channels + embedding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, num_sh_coefficients * num_sh_channels),
        )
        # Zero-initialize the weights of the last layer
        self.mlp[-1].weight.data.zero_()
        self.mlp[-1].bias.data.zero_()
    
    def forward(self, in_sh_coefficients: torch.Tensor, appearance: Union[KnownView, torch.Tensor]) -> torch.Tensor: # Apperance should be a single camera or embedding tensor
        """
        Apply the MLP to the model sh_coefficients based on a camera position in a residual fashion.
        """
        if isinstance(appearance, torch.Tensor):
            # Check if shape is of a single embedding
            assert appearance.shape == (self.embedding_dim,), "Appearance should be a single embedding tensor"
            embedding = appearance # If appearance is a tensor, we assume it is the embedding
        else:
            embedding = self.embedding_layer(torch.tensor([self.camera_id_to_embedding_index[appearance.id]], dtype=torch.long, device=in_sh_coefficients.device))
        # Expand the embedding to match the shape of sh_coefficients
        embedding = embedding.expand(in_sh_coefficients.shape[0], -1)

        # Flatten sh_coefficients
        sh_coefficients = in_sh_coefficients.view(in_sh_coefficients.shape[0], -1)
        # Concatenate sh_coefficients and camera_position
        input = torch.cat([sh_coefficients, embedding], dim=1)
        # Apply MLP
        output = self.mlp(input)
        # Reshape output to match the shape of sh_coefficients, and add it to sh_coefficients
        output = output.view(in_sh_coefficients.shape)
        return in_sh_coefficients + output
