import torch
import torch.nn as nn
import math

class SinusoidalEncoding(nn.Module):
    def __init__(self, input_dims=3, num_basis=256):
        """
        Initialize the SinusoidalEncoding module.
        
        Args:
            D (int): Number of dimensions of the input points.
            B (int): Number of basis functions.
        """
        super(SinusoidalEncoding, self).__init__() # Call the constructor of the parent class
        self.input_dims = input_dims  # Number of input dimensions
        self.num_basis = num_basis  # Number of basis functions
        
        # Create a parameter tensor for frequencies, initialized based on the number of basis functions
        self.frequencies = nn.Parameter(torch.linspace(1, 10, self.num_basis), requires_grad=False)
    
    def forward(self, x):
        """
        Apply sinusoidal encoding to the input tensor x.
        
        Args:
            x (torch.Tensor): Input tensor of shape (N, D) where N is the number of points and D is the dimensionality.
        
        Returns:
            torch.Tensor: Encoded tensor of shape (N, B)
        """
        N, D = x.shape
        x = x.unsqueeze(-1)  # Shape (N, D, 1)
        
        # Frequencies for encoding: shape (1, 1, B)
        freqs = self.frequencies.view(1, 1, self.num_basis)
        
        # Apply the sinusoidal encoding
        x_proj = 2 * math.pi * x * freqs  # Shape (N, D, B)
        
        # Apply sin and cos and then mean across the dimension 1 (D coordinates)
        sin_x = torch.sin(x_proj)
        cos_x = torch.cos(x_proj)
        encoded_x = torch.cat((sin_x, cos_x), dim=1)  # Concatenate along dimension 1 -> Shape (N, 2*D, B)
        encoded_x = torch.mean(encoded_x, dim=1)  # Reduce mean along the dimension 1 -> Shape (N, B)
        
        return encoded_x