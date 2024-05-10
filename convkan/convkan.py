import torch
from torch.nn import functional as F

from convkan.kanlinear import KANLinear


class ConvKAN(torch.nn.Module):
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 kernel_size: int or tuple = 3,
                 stride: int = 1,
                 padding: int = 0,
                 dilation: int = 1,
                 grid_size: int = 5,
                 spline_order: int = 3,
                 scale_noise: float = 0.1,
                 scale_base: float = 1.0,
                 scale_spline: float = 1.0,
                 enable_standalone_scale_spline: bool = True,
                 base_activation: torch.nn.Module = torch.nn.SiLU,
                 grid_eps: float = 0.02,
                 grid_range: tuple = (-1, 1),
                 ):
        """
        Convolutional layer with KAN kernels.

        Args:
            in_dim (int): Number of channels in the input image
            out_dim (int): Number of channels produced by the convolution
            kernel_size (int or tuple): Size of the convolving kernel
            stride (int): Stride of the convolution. Default: 1
            padding (int): Zero-padding added to both sides of the input. Default: 0
            dilation (int): Spacing between kernel elements. Default: 1
            grid_size (int): Number of grid points for the spline. Default: 5
            spline_order (int): Order of the spline. Default: 3
            scale_noise (float): Scale of the noise. Default: 0.1
            scale_base (float): Scale of the base. Default: 1.0
            scale_spline (float): Scale of the spline. Default: 1.0
            enable_standalone_scale_spline (bool): Enable standalone scale for the spline. Default: True
            base_activation (torch.nn.Module): Activation function for the base. Default: torch.nn.SiLU
            grid_eps (float): Epsilon for the grid
            grid_range (tuple): Range of the grid. Default: (-1, 1).
        """
        super().__init__()
        self.input_channels = in_dim
        self.output_channels = out_dim

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        self.kan_layer = KANLinear(
            in_dim * kernel_size[0] * kernel_size[1],
            out_dim,
            grid_size=grid_size,
            spline_order=spline_order,
            scale_noise=scale_noise,
            scale_base=scale_base,
            scale_spline=scale_spline,
            enable_standalone_scale_spline=enable_standalone_scale_spline,
            base_activation=base_activation,
            grid_eps=grid_eps,
            grid_range=grid_range,
        )

    def forward(self, x):
        # Batch size and input dimensions
        batch_size, channels, height, width = x.shape

        # Apply unfold (im2col)
        x_unf = F.unfold(
            x, kernel_size=self.kernel_size, padding=self.padding, stride=self.stride, dilation=self.dilation,
        )
        # x_unf.shape = (batch_size, channels * kernel_size * kernel_size, L), L is the number of resulting columns

        # Reshape for passing to the module
        x_unf = x_unf.transpose(1, 2).contiguous().view(
            -1, self.input_channels * self.kernel_size[0] * self.kernel_size[1]
        )

        # Pass through the KAN layer
        output = self.kan_layer(x_unf)

        # Calculate output dimensions
        output_height = (height + 2 * self.padding - self.kernel_size[0]) // self.stride + 1
        output_width = (width + 2 * self.padding - self.kernel_size[1]) // self.stride + 1

        # Reshape output to image format
        output = output.view(batch_size, -1, output_height, output_width)

        return output
