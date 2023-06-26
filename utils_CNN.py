import warnings
from typing import Callable, List, Optional, Sequence, Tuple, Union
import collections
import math
import pathlib
from itertools import repeat
from types import FunctionType
from typing import Any, BinaryIO, List, Optional, Tuple, Union

import copy
import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn.parameter import Parameter, UninitializedParameter
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _reverse_repeat_tuple
from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t
from torch.nn.modules.utils import _single, _pair, _triple, _reverse_repeat_tuple
from torch.nn.modules._functions import SyncBatchNorm as sync_batch_norm
from torch.nn.modules.batchnorm import _NormBase
from torchvision.utils import _log_api_usage_once, _make_ntuple
from torch.nn import init, Module
from torch.nn import functional as Ftup
from utils import Noisy_Inference


class Conv2d(_ConvNd):
    __doc__ = r"""Applies a 2D convolution over an input signal composed of several input
    planes.

    .. _link:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',  # TODO: refine this type
        device=None,
        dtype=None,
        noise_inference=False,
        noise_sd=1e-3
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        kernel_size_ = _pair(kernel_size)
        stride_ = _pair(stride)
        padding_ = padding if isinstance(padding, str) else _pair(padding)
        dilation_ = _pair(dilation)
        super().__init__(
            in_channels, out_channels, kernel_size_, stride_, padding_, dilation_,
            #False, _pair(0), groups, bias, padding_mode, noise_inference, noise_sd, **factory_kwargs)
            False, _pair(0), groups, bias, padding_mode, **factory_kwargs)
        
        # set the noisy inference params
        self.noise_inference = noise_inference
        self.noise_sd = noise_sd
        if noise_inference: 
            Noisy_Inference.noise_sd = noise_sd
            self.noiser = Noisy_Inference.apply

    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            weight, bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input: Tensor) -> Tensor:
        if self.noise_inference:
            return self._conv_forward(input, self.noiser( self.weight ), self.noiser(self.bias) if self.bias is not None else self.bias )
        else:
            return self._conv_forward(input, self.weight, self.bias)



class ConvNormActivation(torch.nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, ...]] = 3,
        stride: Union[int, Tuple[int, ...]] = 1,
        padding: Optional[Union[int, Tuple[int, ...], str]] = None,
        groups: int = 1,
        norm_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.BatchNorm2d,
        activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
        dilation: Union[int, Tuple[int, ...]] = 1,
        inplace: Optional[bool] = True,
        bias: Optional[bool] = None,
        conv_layer: Callable[..., torch.nn.Module] = Conv2d, #torch.nn.Conv2d,
        noise_inference: Optional[bool] = False, 
        noise_inference_bn : Optional[bool] = False,
        noise_sd : Optional[float] = 1e-1
    ) -> None:

        if padding is None:
            if isinstance(kernel_size, int) and isinstance(dilation, int):
                padding = (kernel_size - 1) // 2 * dilation
            else:
                _conv_dim = len(kernel_size) if isinstance(kernel_size, Sequence) else len(dilation)
                kernel_size = _make_ntuple(kernel_size, _conv_dim)
                dilation = _make_ntuple(dilation, _conv_dim)
                padding = tuple((kernel_size[i] - 1) // 2 * dilation[i] for i in range(_conv_dim))
        if bias is None:
            bias = norm_layer is None

        layers = [
            conv_layer(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
                noise_inference=noise_inference,
                noise_sd=noise_sd
            )
        ]

        if norm_layer is not None:
            layers.append(norm_layer(out_channels, noise_inference=noise_inference_bn, noise_sd=noise_sd))

        if activation_layer is not None:
            params = {} if inplace is None else {"inplace": inplace}
            layers.append(activation_layer(**params))
        super().__init__(*layers)
        _log_api_usage_once(self)
        self.out_channels = out_channels

        if self.__class__ == ConvNormActivation:
            warnings.warn(
                "Don't use ConvNormActivation directly, please use Conv2dNormActivation and Conv3dNormActivation instead."
            )


class Conv2dNormActivation(ConvNormActivation):
    """
    Configurable block used for Convolution2d-Normalization-Activation blocks.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]] = 3,
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Optional[Union[int, Tuple[int, int], str]] = None,
        groups: int = 1,
        norm_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.BatchNorm2d,
        activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
        dilation: Union[int, Tuple[int, int]] = 1,
        inplace: Optional[bool] = True,
        bias: Optional[bool] = None,
        noise_inference: Optional[bool] = False, 
        noise_inference_bn: Optional[bool] = False, 
        noise_sd : Optional[float] = 1e-1,
    ) -> None:

        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups,
            norm_layer,
            activation_layer,
            dilation,
            inplace,
            bias,
            #noise_inference, 
            #noise_sd,
            Conv2d, #torch.nn.Conv2d,
        )

        if noise_inference:
            self[0].noise_inference = noise_inference
            self[0].noise_sd = noise_sd
            Noisy_Inference.noise_sd = noise_sd
            self[0].noiser = Noisy_Inference.apply

        if noise_inference_bn:
            self[1].noise_inference = noise_inference_bn
            self[1].noise_sd = noise_sd
            Noisy_Inference.noise_sd = noise_sd
            self[1].noiser = Noisy_Inference.apply
        

############################################
# BATCHNORM WITH NOISE ON WEIGHTS AND BIASES
############################################

class _BatchNorm(_NormBase):
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = False, ##### BE CAREFUL, THIS WAS SET TO TRUE, but when False it helps dealing with variability of parameters
        device=None,
        dtype=None,
        noise_inference: bool = False,
        noise_sd=0.1,
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(
            num_features, eps, momentum, affine, track_running_stats, **factory_kwargs
        )


        # set the noisy inference params
        self.noise_inference = noise_inference
        self.noise_sd = noise_sd
        Noisy_Inference.noise_sd = noise_sd
        self.noiser = Noisy_Inference.apply

    def forward(self, input: Tensor) -> Tensor:
        self._check_input_dim(input)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:  # type: ignore[has-type]
                self.num_batches_tracked.add_(1)  # type: ignore[has-type]
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        r"""
        Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        r"""
        Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """
        if self.noise_inference and self.track_running_stats:
            running_mean = self.noiser( self.running_mean )
            running_var  = self.noiser( self.running_var )
        elif self.track_running_stats and not self.noise_inference:
            running_mean = self.running_mean
            running_var  = self.running_var
        else:
            running_mean = None
            running_var  = None

        return F.batch_norm(
            input,
            # If buffers are not to be tracked, ensure that they won't be updated
            running_mean if not self.training or self.track_running_stats else None,
            running_var if not self.training or self.track_running_stats else None,
            self.noiser(self.weight) if self.noise_inference and self.weight is not None else self.weight,
            self.noiser(self.bias) if self.noise_inference and self.bias is not None else self.bias,
            bn_training,
            exponential_average_factor,
            self.eps,
        )
    

class BatchNorm2d(_BatchNorm):
    r"""Applies Batch Normalization over a 4D input (a mini-batch of 2D inputs
    with additional channel dimension) as described in the paper
    `Batch Normalization: Accelerating Deep Network Training by Reducing
    Internal Covariate Shift <https://arxiv.org/abs/1502.03167>`__ .

    .. math::

        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    Shape:
        - Input: :math:`(N, C, H, W)`
        - Output: :math:`(N, C, H, W)` (same shape as input)
    """

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError("expected 4D input (got {}D input)".format(input.dim()))
