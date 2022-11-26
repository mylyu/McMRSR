import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import log10
from torch.optim import lr_scheduler
import scipy.io as sio
import pdb
from torchmetrics import StructuralSimilarityIndexMeasure

from typing import List, Optional
import torch.fft


def fft2c_new(data: torch.Tensor, norm: str = "ortho") -> torch.Tensor:
    """
    Apply centered 2 dimensional Fast Fourier Transform.

    Args:
        data: Complex valued input data containing at least 3 dimensions:
            dimensions -3 & -2 are spatial dimensions and dimension -1 has size
            2. All other dimensions are assumed to be batch dimensions.
        norm: Normalization mode. See ``torch.fft.fft``.

    Returns:
        The FFT of the input.
    """
    if not data.shape[-1] == 2:
        raise ValueError("Tensor does not have separate complex dim.")

    data = ifftshift(data, dim=[-3, -2])
    data = torch.view_as_real(
        torch.fft.fftn(  # type: ignore
            torch.view_as_complex(data), dim=(-2, -1), norm=norm
        )
    )
    data = fftshift(data, dim=[-3, -2])

    return data


def ifft2c_new(data: torch.Tensor, norm: str = "ortho") -> torch.Tensor:
    """
    Apply centered 2-dimensional Inverse Fast Fourier Transform.

    Args:
        data: Complex valued input data containing at least 3 dimensions:
            dimensions -3 & -2 are spatial dimensions and dimension -1 has size
            2. All other dimensions are assumed to be batch dimensions.
        norm: Normalization mode. See ``torch.fft.ifft``.

    Returns:
        The IFFT of the input.
    """
    if not data.shape[-1] == 2:
        raise ValueError("Tensor does not have separate complex dim.")

    data = ifftshift(data, dim=[-3, -2])
    data = torch.view_as_real(
        torch.fft.ifftn(  # type: ignore
            torch.view_as_complex(data), dim=(-2, -1), norm=norm
        )
    )
    data = fftshift(data, dim=[-3, -2])

    return data


# Helper functions


def roll_one_dim(x: torch.Tensor, shift: int, dim: int) -> torch.Tensor:
    """
    Similar to roll but for only one dim.

    Args:
        x: A PyTorch tensor.
        shift: Amount to roll.
        dim: Which dimension to roll.

    Returns:
        Rolled version of x.
    """
    shift = shift % x.size(dim)
    if shift == 0:
        return x

    left = x.narrow(dim, 0, x.size(dim) - shift)
    right = x.narrow(dim, x.size(dim) - shift, shift)

    return torch.cat((right, left), dim=dim)


def roll(
    x: torch.Tensor,
    shift: List[int],
    dim: List[int],
) -> torch.Tensor:
    """
    Similar to np.roll but applies to PyTorch Tensors.

    Args:
        x: A PyTorch tensor.
        shift: Amount to roll.
        dim: Which dimension to roll.

    Returns:
        Rolled version of x.
    """
    if len(shift) != len(dim):
        raise ValueError("len(shift) must match len(dim)")

    for (s, d) in zip(shift, dim):
        x = roll_one_dim(x, s, d)

    return x


def fftshift(x: torch.Tensor, dim: Optional[List[int]] = None) -> torch.Tensor:
    """
    Similar to np.fft.fftshift but applies to PyTorch Tensors

    Args:
        x: A PyTorch tensor.
        dim: Which dimension to fftshift.

    Returns:
        fftshifted version of x.
    """
    if dim is None:
        # this weird code is necessary for toch.jit.script typing
        dim = [0] * (x.dim())
        for i in range(1, x.dim()):
            dim[i] = i

    # also necessary for torch.jit.script
    shift = [0] * len(dim)
    for i, dim_num in enumerate(dim):
        shift[i] = x.shape[dim_num] // 2

    return roll(x, shift, dim)


def ifftshift(x: torch.Tensor, dim: Optional[List[int]] = None) -> torch.Tensor:
    """
    Similar to np.fft.ifftshift but applies to PyTorch Tensors

    Args:
        x: A PyTorch tensor.
        dim: Which dimension to ifftshift.

    Returns:
        ifftshifted version of x.
    """
    if dim is None:
        # this weird code is necessary for toch.jit.script typing
        dim = [0] * (x.dim())
        for i in range(1, x.dim()):
            dim[i] = i

    # also necessary for torch.jit.script
    shift = [0] * len(dim)
    for i, dim_num in enumerate(dim):
        shift[i] = (x.shape[dim_num] + 1) // 2

    return roll(x, shift, dim)


def get_nonlinearity(name):
    """Helper function to get non linearity module, choose from relu/softplus/swish/lrelu"""
    if name == 'relu':
        return nn.ReLU(inplace=True)
    elif name == 'softplus':
        return nn.Softplus()
    elif name == 'swish':
        return Swish(inplace=True)
    elif name == 'lrelu':
        return nn.LeakyReLU()


class Swish(nn.Module):
    def __init__(self, inplace=False):
        """The Swish non linearity function"""
        super().__init__()
        self.inplace = True

    def forward(self, x):
        if self.inplace:
            x.mul_(F.sigmoid(x))
            return x
        else:
            return x * F.sigmoid(x)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_scheduler(optimizer, opts, last_epoch=-1):
    if 'lr_policy' not in opts or opts.lr_policy == 'constant':
        scheduler = None
    elif opts.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opts.step_size,
                                        gamma=opts.gamma, last_epoch=last_epoch)
    elif opts.lr_policy == 'lambda':
        def lambda_rule(ep):
            lr_l = 1.0 - max(0, ep - opts.epoch_decay) / float(opts.n_epochs - opts.epoch_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule, last_epoch=last_epoch)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opts.lr_policy)
    return scheduler


def get_recon_loss(opts):
    loss = None
    if opts['recon'] == 'L2':
        loss = nn.MSELoss()
    elif opts['recon'] == 'L1':
        loss = nn.L1Loss()

    return loss


def psnr(sr_image, gt_image):
    assert sr_image.size(0) == gt_image.size(0) == 1

    peak_signal = (gt_image.max() - gt_image.min()).item()

    mse = (sr_image - gt_image).pow(2).mean().item()

    return 10 * log10(peak_signal ** 2 / mse)


def mse(sr_image, gt_image):
    assert sr_image.size(0) == gt_image.size(0) == 1

    mse = (sr_image - gt_image).pow(2).mean().item()

    return mse

'''
K-Space
'''
def data_consistency(k, k0, mask, noise_lvl=None):
    """
    k    - input in k-space [b,w,h,2] need to [b,2,w,h]
    k0   - initially sampled elements in k-space
    dc_mask - corresponding nonzero location
    """
    k = k.permute(0,3,1,2)
    v = noise_lvl
    if v:  # noisy case
        out = (1 - mask) * k + mask * (k + v * k0) / (1 + v)
    else:  # noiseless case
        out = (1 - mask) * k + mask * k0
    return out


class DataConsistencyInKspace_I(nn.Module):
    """ Create data consistency operator

    Warning: note that FFT2 (by the default of torch.fft) is applied to the last 2 axes of the input.
    This method detects if the input tensor is 4-dim (2D data) or 5-dim (3D data)
    and applies FFT2 to the (nx, ny) axis.

    """

    def __init__(self, noise_lvl=None):
        super(DataConsistencyInKspace_I, self).__init__()
        self.noise_lvl = noise_lvl

    def forward(self, *input, **kwargs):
        return self.perform(*input)

    def perform(self, x, k0, mask):
        """
        x    - input in image domain, of shape (n, 2, nx, ny)
        k0   - initially sampled elements in k-space
        dc_mask - corresponding nonzero location
        """

        if x.dim() == 4: # input is 2D
            x = x.permute(0, 2, 3, 1) #[n,w,h,2]
        else:
            raise ValueError("error in data consistency layer!")

        k = fft2(x)
        # out = data_consistency(k, k0, dc_mask.repeat(1, 1, 1, 2), self.noise_lvl)
        out = data_consistency(k, k0, mask, self.noise_lvl)
        x_res = ifft2(out) #[b,2,w,h]

        if x.dim() == 4:
            # x_res = x_res.permute(0, 3, 1, 2)
            x_res = x_res
        else:
            raise ValueError("Iuput dimension is wrong, it has to be a 2D input!")

        return x_res, out


class DataConsistencyInKspace_K(nn.Module):
    """ Create data consistency operator

    Warning: note that FFT2 (by the default of torch.fft) is applied to the last 2 axes of the input.
    This method detects if the input tensor is 4-dim (2D data) or 5-dim (3D data)
    and applies FFT2 to the (nx, ny) axis.

    """

    def __init__(self, noise_lvl=None):
        super(DataConsistencyInKspace_K, self).__init__()
        self.noise_lvl = noise_lvl

    def forward(self, *input, **kwargs):
        return self.perform(*input)

    def perform(self, k, k0, mask):
        """
        k    - input in frequency domain, of shape (n, 2, nx, ny)
        k0   - initially sampled elements in k-space
        dc_mask - corresponding nonzero location
        """

        if k.dim() == 4:  # input is 2D [b,2,w,h]
            k = k.permute(0, 2, 3, 1) #[b,w,h,2]
        else:
            raise ValueError("error in data consistency layer!")

        out = data_consistency(k, k0, mask, self.noise_lvl) #[b,2,w,h]
        x_res = ifft2(out) #[b,2,w,h]
        # ========
        # ks_net_fin_out = x_res.cpu().detach().numpy()
        # sio.savemat('ks_net_fin_out.mat', {'data': ks_net_fin_out});
        # ========

        if k.dim() == 4:
            # x_res = x_res.permute(0, 3, 1, 2)
            x_res = x_res
        else:
            raise ValueError("Iuput dimension is wrong, it has to be a 2D input!")

        return x_res, out

# Basic functions / transforms
def to_tensor(data):
    """
    Convert numpy array to PyTorch tensor. For complex arrays, the real and imaginary parts
    are stacked along the last dimension.
    Args:
        data (np.array): Input numpy array
    Returns:
        torch.Tensor: PyTorch version of data
    """
    if np.iscomplexobj(data):
        data = np.stack((data.real, data.imag), axis=-1)
    return torch.from_numpy(data)


def fft2(data):
    """
    Apply centered 2 dimensional Fast Fourier Transform.
    input w, h,2 or b,w,h,2
    output w, h,2 or b,w,h,2
    Args:
        data (torch.Tensor): Complex valued input data containing at least 3 dimensions: dimensions
            -3 & -2 are spatial dimensions and dimension -1 has size 2. All other dimensions are
            assumed to be batch dimensions.[w,h,2]
    Returns:
        torch.Tensor: The FFT of the input.
    """
#     old_shape = data.size()
#     if data.dim() == 3:
#         data = data.unsqueeze(0)
    
#     data = data.permute(0, 2, 3, 1) #[b, w, h, 2]
    # print(data.size())
    # assert data.size(-1) == 2
    # data = ifftshift(data, dim=(-3, -2))
    # data = torch.fft.fft(data, 2)
    # data = fftshift(data, dim=(-3, -2))
    data = fft2c_new(data)
    
    # data = data.permute(0, 3, 1, 2) #[b, 2, w, h]
    # data = data.reshape(old_shape)
    return data

def fft2_net(data):
    """
    Apply centered 2 dimensional Fast Fourier Transform.
    input b,2, w, h need to b,w, h, 2
    output b,2, w, h
    Args:
        data (torch.Tensor): Complex valued input data containing at least 3 dimensions: dimensions
            -3 & -2 are spatial dimensions and dimension -1 has size 2. All other dimensions are
            assumed to be batch dimensions.[w,h,2]
    Returns:
        torch.Tensor: The FFT of the input.
    """
    # data = data.permute(1,2,0) #[w,h,2]
    print('warning, using fft2_net')
    # assert data.size(-1) == 2
    # print(data.size())
    data = data.permute(0, 2, 3, 1) #[b,w,h,2]
    # data = ifftshift(data, dim=(-3, -2))
    # data = torch.fft.fft(data, 2)
    # data = fftshift(data, dim=(-3, -2))
    data = fft2c_new(data)
    data = data.permute(0, 3, 1, 2) #[b,2,w,h]
    # data = data.permute(2,0,1) #[2,w,h]
    return data


def ifft2(data):
    """
    Apply centered 2-dimensional Inverse Fast Fourier Transform.
    Args:
        data (torch.Tensor): Complex valued input data containing at least 3 dimensions: dimensions
            -2 & -1 are spatial dimensions and dimension -3 has size 2. All other dimensions are
            assumed to be batch dimensions.
    Returns:
        torch.Tensor: The IFFT of the input [b,2,w,h].
    """
    # assert data.size(-1) == 2
    # old_shape = data.size()
    # if data.dim() == 3:
    #     data = data.unsqueeze(0)
    # data = data.permute(0, 2, 3, 1) #[b, w, h, 2]

    # assert data.size(-1) == 2
    # data = ifftshift(data, dim=(-3, -2))
    # data = torch.fft.fft(data, 2)
    # data = fftshift(data, dim=(-3, -2))
    permute_flag = 0
    if data.size(-1) != 2 and data.size(1) ==2:
        permute_flag = 1
        data = data.permute(0, 2, 3, 1) #[b, w, h, 2]
    data = ifft2c_new(data)
    if permute_flag:
        data = data.permute(0, 3, 1, 2) #[b, 2, w, h]
    
#     data = data.permute(0, 3, 1, 2) #[b, 2, w, h]
#     data = data.reshape(old_shape)
    return data


# def roll(x, shift, dim):
#     """
#     Similar to np.roll but applies to PyTorch Tensors
#     """
#     if isinstance(shift, (tuple, list)):
#         assert len(shift) == len(dim)
#         for s, d in zip(shift, dim):
#             x = roll(x, s, d)
#         return x
#     shift = shift % x.size(dim)
#     if shift == 0:
#         return x
#     left = x.narrow(dim, 0, x.size(dim) - shift)
#     right = x.narrow(dim, x.size(dim) - shift, shift)
#     return torch.cat((right, left), dim=dim)


# def fftshift(x, dim=None):
#     """
#     Similar to np.fft.fftshift but applies to PyTorch Tensors
#     """
#     if dim is None:
#         dim = tuple(range(x.dim()))
#         shift = [dim // 2 for dim in x.shape]
#     elif isinstance(dim, int):
#         shift = x.shape[dim] // 2
#     else:
#         shift = [x.shape[i] // 2 for i in dim]
#     return roll(x, shift, dim)


# def ifftshift(x, dim=None):
#     """
#     Similar to np.fft.ifftshift but applies to PyTorch Tensors
#     """
#     if dim is None:
#         dim = tuple(range(x.dim()))
#         shift = [(dim + 1) // 2 for dim in x.shape]
#     elif isinstance(dim, int):
#         shift = (x.shape[dim] + 1) // 2
#     else:
#         shift = [(x.shape[i] + 1) // 2 for i in dim]
#     return roll(x, shift, dim)


def complex_abs(data):
    """
    Compute the absolute value of a complex valued input tensor.
    Args:
        data (torch.Tensor): A complex valued tensor, where the size of the final dimension
            should be 2.
    Returns:
        torch.Tensor: Absolute value of data
    """
    assert data.size(-1) == 2
    return (data ** 2).sum(dim=-1).sqrt()


def complex_abs_eval(data):
    assert data.size(1) == 2
    return (torch.abs(data[:, 0:1, :, :]) ** 2 + torch.abs(data[:, 1:2, :, :]) ** 2).sqrt()


def to_spectral_img(data):
    """
    Compute the spectral images of a kspace data
    with keeping each column for creation of one spectral image
    Args:
        data (torch.Tensor): Complex valued input data containing at least 3 dimensions: dimensions
            -3 & -2 are spatial dimensions and dimension -1 has size 2. All other dimensions are
            assumed to be batch dimensions.
    Returns:
        torch.Tensor: The IFFT of the input.
    """
    assert data.size(-1) == 2

    spectral_vol = torch.zeros([data.size(-2), data.size(-2), data.size(-2)])

    for i in range(data.size(-2)):
        kspc1 = torch.zeros(data.size())
        kspc1[:, i, :] = data[:, i, :]
        img1 = ifft2(kspc1)
        img1_abs = complex_abs(img1)

        spectral_vol[i, :, :] = img1_abs

    return spectral_vol