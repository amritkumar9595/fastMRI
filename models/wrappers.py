from data.transforms import ifft2,fft2,fftshift,complex_abs
from torch import nn

def data_consistency(k, k0, mask, noise_lvl=None):
    """
    k    - input in k-space (predicted filling)
    k0   - initially sampled elements in k-space
    mask - corresponding nonzero location
    """
    v = noise_lvl
    if v is not None:  # noisy case
        out = (1 - mask) * k + mask * (k + v * k0) / (1 + v)
    else:  # noiseless case
        out = (1 - mask) * k + mask * k0
    return out

ifft_func = lambda ksp: ifft2(ksp.permute((0,2,3,1))).permute((0,3,1,2))
fft_func = lambda im:fft2(im.permute(0,2,3,1)).permute((0,3,1,2))


class DataConsistencyInKspace(nn.Module):
    """ Create data consistency operator
    Warning: note that FFT2 (by the default of torch.fft) is applied to the last 2 axes of the input.
    This method detects if the input tensor is 4-dim (2D data) or 5-dim (3D data)
    and applies FFT2 to the (nx, ny) axis.
    """

    def __init__(self, noise_lvl=None, norm='ortho'):
        super(DataConsistencyInKspace, self).__init__()
        self.normalized = norm == 'ortho'
        self.noise_lvl = noise_lvl

    def forward(self, *input, **kwargs):
        return self.perform(*input)

    def perform(self, x, k0, mask):
        """
        x    - input in image domain, of shape (n, 2, nx, ny[, nt])
        k0   - initially sampled elements in k-space
        mask - corresponding nonzero location
        """

        # if x.dim() == 4: # input is 2D
        # x    = x.permute(0, 2, 3, 1)
        # k0   = k0.permute(0, 2, 3, 1)
        # mask = mask.permute(0, 2, 3, 1)

        # k = fftshift(torch.fft(fftshift(x,2), 2, normalized=self.normalized),2)
        k = fft_func(x) 
        out = data_consistency(k, k0, mask, self.noise_lvl)
        # x_res = fftshift(torch.ifft(ifftshift(out,2), 2, normalized=self.normalized),2)
        x_res = ifft_func(out)
        # if x.dim() == 4:
        # x_res = x_res.permute(0, 3, 1, 2)

        return x_res

class ResidualForm(nn.Module):
    """for inline written sequential models at refinement block"""
    def __init__(self, module):
        super(ResidualForm,self).__init__()
        self.module_ = module

    def forward(self, inp):
        return self.module_(inp)+inp


class ModelWithDC(nn.Module):
    def __init__(self, module):
        super(ModelWithDC,self).__init__()
        self.module_ = module
        self.dcs = DataConsistencyInKspace()
    
    def forward(self,inp, ksp):
        msk = (~(ksp==0)).float()
        return self.dcs(self.module_(inp), ksp, msk)

