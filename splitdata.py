import glob
import h5py
import torch
from matplotlib import pyplot as plt
import numpy as np
import os
import shutil

from data import transforms as T


def c3_multiplier_npy(shape=(320,320)):
    shp = (shape[0],shape[1])

    mul_mat=np.resize([1,-1],shp)
    
    return mul_mat * mul_mat.T


def c3_torch(shp): 
    c3m = c3_multiplier_npy(shp)
    return torch.from_numpy(np.dstack((c3m,c3m))).float()

ifft_c3 = lambda kspc3: torch.ifft(T.ifftshift(kspc3,dim=(-3,-2)),2,normalized=True)

fft_c3 = lambda im: T.fftshift(torch.fft(im,2,normalized=True),dim=(-3,-2))


shp = (360,360)


c3m = c3_torch(shp)


def tosquare(ksp,shp):
    rec = T.ifft2(ksp)
    sz = rec.shape
    
    return c3m * T.fft2(T.complex_center_crop(rec,shp)) * 100000


datadir = '../multicoil_val/multicoil_val/'
outdir = datadir+'/../multicoil_val2/'

completed = []
for fi in glob.glob(datadir+'/*.h5'):
    with h5py.File(fi,'r') as h5:
        print(fi)
        volume_ksp = h5['kspace']
        print(volume_ksp.shape,volume_ksp.dtype)
        nslice,nch, ht, wd = volume_ksp.shape
        
        if wd < shp[1]:
            continue

        for sl in range(2,nslice-2):
            ksp = T.to_tensor(volume_ksp[sl])
            sq = tosquare(ksp,shp)
            
            with h5py.File('%s/%s-%.2d.h5' % (outdir,os.path.basename(fi)[:-3],sl),'w') as hw:
                hw['kspace']=sq.numpy()
        completed.append(fi)

    if len(completed)%3==0:    
        for rfi in completed:
            if os.path.exists(rfi):
                os.unlink(rfi)
#         break        

