import numpy as np

from data import transforms


def randomflip(ksp):
    outcome = np.random.binomial(1,0.5,3)
    # print(outcome)
    if outcome[0]==1:
        ksp = np.roll(np.fliplr(ksp),(0,1))
    if outcome[1]==1:
        ksp = np.conj(ksp)
    if outcome[2]==1 and False:
        ksp = ksp.T # never
    return ksp

class Augmentation:
    def __init__(self, shp=(320,320)):
        xr = np.linspace(-shp[0]//2,shp[0]//2,num=shp[0])
        yr = np.linspace(-shp[1]//2,shp[1]//2,num=shp[1])

        self.k1,self.k2 = np.meshgrid(xr,yr)
        self.shp = shp
    
    def translation(self, shp=(320,320), txmax=7, tymax=1):

        tx = np.random.randint(-txmax,txmax)
        ty = np.random.randint(-tymax,tymax)

        twopi = 2*np.pi

        # https://www.clear.rice.edu/elec301/Projects01/image_filt/properties.html
        return np.exp(-1j*twopi*(tx*self.k1 + ty*self.k2)/self.shp[0])


    def apply(self,ksp_torch):
        ksp_npy = ksp_torch[:,:,0].numpy()+1j*ksp_torch[:,:,1].numpy()

        return transforms.to_tensor(randomflip(ksp_npy)*self.translation()).float()


class SquareDataTransform:
    """
    Data Transformer for training U-Net models.
    """

    def __init__(self, mask_func, resolution, which_challenge, use_seed=True, augment=True):
        """
        Args:
            mask_func (common.subsample.MaskFunc): A function that can create a mask of
                appropriate shape.
            resolution (int): Resolution of the image.
            which_challenge (str): Either "singlecoil" or "multicoil" denoting the dataset.
            use_seed (bool): If true, this class computes a pseudo random number generator seed
                from the filename. This ensures that the same mask is used for all the slices of
                a given volume every time.
        """
        if which_challenge not in ('singlecoil', 'multicoil'):
            raise ValueError(f'Challenge should either be "singlecoil" or "multicoil"')
        self.mask_func = mask_func
        self.resolution = resolution
        self.which_challenge = which_challenge
        self.use_seed = use_seed

        self.augmentation = Augmentation((self.resolution,self.resolution))

    def __call__(self, kspace, target, attrs, fname, slice):
        """
        Args:
            kspace (numpy.array): Input k-space of shape (num_coils, rows, cols, 2) for multi-coil
                data or (rows, cols, 2) for single coil data.
            target (numpy.array): Target image
            attrs (dict): Acquisition related information stored in the HDF5 object.
            fname (str): File name
            slice (int): Serial number of the slice.
        Returns:
            (tuple): tuple containing:
                image (torch.Tensor): Zero-filled input image.
                target (torch.Tensor): Target image converted to a torch Tensor.
                mean (float): Mean value used for normalization.
                std (float): Standard deviation value used for normalization.
                norm (float): L2 norm of the entire volume.
        """
        ## converting the given data to tensors
        # target = transforms.to_tensor(target)
        kspace_rect = transforms.to_tensor(kspace)   ##rectangular kspace

        seed = None if not self.use_seed else tuple(map(ord, fname))
        # print("shape",kspace_rect.shape)
        image_rect = transforms.ifft2(kspace_rect)    ##rectangular FS image
        image_square = transforms.complex_center_crop(image_rect, (self.resolution, self.resolution))  ##cropped to FS square image
        kspace_square = transforms.fft2(image_square)*10000  ##kspace of square iamge

        kspace_square = self.augmentation.apply(kspace_square)
        image_square = transforms.ifft2(kspace_square)

        masked_kspace_square, mask = transforms.apply_mask(kspace_square, self.mask_func, seed) ##ZF square kspace
        image_square_us = transforms.ifft2(masked_kspace_square)   ## US square complex image
        image_square_abs = transforms.complex_abs(image_square_us)    ## US square real image

        ## normalizing 
        image, mean, std = transforms.normalize_instance(image_square_abs/10000, eps=1e-11)
        # target = transforms.normalize(target, mean, std, eps=1e-11)
        
        ## clamp
        # image = image.clamp(-6, 6)
        # target = target.clamp(-6, 6)
        # masked_kspace_square = masked_kspace_square.clamp(-6,6)


        # print("kspace",masked_kspace_square.shape)
        # print("imagemax",torch.max(image))
        # print("imagemin",torch.min(image))
        # print("target",target.shape)
        
    
        # Inverse Fourier Transform to get zero filled solution
        
        # Crop input image
        
        
        # print("masked_kspace_crop",masked_kspace_crop.shape)
        
        # Absolute value

        # image = transforms.complex_abs(image)
        # Apply Root-Sum-of-Squares if multicoil data
        # if self.which_challenge == 'multicoil':
        #     image = transforms.root_sum_of_squares(image)
        # Normalize input
        # image, mean, std = transforms.normalize_instance(image, eps=1e-11)
        
        # print("image_shape",image.shape)

        
        # Normalize target
        # target = transforms.normalize(target, mean, std, eps=1e-11)
    
        # )
        target = image_square.permute(2,0,1)

        # return masked_kspace_square.permute((2,0,1)), image, image_square.permute(2,0,1), mean, std, attrs['norm'].astype(np.float32)
        # ksp, zf, target, me, st, nor
        return masked_kspace_square.permute((2,0,1)), image_square_us.permute((2,0,1)), \
            target,  \
            mean, std, attrs['norm'].astype(np.float32)
