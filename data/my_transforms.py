import numpy as np

from data import transforms

import torch


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


class C3Convert:
    def __init__(self, shp=(320,320)):
        self.c3m = transforms.c3_torch(shp) # ksp.shape[-2:]

    def apply(self,ksp):
        # expect (bat, w, h, 2)
        return ksp * self.c3m


# torch fft requires (bat, w,h, 2)

ifft_c3 = lambda kspc3: torch.ifft(transforms.ifftshift(kspc3,dim=(-3,-2)),2,normalized=True)

fft_c3 = lambda im: transforms.fftshift(torch.fft(im,2,normalized=True),dim=(-3,-2))

class NoTransform:
    def __init__(self):
        pass

    def __call__(self, kspace, target, attrs, fname, slice):
        return kspace


class BasicMaskingTransform:
    
    def __init__(self, mask_func, resolution, which_challenge, use_seed=True, augment=True):
        self.mask_func = mask_func
        self.use_seed = use_seed
        #self.resolution = resolution


    def __call__(self, kspace, target, attrs, fname, slice):
        kspace_square = transforms.to_tensor(kspace)   ##kspace
        image_square = ifft_c3(kspace_square)

        seed = None if not self.use_seed else tuple(map(ord, fname))        
        masked_kspace_square, mask = transforms.apply_mask2(kspace_square, self.mask_func, seed) ##ZF square kspace
        image_square_us = ifft_c3(masked_kspace_square)   ## US square complex imagea
        sz = kspace_square.size
        stacked_kspace_square = kspace_square.permute((0,3,1,2)).reshape((-1,sz(-3),sz(-2)))
        stacked_masked_kspace_square = masked_kspace_square.permute((0,3,1,2)).reshape((-1,sz(-3),sz(-2)))

        stacked_image_square = image_square.permute((0,3,1,2)).reshape((-1,sz(-3),sz(-2)))
        us_image_square_rss = torch.sqrt(torch.sum(image_square_us**2,dim=(0,-1)))
        image_square_rss = torch.sqrt(torch.sum(image_square**2,dim=(0,-1)))

        return stacked_kspace_square,stacked_masked_kspace_square , stacked_image_square , \
            us_image_square_rss ,   \
            image_square_rss 
            #target *10000 \


class SquareDataTransformC3:
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

        self.augmentation = None
        if augment:
            self.augmentation = Augmentation((self.resolution,self.resolution))
                
        self.c3object= C3Convert((self.resolution,self.resolution))


    def __call__(self, kspace, target, attrs, fname, slice):
        kspace_rect = transforms.to_tensor(kspace)   ##rectangular kspace

        image_rect = transforms.ifft2(kspace_rect)    ##rectangular FS image
        image_square = transforms.complex_center_crop(image_rect, (self.resolution, self.resolution))  ##cropped to FS square image
        kspace_square = self.c3object.apply(transforms.fft2(image_square)) #* 10000  ##kspace of square iamge

        if self.augmentation:
            kspace_square = self.augmentation.apply(kspace_square)

        image_square = ifft_c3(kspace_square)

        # Apply mask
        seed = None if not self.use_seed else tuple(map(ord, fname))        
        masked_kspace_square, mask = transforms.apply_mask(kspace_square, self.mask_func, seed) ##ZF square kspace
    
        # Inverse Fourier Transform to get zero filled solution
        # image = transforms.ifft2(masked_kspace)
        image_square_us = ifft_c3(masked_kspace_square)   ## US square complex image

        # Crop input image
        # image = transforms.complex_center_crop(image, (self.resolution, self.resolution))
        # Absolute value
        # image = transforms.complex_abs(image)
        image_square_abs = transforms.complex_abs(image_square_us)    ## US square real image
        
        # Apply Root-Sum-of-Squares if multicoil data
        # if self.which_challenge == 'multicoil':
        #     image = transforms.root_sum_of_squares(image)
        # Normalize input
        # image, mean, std = transforms.normalize_instance(image, eps=1e-11)
        _, mean, std = transforms.normalize_instance(image_square_abs, eps=1e-11)
        # image = image.clamp(-6, 6)
        
        # target = transforms.to_tensor(target)        
        target = image_square.permute(2,0,1)
        # Normalize target
        # target = transforms.normalize(target, mean, std, eps=1e-11)
        # target = target.clamp(-6, 6)    
        # return image, target, mean, std, attrs['norm'].astype(np.float32)        

        # return masked_kspace_square.permute((2,0,1)), image, image_square.permute(2,0,1), mean, std, attrs['norm'].astype(np.float32)

        # ksp, zf, target, me, st, nor
        return masked_kspace_square.permute((2,0,1)), image_square_us.permute((2,0,1)), \
            target,  \
            mean, std, attrs['norm'].astype(np.float32)


class SquareDataTransformC3_multi:
    def __init__(self, mask_func, resolution, which_challenge, use_seed=True, augment=False): #default should b true!
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

        self.augmentation = None
        if augment:
            self.augmentation = Augmentation((self.resolution,self.resolution))
                
        self.c3object= C3Convert((self.resolution,self.resolution))


    def __call__(self, kspace, target, attrs, fname, slice):
        kspace_rect = transforms.to_tensor(kspace)   ##rectangular kspace
        
        image_rect = transforms.ifft2(kspace_rect)    ##rectangular FS image
        image_square = transforms.complex_center_crop(image_rect, (self.resolution, self.resolution))  ##cropped to FS square image
        

        kspace_square = self.c3object.apply(transforms.fft2(image_square)) * 10000  ##kspace of square iamge
        image_square2 = ifft_c3(kspace_square)   ##for training domain_transform

        if self.augmentation:
            kspace_square = self.augmentation.apply(kspace_square)

        # image_square = ifft_c3(kspace_square)

        # Apply mask
        seed = None if not self.use_seed else tuple(map(ord, fname))        
        masked_kspace_square, mask = transforms.apply_mask(kspace_square, self.mask_func, seed) ##ZF square kspace

    
        # Inverse Fourier Transform to get zero filled solution
        # image = transforms.ifft2(masked_kspace)
        us_image_square = ifft_c3(masked_kspace_square)   ## US square complex image

        # Crop input image
        # image = transforms.complex_center_crop(image, (self.resolution, self.resolution))
        # Absolute value
        # image = transforms.complex_abs(image)
        us_image_square_abs = transforms.complex_abs(us_image_square)    ## US square real image
        us_image_square_rss = transforms.root_sum_of_squares(us_image_square_abs , dim=0)

        stacked_kspace_square = []
        for i in (range(len(kspace_square[:,0,0,0]))):
            stacked_kspace_square.append(kspace_square[i,:,:,0])
            stacked_kspace_square.append(kspace_square[i,:,:,1])
        
        stacked_kspace_square = torch.stack(stacked_kspace_square)

        stacked_masked_kspace_square = []
        # masked_kspace_square = transforms.to_tensor(masked_kspace_square)
        # for i in range(len(masked_kspace_square[:,0,0,0])):
            # stacked_masked_kspace_square.stack(masked_kspace_square[i,:,:,0],masked_kspace_square[i,:,:,1])

        for i in (range(len(masked_kspace_square[:,0,0,0]))):
            stacked_masked_kspace_square.append(masked_kspace_square[i,:,:,0])
            stacked_masked_kspace_square.append(masked_kspace_square[i,:,:,1])
        
        stacked_masked_kspace_square = torch.stack(stacked_masked_kspace_square)

        stacked_image_square = []
        for i in (range(len(image_square[:,0,0,0]))):
            stacked_image_square.append(image_square2[i,:,:,0])
            stacked_image_square.append(image_square2[i,:,:,1])
        
        stacked_image_square = torch.stack(stacked_image_square)




        return stacked_kspace_square,stacked_masked_kspace_square , stacked_image_square , \
            us_image_square_rss ,   \
            target *10000 \
            #mean, std, attrs['norm'].astype(np.float32)
        




        







        




        '''
        # Apply Root-Sum-of-Squares if multicoil data
        # if self.which_challenge == 'multicoil':
        #     image = transforms.root_sum_of_squares(image)
        # Normalize input
        # image, mean, std = transforms.normalize_instance(image, eps=1e-11)
        _, mean, std = transforms.normalize_instance(image_square_abs, eps=1e-11)
        # image = image.clamp(-6, 6)
        
        # target = transforms.to_tensor(target)        
        target = image_square.permute(2,0,1)
        # Normalize target
        # target = transforms.normalize(target, mean, std, eps=1e-11)
        # target = target.clamp(-6, 6)    
        # return image, target, mean, std, attrs['norm'].astype(np.float32)        

        # return masked_kspace_square.permute((2,0,1)), image, image_square.permute(2,0,1), mean, std, attrs['norm'].astype(np.float32)

        # ksp, zf, target, me, st, nor
        return masked_kspace_square.permute((2,0,1)), image_square_us.permute((2,0,1)), \
            target,  \
            mean, std, attrs['norm'].astype(np.float32)
            '''


            
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

        self.augmentation = None
        if augment:
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
        kspace_rect = transforms.to_tensor(kspace)   ##rectangular kspace

        image_rect = transforms.ifft2(kspace_rect)    ##rectangular FS image
        image_square = transforms.complex_center_crop(image_rect, (self.resolution, self.resolution))  ##cropped to FS square image
        kspace_square = transforms.fft2(image_square)  ##kspace of square iamge

        if self.augmentation:
            kspace_square = self.augmentation.apply(kspace_square)
            image_square = transforms.ifft2(kspace_square)

        # Apply mask
        seed = None if not self.use_seed else tuple(map(ord, fname))        
        masked_kspace_square, mask = transforms.apply_mask(kspace_square, self.mask_func, seed) ##ZF square kspace
    
        # Inverse Fourier Transform to get zero filled solution
        # image = transforms.ifft2(masked_kspace)
        image_square_us = transforms.ifft2(masked_kspace_square)   ## US square complex image

        # Crop input image
        # image = transforms.complex_center_crop(image, (self.resolution, self.resolution))
        # Absolute value
        # image = transforms.complex_abs(image)
        image_square_abs = transforms.complex_abs(image_square_us)    ## US square real image
        
        # Apply Root-Sum-of-Squares if multicoil data
        # if self.which_challenge == 'multicoil':
        #     image = transforms.root_sum_of_squares(image)
        # Normalize input
        # image, mean, std = transforms.normalize_instance(image, eps=1e-11)
        _, mean, std = transforms.normalize_instance(image_square_abs, eps=1e-11)
        # image = image.clamp(-6, 6)
        
        # target = transforms.to_tensor(target)        
        target = image_square.permute(2,0,1)
        # Normalize target
        # target = transforms.normalize(target, mean, std, eps=1e-11)
        # target = target.clamp(-6, 6)    
        # return image, target, mean, std, attrs['norm'].astype(np.float32)        

        # return masked_kspace_square.permute((2,0,1)), image, image_square.permute(2,0,1), mean, std, attrs['norm'].astype(np.float32)

        # ksp, zf, target, me, st, nor
        return masked_kspace_square.permute((2,0,1)), image_square_us.permute((2,0,1)), \
            target,  \
            mean, std, attrs['norm'].astype(np.float32)
