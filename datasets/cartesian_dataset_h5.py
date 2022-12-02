import os
import numpy as np
import scipy.io as sio
import torch
import torch.utils.data as data
from datasets.utilizes import *
from models.utils import fft2, ifft2, to_tensor
import h5py
from . import mrfft
import glob

def zpad(array_in, outshape):
    import math
    #out = np.zeros(outshape, dtype=array_in.dtype)
    oldshape = array_in.shape
    assert len(oldshape)==len(outshape)
    #kspdata = np.array(kspdata)
    pad_list=[]
    for iold, iout in zip(oldshape, outshape):
        left = math.floor((iout-iold)/2)
        right = math.ceil((iout-iold)/2)
        pad_list.append((left, right))

    zfill = np.pad(array_in, pad_list, 'constant')                     # fill blade into square with 0
    return zfill

def crop(img, bounding):
    import operator
    start = tuple(map(lambda a, da: a//2-da//2, img.shape, bounding))
    end = tuple(map(operator.add, start, bounding))
    slices = tuple(map(slice, start, end))
    return img[slices].copy()

class MRIDataset_Cartesian(data.Dataset):
    def __init__(self, opts, mode):
        self.mode = mode
        if self.mode == 'TRAIN':
            self.data_dir_flair = os.path.join(opts.data_root)
            self.sample_list = open(os.path.join(opts.list_dir, self.mode + '.txt')).readlines()
            self.seed = None

        if self.mode == 'VALI':
            self.data_dir_flair = os.path.join(opts.data_root)
            self.sample_list = open(os.path.join(opts.list_dir, self.mode + '.txt')).readlines()
            self.seed = 1234

        if self.mode == 'TEST':
            self.data_dir_flair = os.path.join(opts.data_root)
            self.sample_list = open(os.path.join(opts.list_dir, self.mode + '.txt')).readlines()
            self.seed = 5678

        if self.mode == 'TEST_Tumor':
            self.data_dir_flair = os.path.join(opts.data_root, 'test_Tumor')
            self.sample_list = open(os.path.join(opts.list_dir, self.mode + '.txt')).readlines()
            self.seed = 5678

        self.data_dir_flair = os.path.join(self.data_dir_flair)    # ref kspace directory (input)
        self.mask_path = opts.mask_path
        self.sr_factor = opts.sr_factor
        self.input_contrast = opts.input_contrast
        self.ref_contrast = opts.ref_contrast
        self.online_reg = opts.online_reg
        self.multinex = opts.multinex
        if self.online_reg is not None:
            print('online registration.... maybe slow')
        if self.multinex:
            # print(self.multinex)
            print('using multiple nex to train')
    def __getitem__(self, idx):

        mask = sio.loadmat(self.mask_path)['lr_mask']
        mask = mask[np.newaxis,:,:]
        mask = np.concatenate([mask, mask], axis=0)
        mask = torch.from_numpy(mask.astype(np.float32))

        study_name = self.sample_list[idx].strip('\n')
        # choose slices
        if self.mode == 'TRAIN':
            iSlice = np.random.randint(5, 15)
        else:
            iSlice = 9
        
        input_file_list = glob.glob(os.path.join(self.data_dir_flair, "{}_{}0*.h5".format(study_name,self.input_contrast)))
        ref_file_list = glob.glob(os.path.join(self.data_dir_flair, "{}_{}0*.h5".format(study_name,self.ref_contrast)))
        if not self.multinex:
            # just use one nex
            input_file_list = [os.path.join(self.data_dir_flair, "{}_{}01.h5".format(study_name,self.input_contrast))]
            ref_file_list = [os.path.join(self.data_dir_flair, "{}_{}01.h5".format(study_name,self.ref_contrast))]
        
        scale_factor = 100
        nSlice, nCoil, nFe, nPe = 18, 4, 256, 256
        try:
            input_img = np.zeros(shape=(nFe, nPe),dtype=complex)
            input_lowres = np.zeros(shape=(nFe//self.sr_factor, nPe//self.sr_factor),dtype=complex)
            for i_file, input_file in enumerate(input_file_list):
                with h5py.File(input_file) as hf:
                    # nSlice, nCoil, nFe, nPe = hf["kspace"].shape
                    ksp = hf["kspace"][iSlice, :, :, :]/scale_factor
                    input_img_this = mrfft.rsos(mrfft.ifft2c(ksp))
                    # take the average, let us assume there are not motion between input NEXs
                    input_img = (input_img*i_file + input_img_this)/(i_file+1)
                    ksp_lowres_this = crop(ksp, (nCoil, nFe//self.sr_factor, nPe//self.sr_factor))
                    input_lowres_this = mrfft.rsos(mrfft.ifft2c(ksp_lowres_this)/self.sr_factor)
                    input_lowres = (input_lowres*i_file + input_lowres_this)/(i_file+1)
                    
            input_img = input_img * np.exp(1j*np.pi*0.25) # simulate phase here for now
            input_lowres = input_lowres * np.exp(1j*np.pi*0.25)
            # print(input_img.shape)
            # ksp_lowres = crop(mrfft.fft2c(input_img), (nFe//self.sr_factor, nPe//self.sr_factor))
            # input_lowres = mrfft.ifft2c(ksp_lowres)/self.sr_factor
            
            if self.ref_contrast=="self_lowres":
                ref_lowres = input_lowres
                ref_img =  mrfft.ifft2c(mrfft.zpad(mrfft.fft2c(ref_lowres)*self.sr_factor, (nFe, nPe)))
            else:
                ref_img = np.zeros(shape=(nFe, nPe),dtype=complex)
                for i_file, ref_file in enumerate(ref_file_list):
                    with h5py.File(ref_file) as hf:
                        # nSlice, nCoil, _, _ = hf["kspace"].shape
                        ksp = hf["kspace"][iSlice, :, :, :]/scale_factor
                        ref_img_this = mrfft.rsos(mrfft.ifft2c(ksp))
                        ref_img = (ref_img*i_file + ref_img_this)/(i_file+1)

                if self.online_reg is not None:
                    import ants
                    # uses low-res ref and low-res input to get register info
                    fixed = ants.from_numpy(np.abs(input_lowres))
                    fixed = ants.resample_image(fixed,(nPe, nFe), 1, 0)

                    # fixed = ants.from_numpy(np.abs(input_img)) ## not realistic
                    # moving = ants.from_numpy(np.abs(ref_img))

                    temp_ksp_lowres = crop(mrfft.fft2c(ref_img), (nFe//self.sr_factor, nPe//self.sr_factor))
                    temp_ref_lowres = mrfft.ifft2c(temp_ksp_lowres)/self.sr_factor
                    moving = ants.from_numpy(np.abs(temp_ref_lowres))
                    moving = ants.resample_image(moving, (nPe, nFe), 1, 0)
                    mytx = ants.registration(fixed=fixed, moving=moving,
                                             type_of_transform = 'Rigid')
                    ref_img = ants.apply_transforms(fixed=fixed, 
                                                    moving=ants.from_numpy(np.abs(ref_img)),
                                                    transformlist=mytx['fwdtransforms']).numpy()
                ref_img = ref_img * np.exp(1j*np.pi*0.25) # simulate phase here for now        
                ref_ksp_lowres = crop(mrfft.fft2c(ref_img), (nFe//self.sr_factor, nPe//self.sr_factor))
                ref_lowres = mrfft.ifft2c(ref_ksp_lowres)/self.sr_factor
                
        except Exception as e:
            print(e)
            raise e
            print("ERROR at {}, {}, slice {}".format(ref_file, input_file,iSlice))
            shape = (256,256)
            shape_small = (256//self.sr_factor, 256//self.sr_factor)
            ref_img = np.random.uniform(-1, 1, shape) + 1.j * np.random.uniform(-1, 1, shape)
            ref_lowres = np.random.uniform(-1, 1, shape_small) + 1.j * np.random.uniform(-1, 1, shape_small)
            input_img = np.random.uniform(-1, 1, shape) + 1.j * np.random.uniform(-1, 1, shape)
            input_lowres = np.random.uniform(-1, 1, shape_small) + 1.j * np.random.uniform(-1, 1, shape_small)
        #=======
        ref_256_img_real = ref_img.real  # ZF
        ref_256_img_real = ref_256_img_real[np.newaxis, :, :]
        ref_256_img_imag = ref_img.imag
        ref_256_img_imag = ref_256_img_imag[np.newaxis, :, :]
        
        ref_256_img_real = (ref_256_img_real + 1) / 2
        ref_256_img_imag = (ref_256_img_imag + 1) / 2
        
        ref_256_img = np.concatenate([ref_256_img_real, ref_256_img_imag], axis=0)  # 2,w,h
        ref_256_img = to_tensor(ref_256_img).float()  # .permute(2, 0, 1) #  flair zf
        
        
        # =======
        # ref_256_img_k = ref_256_img.permute(1, 2, 0)
        # ref_256_img_k_ks = fft2(ref_256_img_k)
        # ref_256_img_ks = ref_256_img_k_ks.permute(2, 0, 1)
       
        #=======
        ref_lowres_real = ref_lowres.real
        ref_lowres_real = ref_lowres_real[np.newaxis, :, :]
        ref_lowres_imag = ref_lowres.imag
        ref_lowres_imag = ref_lowres_imag[np.newaxis, :, :]
        
        ref_lowres_real = (ref_lowres_real + 1) / 2
        ref_lowres_imag = (ref_lowres_imag + 1) / 2
        
        ref_lowres_img = np.concatenate([ref_lowres_real, ref_lowres_imag], axis=0)
        ref_lowres_img = to_tensor(ref_lowres_img).float()
        #=======
        # ref_128_img_k = ref_128_img.permute(1, 2, 0)  # [w,h,2]
        # # print(ref_128_img_k.size())
        # ref_128_ks = fft2(ref_128_img_k)
        # ref_128_ks = ref_128_ks.permute(2,0,1)


        input_256_real = input_img.real
        input_256_real = input_256_real[np.newaxis, :, :]
        input_256_imag = input_img.imag
        input_256_imag = input_256_imag[np.newaxis, :, :]
        
        input_256_real = (input_256_real + 1) / 2
        input_256_imag = (input_256_imag + 1) / 2
        
        input_256_img = np.concatenate([input_256_real, input_256_imag], axis=0)
        input_256_img = to_tensor(input_256_img).float()
        # =======input lowres
        input_lowres_real = input_lowres.real
        input_lowres_real = input_lowres_real[np.newaxis, :, :]
        input_lowres_imag = input_lowres.imag
        input_lowres_imag = input_lowres_imag[np.newaxis, :, :]
        
        input_lowres_real = (input_lowres_real + 1) / 2
        input_lowres_imag = (input_lowres_imag + 1) / 2
        
        input_lowres_img = np.concatenate([input_lowres_real, input_lowres_imag], axis=0)
        input_lowres_img = to_tensor(input_lowres_img).float()
        #=======
        input_img_k = input_256_img.permute(1, 2, 0)
        input_ks = fft2(input_img_k)
        input_ks = input_ks.permute(2, 0, 1)
# ---------------------over------
        return {'ref_image_full': ref_256_img,
                'ref_image_sub': ref_lowres_img,
                # 'ref_kspace_full': input_ks,

                'tag_image_full': input_256_img,
                'tag_kspace_full': input_ks,

                'tag_image_sub': input_lowres_img,
                # 'tag_image_sub_sub': ref_128_img,
                # 'tag_kspace_sub': ref_128_ks,

                'tag_kspace_mask2d': mask
                }

    def __len__(self):
        return len(self.sample_list)

if __name__ == '__main__':
    a = 1
