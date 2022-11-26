import os
import numpy as np
import scipy.io as sio
import torch
import torch.utils.data as data
from datasets.utilizes import *
from models.utils import fft2, ifft2, to_tensor
import h5py
from . import mrfft

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

        self.data_dir_flair = os.path.join(self.data_dir_flair)    # ref kspace directory (T1)
        self.mask_path = opts.mask_path
        self.upscale = opts.upscale
        self.input_contrast = opts.input_contrast
        self.ref_contrast = opts.ref_contrast
    def __getitem__(self, idx):

        mask = sio.loadmat(self.mask_path)['lr_mask']
        mask = mask[np.newaxis,:,:]
        mask = np.concatenate([mask, mask], axis=0)
        mask = torch.from_numpy(mask.astype(np.float32))

        study_name = self.sample_list[idx].strip('\n')
        T2_file = os.path.join(self.data_dir_flair, "{}_{}01.h5".format(study_name,self.input_contrast))
        T1_file = os.path.join(self.data_dir_flair, "{}_{}01.h5".format(study_name,self.ref_contrast))
        if self.mode == 'TRAIN':
            iSlice = np.random.randint(5, 15)
        else:
            iSlice = 9
        # iCoil = 1
        with h5py.File(T2_file) as hf:
            nSlice = hf["kspace"].shape[0]
            nCoil = hf["kspace"].shape[1]
            # iSlice = np.random.randint(0, nSlice)
            ksp = hf["kspace"][iSlice, :, :, :]/100
            # img_mc = mrfft.ifft2c(ksp)
            T2 = mrfft.sos(mrfft.ifft2c(ksp), 0) * np.exp(1j*np.pi*0.25)
            # T2 = mrfft.ifft2c(ksp)
            ksp_64 = crop(ksp, (nCoil, 256//self.upscale, 256//self.upscale))
            T2_64 = mrfft.sos(mrfft.ifft2c(ksp_64)/self.upscale, 0)* np.exp(1j*np.pi*0.25)
            # print(T2_64.shape)
            # T2_64 = mrfft.ifft2c(ksp_64)/self.upscale
        with h5py.File(T1_file) as hf:
            nSlice = hf["kspace"].shape[1]
            # iSlice = np.random.randint(0, nSlice)
            ksp = hf["kspace"][iSlice, :, :, :]/100 
            T1 = mrfft.sos(mrfft.ifft2c(ksp), 0) * np.exp(1j*np.pi*0.25)
            # T1 = mrfft.ifft2c(ksp) 
            ksp_64 = crop(ksp, (nCoil, 256//self.upscale, 256//self.upscale))
            T1_64 = mrfft.sos(mrfft.ifft2c(ksp_64)/self.upscale, 0) * np.exp(1j*np.pi*0.25)
            # T1_64 = mrfft.ifft2c(ksp_64)/self.upscale
        #=======
        T2_256_img_real = T2.real  # ZF
        T2_256_img_real = T2_256_img_real[np.newaxis, :, :]
        T2_256_img_imag = T2.imag
        T2_256_img_imag = T2_256_img_imag[np.newaxis, :, :]
        
        T2_256_img_real = (T2_256_img_real + 1) / 2
        T2_256_img_imag = (T2_256_img_imag + 1) / 2
        
        T2_256_img = np.concatenate([T2_256_img_real, T2_256_img_imag], axis=0)  # 2,w,h
        T2_256_img = to_tensor(T2_256_img).float()  # .permute(2, 0, 1) #  flair zf
        
        
        # =======
        T2_256_img_k = T2_256_img.permute(1, 2, 0)
        T2_256_img_k_ks = fft2(T2_256_img_k)
        T2_256_img_ks = T2_256_img_k_ks.permute(2, 0, 1)
       
        #=======
        T2_64_real = T2_64.real
        T2_64_real = T2_64_real[np.newaxis, :, :]
        T2_64_imag = T2_64.imag
        T2_64_imag = T2_64_imag[np.newaxis, :, :]
        
        T2_64_real = (T2_64_real + 1) / 2
        T2_64_imag = (T2_64_imag + 1) / 2
        
        T2_64_img = np.concatenate([T2_64_real, T2_64_imag], axis=0)
        T2_64_img = to_tensor(T2_64_img).float()
        #=======
        # T2_128_img_k = T2_128_img.permute(1, 2, 0)  # [w,h,2]
        # # print(T2_128_img_k.size())
        # T2_128_ks = fft2(T2_128_img_k)
        # T2_128_ks = T2_128_ks.permute(2,0,1)


        T1_256_real = T1.real
        T1_256_real = T1_256_real[np.newaxis, :, :]
        T1_256_imag = T1.imag
        T1_256_imag = T1_256_imag[np.newaxis, :, :]
        
        T1_256_real = (T1_256_real + 1) / 2
        T1_256_imag = (T1_256_imag + 1) / 2
        
        T1_256_img = np.concatenate([T1_256_real, T1_256_imag], axis=0)
        T1_256_img = to_tensor(T1_256_img).float()
        # =======T1 64
        T1_64_real = T1_64.real
        T1_64_real = T1_64_real[np.newaxis, :, :]
        T1_64_imag = T1_64.imag
        T1_64_imag = T1_64_imag[np.newaxis, :, :]
        
        T1_64_real = (T1_64_real + 1) / 2
        T1_64_imag = (T1_64_imag + 1) / 2
        
        T1_64_img = np.concatenate([T1_64_real, T1_64_imag], axis=0)
        T1_64_img = to_tensor(T1_64_img).float()
        #=======
        # T1_img_k = T1_img.permute(1, 2, 0)
        # T1_ks = fft2(T1_img_k)
        # T1_ks = T1_ks.permute(2, 0, 1)
# ---------------------over------
        return {'ref_image_full': T1_256_img,
                'ref_image_sub': T1_64_img,
                # 'ref_kspace_full': T1_ks,

                'tag_image_full': T2_256_img,
                'tag_kspace_full':T2_256_img_ks,

                'tag_image_sub': T2_64_img,
                # 'tag_image_sub_sub': T2_128_img,
                # 'tag_kspace_sub': T2_128_ks,

                'tag_kspace_mask2d': mask
                }

    def __len__(self):
        return len(self.sample_list)

if __name__ == '__main__':
    a = 1
