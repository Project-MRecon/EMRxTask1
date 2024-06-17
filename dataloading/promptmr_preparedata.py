
import os
import sys
import pathlib
sys.path.insert(0, os.path.dirname(os.path.dirname(pathlib.Path(__file__).parent.absolute())))

import shutil
import argparse
import numpy as np
import h5py
import glob
from os.path import join
from tqdm import tqdm

import fastmri
from fastmri.data import transforms as T
import torch

def ifft2c(kdata_tensor, dim=(-2,-1), norm='ortho'):
    """
    ifft2c -  ifft2 from centered kspace data tensor
    """
    kdata_tensor_uncentered = torch.fft.fftshift(kdata_tensor,dim=dim)
    image_uncentered = torch.fft.ifft2(kdata_tensor_uncentered,dim=dim, norm=norm)
    image = torch.fft.fftshift(image_uncentered,dim=dim)
    return image

def zf_recon(filename):
    '''
    load kdata and direct IFFT + RSS recon
    return shape [t,z,y,x]
    '''
    kdata = h5py.File(filename)["kspace_full"]
    kdata = kdata['real'] + 1j*kdata['imag']
    kdata_tensor = torch.tensor(kdata).cuda()
    image_coil = ifft2c(kdata_tensor)
    image = (image_coil.abs()**2).sum(2)**0.5
    image_np = image.cpu().numpy()
    return kdata, image_np

def split_train_val(h5_folder, train_num=100):
    train_folder = join(h5_folder,'train')
    val_folder = join(h5_folder,'val')

    if not os.path.exists(train_folder):
        os.makedirs(train_folder)
        
    if not os.path.exists(val_folder):
        os.makedirs(val_folder)

    if os.path.exists(h5_folder):
        num_folders = len(os.listdir(h5_folder))

        for i in range(1, num_folders+1):
            case_folder = join(h5_folder, f"P{i:03d}")
            
            if os.path.exists(case_folder):
                if i<=train_num:
                    shutil.move(case_folder, train_folder)
                else:
                    shutil.move(case_folder, val_folder)


if __name__ == '__main__':
   
    data_path = '/homes/syli/dataset/MultiCoil/Cine/TrainingSet'
    save_folder_name = '/homes/ljchen/data/cmrecon_temp'

    fully_cine_matlab_folder = join(data_path, "FullSample")

    assert os.path.exists(fully_cine_matlab_folder), f"Path {fully_cine_matlab_folder} does not exist."


    # 0. get input file list
    f_cine = sorted(glob.glob(join(fully_cine_matlab_folder, '**/cine_lax.mat'), recursive=True))
    f_mask = sorted(glob.glob(join(data_path, 'Mask_Task1', '**/cine_lax_mask_Uniform4.mat'), recursive=True))
    f = f_cine
    print('total number of files: ', len(f))
    print('cine cases: ', len(os.listdir(fully_cine_matlab_folder)),' , cine files: ', len(f_cine))
    # 1. save as fastMRI style h5py files
    for i, ff in enumerate(tqdm(f)):
        if i == 3:
            break
        save_path = os.path.join(save_folder_name, os.path.basename(os.path.dirname(ff))+".h5py")
        if not os.path.isdir(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        kdata, image = zf_recon(ff)
        mask_path = ff.replace('FullSample', 'Mask_Task1').replace(".mat",'_mask_Uniform4.mat')

        # Open the HDF5 file in write mode
        file = h5py.File(save_path, 'w')

        # Create a dataset 
        # kdata is of shape (time, slice, coil, phase_enc, readout) for cine data; and (contrast, slice, coil, phase_enc, readout) for mapping data
        # we need to reshape and transpose it to (time* slice, coil, readout, phase_enc) as 'kspace' for fastMRI style
        save_kdata = kdata.reshape(-1,kdata.shape[2],kdata.shape[3],kdata.shape[4]).transpose(0,1,3,2)
        print(save_kdata.shape)
        file.create_dataset('kspace', data=save_kdata)

        mask = h5py.File(mask_path)["mask"]
        file.create_dataset('mask', data=mask)

        # image is of shape (time, slice, phase_enc, readout) for cine data; and (contrast, slice, phase_enc, readout) for mapping data
        # we need to reshape and transpose it to (time * slice, readout, phase_enc) as 'reconstruction_rss' for fastMRI style
        save_image = image.reshape(-1,image.shape[2],image.shape[3]).transpose(0,2,1)
        file.create_dataset('reconstruction_rss', data=save_image)
        file.create_dataset("num_low_frequencies", data=16)
        # file.attrs['max'] = image.max()
        # file.attrs['norm'] = np.linalg.norm(image)

        # Add attributes to the dataset

        # file.attrs['patient_id'] = os.path.basename(save_path[:-5])
        # file.attrs['shape'] = kdata.shape
        # file.attrs['padding_left'] = 0
        # file.attrs['padding_right'] = save_kdata.shape[3]
        # file.attrs['encoding_size'] = (save_kdata.shape[2],save_kdata.shape[3],1)
        # print(file.attrs['encoding_size'])
        # file.attrs['recon_size'] = (save_kdata.shape[2],save_kdata.shape[3],1)
        

        # Close the file
        file.close()
    
