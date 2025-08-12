import os
import pickle

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import argparse
import json
import nibabel as nib
import nilearn.image
from tqdm import tqdm

from batchgenerators.utilities.file_and_folder_operations import *

import os 
# os.environ["CUDA_DEVICE_ORDER"] = "3" 

# set nnuet env parameters
os.environ['nnUNet_raw_data_base'] = "./Dataset/nnUNet_raw"
os.environ['nnUNet_preprocessed'] = "./Dataset/nnUNet_preprocessed"
os.environ['RESULTS_FOLDER'] = "./Dataset/nnUNet_trained_models"


from nnunet.training.model_restore import load_model_and_checkpoint_files
from nnunet.inference.segmentation_export import save_segmentation_nifti_from_softmax,save_segmentation_nifti

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-m', "--input_model",required=True, type=argparse.FileType('r'), help='json file of model parameters')
    parser.add_argument('-p', "--input_pet",required=True, type=str, help='Absolute Path of input pet file')
    parser.add_argument('-c', "--input_ct",required=True, type=str, help='Absolute path of input ct file')
    parser.add_argument("-o", "--out_dir",required=True, type=str, help='Absolute path of output mask path')
    
    args = parser.parse_args()

    return args

def resample_ct(pet_nii_in, ct_nii_in, ct_nii_out):
    # resample CT to PET and mask resolution
    # ct   = nib.load(nii_out_path/'CT.nii.gz')
    # pet  = nib.load(nii_out_path/'PET.nii.gz')

    ct = nib.load(ct_nii_in)
    pet = nib.load(pet_nii_in)

    # ct = sitk.ReadImage(ct_nii_in)
    # pet = sitk.ReadImage(pet_nii_in)
    ctres = nilearn.image.resample_to_img(ct, pet, fill_value=-1024)
    # nib.save(CTres, ct_nii_out/'CTres.nii.gz')
    nib.save(ctres, ct_nii_out)

def get_trainer(save_checkpoint_params:dict):
    model_resolution = save_checkpoint_params['resolution']# '3d_fullres'
    model_path = save_checkpoint_params['save_path']
    checkpoint_name = save_checkpoint_params['model']# 'model_best'

    folds = save_checkpoint_params['folds']# [4]
    trainer, params = load_model_and_checkpoint_files(model_path, folds, mixed_precision=True,
                                                      checkpoint_name=checkpoint_name)

    training = False
    trainer.load_checkpoint_ram(params[0], training)
    
    return trainer


def demo(input_pet_file, input_ct_file, output_dir, input_model):
    # argument_args = get_args()
    # input_pet_file = argument_args.input_pet
    # input_ct_file = argument_args.input_ct
    # output_dir = argument_args.out_dir

    # Check if output folder existed, if not then create
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Use pet file name as output mask name
    # pet_file_path, pet_file_fullname=  os.path.split(input_pet_file)
    # pet_name, file_ext = os.path.splitext(pet_file_fullname)
    # if file_ext == ".gz":
    #     pet_name, _ = os.path.splitext(pet_name)

    # print('pet_file_path', pet_file_path)

    ctres_file = input_ct_file

    if not os.path.isfile(ctres_file):
        resample_ct(pet_nii_in=input_pet_file, ct_nii_in=input_ct_file, ct_nii_out=ctres_file)
     
    input_files = [input_pet_file, ctres_file]
    output_seg_file = os.path.join(output_dir,"PRE.nii.gz")
    
    print('output dir : ',output_seg_file)


    f = open(input_model)
    model_parameters = json.load(f)
    f.close()

    # load net_trainer 
    net_trainer = get_trainer(model_parameters)
    
    data, _, properties = net_trainer.preprocess_patient(input_files)

    # Predict params
    mirror_axes = net_trainer.data_aug_params['mirror_axes']
    use_gaussian = True
    all_in_gpu = 'None'  #args.all_in_gpu
    step_size = 0.5  #args.step_size
    mixed_precision = True
    do_tta = True

    #net_trainer.network.decoder.deep_supervision = False
    seg = net_trainer.predict_preprocessed_data_return_seg_and_softmax(
        data, do_mirroring=do_tta, mirror_axes=mirror_axes, use_sliding_window=True,
        step_size=step_size, use_gaussian=use_gaussian, all_in_gpu=all_in_gpu, 
        mixed_precision=mixed_precision
    )[1]
    torch.cuda.empty_cache()
    
    # Nii file save params
    if hasattr(net_trainer, 'regions_class_order'):
        region_class_order = net_trainer.regions_class_order
    else:
        region_class_order = None
    
    npz_file = None
    interpolation_order = 1
    force_separate_z = None
    interpolation_order_z = 0

    # print(seg.shape)
    save_segmentation_nifti_from_softmax(
        seg, output_seg_file, properties, interpolation_order, region_class_order,
        None, None,
        npz_file, None, force_separate_z, interpolation_order_z
    )

if __name__ == "__main__":
    # demo
    # lesion segmentation for one patient

    import os 
    import os 
    # os.environ["CUDA_VISIBLE_DEVICES"]="2"
    input_model = './FDG_model.json' # do not change this

    # adapt the following input data path for your own application
    patient_folder = './Dataset_251/imageT'
    mask_dir = './Dataset_251/labelsT'
    patients_gt_list = os.listdir(mask_dir)
    patients_list = patients_gt_list
    

    patients_list = [x.replace('.nii.gz','') for x in patients_list]


    for patient in tqdm(patients_list):
        
        input_pet_file = os.path.join(patient_folder, patient+'_0000.nii.gz')
        # # input_ct_file = os.path.join(patient_folder,patient, 'CT.nii.gz')
        input_ct_file = os.path.join(patient_folder,patient+'_0001.nii.gz') 
        # input_pet_file = os.path.join(patient_folder,patient, 'PET.nii.gz')
        # input_ct_file = os.path.join(patient_folder,patient, 'CTres.nii.gz')
        output_dir = os.path.join('./Dataset_251/predicts',patient)
        
        try:
            demo(input_pet_file, input_ct_file, output_dir, input_model)
        except:
            print('error data')
