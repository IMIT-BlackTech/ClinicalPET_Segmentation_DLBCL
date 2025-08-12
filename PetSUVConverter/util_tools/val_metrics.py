import numpy as np
import nibabel as nib
import pathlib as plb
import cc3d
import csv
import sys


def nii2numpy(nii_path):
    # input: path of NIfTI segmentation file, output: corresponding numpy array and voxel_vol in ml
    mask_nii = nib.load(str(nii_path))
    mask = mask_nii.get_fdata()
    pixdim = mask_nii.header['pixdim']   
    voxel_vol = pixdim[1]*pixdim[2]*pixdim[3]/1000
    return mask, voxel_vol


def con_comp(seg_array):
    # input: a binary segmentation array output: an array with seperated (indexed) connected components of the segmentation array
    connectivity = 18
    conn_comp = cc3d.connected_components(seg_array, connectivity=connectivity)
    return conn_comp


def false_pos_pix(gt_array,pred_array):
    # compute number of voxels of false positive connected components in prediction mask
    pred_conn_comp = con_comp(pred_array)
    
    false_pos = 0
    false_pos_num = 0
    true_pos_num_pred = 0

    pred_lesion_num = pred_conn_comp.max()

    for idx in range(1,pred_conn_comp.max()+1):
        comp_mask = np.isin(pred_conn_comp, idx)
        if (comp_mask*gt_array).sum() == 0:
            false_pos = false_pos+comp_mask.sum()
            false_pos_num = false_pos_num+1
        else :
            true_pos_num_pred = true_pos_num_pred+1

    return false_pos, false_pos_num, true_pos_num_pred, pred_lesion_num


def false_neg_pix(gt_array,pred_array):
    # compute number of voxels of false negative connected components (of the ground truth mask) in the prediction mask
    gt_conn_comp = con_comp(gt_array)
    
    false_neg = 0
    false_neg_num = 0
    true_pos_num_gt = 0

    pos_lesion_num = gt_conn_comp.max()

    for idx in range(1,gt_conn_comp.max()+1):
        comp_mask = np.isin(gt_conn_comp, idx)
        if (comp_mask*pred_array).sum() == 0:
            false_neg = false_neg+comp_mask.sum()
            false_neg_num = false_neg_num+1
        else:
            true_pos_num_gt = true_pos_num_gt+1
            
    return false_neg, false_neg_num, true_pos_num_gt, pos_lesion_num


def get_false_pos_num(gt_array,pred_array):
    # compute number of voxels of false positive connected components in prediction mask
    pred_conn_comp = con_comp(pred_array)
    
    false_pos_num = 0
    for idx in range(1,pred_conn_comp.max()+1):
        comp_mask = np.isin(pred_conn_comp, idx)
        if (comp_mask*gt_array).sum() == 0:
            false_pos_num = false_pos_num+1
    return false_pos_num


def get_false_neg_num(gt_array,pred_array):
    # compute number of voxels of false negative connected components (of the ground truth mask) in the prediction mask
    gt_conn_comp = con_comp(gt_array)
    
    false_neg_num = 0
    for idx in range(1,gt_conn_comp.max()+1):
        comp_mask = np.isin(gt_conn_comp, idx)
        if (comp_mask*pred_array).sum() == 0:
            false_neg_num = false_neg_num+1
            
    return false_neg_num


def get_true_pos_num(gt_array,pred_array):
    # compute number of voxels of false negative connected components (of the ground truth mask) in the prediction mask
    gt_conn_comp = con_comp(gt_array)
    pred_conn_comp = con_comp(pred_array)

    pred_lesion_num =  pred_conn_comp.max()
    pos_lesion_num = gt_conn_comp.max()

    true_pos_num = 0
    for idx in range(1,gt_conn_comp.max()+1):
        comp_mask = np.isin(gt_conn_comp, idx)
        if (comp_mask*pred_array).sum() != 0:
            true_pos_num = true_pos_num+1
            
    return true_pos_num, pred_lesion_num, pos_lesion_num


def dice_score_arc(mask1,mask2):
    # compute foreground Dice coefficient
    overlap = (mask1*mask2).sum()
    sum = mask1.sum()+mask2.sum()
    dice_score = 2*overlap/sum
    return dice_score

def dice_score(mask1,mask2):
    # compute foreground Dice coefficient

    # if case negative, set dice_score=1
    if np.sum(np.ravel(mask1))==0:
        dice_score=1
    else:
        overlap = (mask1*mask2).sum()
        sum = mask1.sum()+mask2.sum()
        dice_score = 2*overlap/sum
    return dice_score


# def compute_metrics(nii_gt_path, nii_pred_path):
#     # main function
#     gt_array, voxel_vol = nii2numpy(nii_gt_path)
#     pred_array, voxel_vol = nii2numpy(nii_pred_path)

#     false_neg_vol = false_neg_pix(gt_array, pred_array)*voxel_vol
#     false_pos_vol = false_pos_pix(gt_array, pred_array)*voxel_vol
#     dice_sc = dice_score(gt_array,pred_array)

#     if np.sum(np.ravel(gt_array))==0 and false_pos_vol!=0:
#         dice_sc = 0

#     return dice_sc, false_pos_vol, false_neg_vol


def compute_metrics(nii_gt_path, nii_pred_path):
    # main function
    gt_array, voxel_vol = nii2numpy(nii_gt_path)
    pred_array, voxel_vol = nii2numpy(nii_pred_path)

    # false_neg_vol = false_neg_pix(gt_array, pred_array)*voxel_vol
    # false_pos_vol = false_pos_pix(gt_array, pred_array)*voxel_vol

    false_neg, false_neg_num, true_pos_num_gt,pos_lesion_num = false_neg_pix(gt_array, pred_array) # loop in ground truth
    false_pos, false_pos_num, true_pos_num_pred, pred_lesion_num = false_pos_pix(gt_array, pred_array) # loop in pred

    false_neg_vol = false_neg*voxel_vol
    false_pos_vol = false_pos*voxel_vol

    dice_sc = dice_score(gt_array,pred_array)

    addition_metrices = {
        'tp_pred_num':true_pos_num_pred, 'tp_gt_num':true_pos_num_gt,
        'fn_num':false_neg_num, 'fp_num':false_pos_num, 
        'pred_num':pred_lesion_num, 'lesion_num':pos_lesion_num
        }
    # print(addition_metrices)

    if np.sum(np.ravel(gt_array))==0 and false_pos_vol!=0:
        dice_sc = 0

    return dice_sc, false_pos_vol, false_neg_vol, addition_metrices


def compute_additional_metrics(nii_gt_path, nii_pred_path):
    # main function
    gt_array, voxel_vol = nii2numpy(nii_gt_path)
    pred_array, voxel_vol = nii2numpy(nii_pred_path)

    false_neg_vol = false_neg_pix(gt_array, pred_array)*voxel_vol
    false_pos_vol = false_pos_pix(gt_array, pred_array)*voxel_vol

    dice_sc = dice_score(gt_array,pred_array)

    true_pos_num, pred_lesion_num, pos_lesion_num = get_true_pos_num(gt_array, pred_array)
    false_pos_num = get_false_pos_num(gt_array, pred_array)
    false_neg_num = get_false_neg_num(gt_array, pred_array)

    if np.sum(np.ravel(gt_array))==0 and false_pos_vol!=0:
        dice_sc = 0

    return dice_sc, false_pos_vol, false_neg_vol


if __name__ == "__main__":

    nii_gt_path, nii_pred_path = sys.argv

    nii_gt_path = plb.Path(nii_gt_path)
    nii_pred_path = plb.Path(nii_pred_path)
    dice_sc, false_pos_vol, false_neg_vol = compute_metrics(nii_gt_path, nii_pred_path)

    csv_header = ['gt_name', 'dice_sc', 'false_pos_vol', 'false_neg_vol']
    csv_rows = [nii_gt_path.name,dice_sc, false_pos_vol, false_neg_vol]

    with open("metrics.csv", "w", newline='') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(csv_header) 
        writer.writerows(csv_rows)