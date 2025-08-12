import os

import numpy as np
import pandas as pd

import glob
import argparse
import time 

from tqdm import tqdm
from val_metrics import compute_metrics


def generate_score_df(method_name:str, pred_path:str, label_path:str, test_patients:list=None, label_name="label.nii.gz", pred_name="pred.nii.gz"):
    # calculate dice, false pos vol, false neg vol for autopet data

    print(f"====Fold: {method_name}=========")

    dice_list = []
    false_pos_vol_list = []
    false_neg_vol_list = []
    name_list = []
    # print(filelist[0])
    true_pos_pred_num_list = []
    true_pos_gt_num_list = []
    false_neg_num_list = []
    false_pos_num_list = []
    pred_lesion_num_list = []
    label_lesion_num_list = []

    for patient in tqdm(test_patients):
        # print('file, ', filelist[i])
        nii_gt_path = os.path.join(label_path, patient, label_name)

        nii_pred_path = os.path.join(pred_path, patient, pred_name)

        dice_sc, false_pos_vol, false_neg_vol, addition_metrices = compute_metrics(nii_gt_path, nii_pred_path)
        # print(dice_sc, false_pos_vol, false_neg_vol, addition_metrices)
        dice_list.append(dice_sc)
        false_pos_vol_list.append(false_pos_vol)
        false_neg_vol_list.append(false_neg_vol)
        name_list.append(patient)
        true_pos_pred_num_list.append(addition_metrices['tp_pred_num'])
        true_pos_gt_num_list.append(addition_metrices['tp_gt_num'])
        false_neg_num_list.append(addition_metrices['fn_num'])
        false_pos_num_list.append(addition_metrices['fp_num'])
        pred_lesion_num_list.append(addition_metrices['pred_num'])
        label_lesion_num_list.append(addition_metrices['lesion_num'])

    df = pd.DataFrame(dict({'gt_name': name_list, 'dice_sc': dice_list, 'false_pos_vol': false_pos_vol_list,
                       'false_neg_vol': false_neg_vol_list, 
                       'true_pos_pred_num':true_pos_pred_num_list, 'true_pos_gt_num':true_pos_gt_num_list,
                       'false_neg_num':false_neg_num_list, 'false_pos_num':false_pos_num_list,
                       'pred_lesion_num':pred_lesion_num_list, 'label_lesion_num':label_lesion_num_list}))
    
    df.to_csv(f'metrics_score_testdata_{method_name}.csv')


def show_competition_metrics(metrics_score_df:pd.DataFrame, stats_tag:str):
    # dice_score = pd.read_csv(dice_score)
    mean_dice_score = metrics_score_df['dice_sc'].mean()
    mean_false_positive = metrics_score_df['false_pos_vol'].mean()
    mean_false_negative = metrics_score_df['false_neg_vol'].mean()
    print(f"***************{stats_tag}***************")
    print(f"mean dice sore:{mean_dice_score}")
    print(f"mean false negative volume:{mean_false_negative}")
    print(f"mean positive volume:{mean_false_positive}")


def show_sensitivity_and_precision(metrics_score_df:pd.DataFrame, stats_tag:str):
    total_pred_tp_lesions = metrics_score_df['true_pos_pred_num'].sum()
    total_pred_fp_lesions = metrics_score_df['false_pos_num'].sum() #误诊病灶
    total_pred_lesions =  metrics_score_df['pred_lesion_num'].sum()

    total_gt_tp_lesions = metrics_score_df['true_pos_gt_num'].sum()
    total_gt_fn_lesions = metrics_score_df['false_neg_num'].sum() #漏诊病灶
    total_gt_lesions = metrics_score_df['label_lesion_num'].sum()

    precision = total_pred_tp_lesions/total_pred_lesions
    sensitivity = total_gt_tp_lesions/total_gt_lesions

    recall = sensitivity

    # f1_socre = 2*precision*recall/(precision+recall)

    print(f"***************Model-Total Metrics: {stats_tag}***************")
    print(f"[Total Pred Precision]: {precision}")
    print(f"[Total Pred Recall(Sensitivity): {sensitivity}")

def main():
    start_time = time.time()

    parse = argparse.ArgumentParser()
    parse.add_argument("-m", "--method", type=str, required=True, help="Used method")

    args = parse.parse_args()
    lesion_detection_method = args.method

    pred_path = "/media/ifs/CRI_FL_DATA/PSMA_dataset/pred_tbr_NSY_PETMR"
    label_path = "/media/ifs/CRI_FL_DATA/PSMA_dataset/label_nii_NSY_PETMR"
    test_patients = [
        '29f', '33f', '37f', '26f', '36f', '25f', '22f_y', 
        '30f', '38', '34f', '28f', '35f', '27f', '31f', '32f'
    ]
    label_name = "label.nii.gz"
    pred_name = "pred_tbr.nii.gz"

    # generate_score_df(lesion_detection_method, pred_path, label_path, test_patients, label_name, pred_name)

    end_time = time.time()

    print("stats lesion mask time use: {:f} sec.".format(end_time-start_time))

    metrics_score_csv = f'metrics_score_testdata_{lesion_detection_method}.csv'
    metrics_score_df = pd.read_csv(metrics_score_csv)

    show_competition_metrics(metrics_score_df, lesion_detection_method)
    show_sensitivity_and_precision(metrics_score_df, lesion_detection_method)


if __name__ == "__main__":
    main()