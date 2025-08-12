import os
import sys
import glob
import shutil
import SimpleITK as sitk

from PetSUVConverter.InfoParser import PatientInfoReader

def copy_labels():
    label_nsy_root = "/media/ifs/CRI_FL_DATA/PSMA_dataset/label_nii_NSY_PETMR/"
    petmr_nsy_data_root = "/media/ifs/CRI_FL_DATA/PSMA_dataset/NSY_PETMR"

    patients = os.listdir(petmr_nsy_data_root)
    patient_labels = glob.glob(os.path.join(label_nsy_root, '*.nii.gz'))

    for patient in patients:
        label_dir = os.path.join(label_nsy_root, patient)
        if not os.path.exists(label_dir):
            os.makedirs(label_dir)
        
        label_file = os.path.join(label_nsy_root, f"p{patient}.nii.gz")
        target_file = os.path.join(label_dir, 'label.nii.gz')
        # print(label_file)
        if label_file in patient_labels:
            print(label_file)
            shutil.copy(label_file, target_file)

def ext_spacing_from_dcm(dcm_folder):
    dcm_reader = PatientInfoReader(dcm_folder)
    values = dcm_reader.read(['SliceThickness', 'PixelSpacing'])
    slice_thickness = float(values["SliceThickness"])
    row_spacing, col_spacing = list(map(lambda x: float(x), values["PixelSpacing"]))
    spacing = [row_spacing, col_spacing, slice_thickness]
    
    # print(spacing)
    return spacing


def check_nifiti_spacing(change_spacing = False, file_name='mr2ct_sample.nii.gz'):
    petmr_nsy_data_root = "/media/ifs/CRI_FL_DATA/PSMA_dataset/NSY_PETMR"
    mr2ct_file = file_name

    patients = os.listdir(petmr_nsy_data_root)

    for patient in patients:
        ct_file = os.path.join(petmr_nsy_data_root, patient, mr2ct_file)
        ct_itk = sitk.ReadImage(ct_file)

        mr_folder = os.path.join(petmr_nsy_data_root, patient, 'mr')
        print(f"======={patient}=======")
        print('nii itk spacing:', ct_itk.GetSpacing())

        spacing = ext_spacing_from_dcm(mr_folder)
        if change_spacing:
            ct_itk.SetSpacing(spacing)
            out_path = os.path.join(petmr_nsy_data_root, patient, 'ct.nii.gz')
            sitk.WriteImage(ct_itk, out_path)
        print('dcm spacing:', spacing)


def main():
    check_nifiti_spacing(False, 'ct.nii.gz')

if __name__=="__main__":
    main()