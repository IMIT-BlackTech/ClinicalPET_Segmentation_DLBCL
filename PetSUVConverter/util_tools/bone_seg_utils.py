from skimage import morphology
import numpy as np
import os
import SimpleITK as sitk
from PetSUVConverter.InfoParser import PatientHadPetMR
from PetSUVConverter.SUVlbmCalculator import SUVCalculator

def nii2numpy(nii_path):
    # input: path of NIfTI segmentation file, output: corresponding numpy array and voxel_vol in ml
    # mask_nii = nib.load(str(nii_path))
    # mask = mask_nii.get_fdata()

    # the axis direction has been changed
    img = sitk.ReadImage(nii_path)
    data = sitk.GetArrayFromImage(img)
    return data, img.GetOrigin(), img.GetSpacing(), img.GetDirection()


def numpy2nii(data, origin, spacing, direction, savepath):
    img = sitk.GetImageFromArray(data)
    img.SetOrigin(origin)
    img.SetSpacing(spacing)
    img.SetDirection(direction)
    sitk.WriteImage(img, savepath)
    # print('SAVED: '+ savepath)


def extract_upper_bottom(mask):
    mip_2D = np.max(mask, axis=1)
    mip_1D = np.max(mip_2D, 1)
    bottom = np.unravel_index(
        np.argmax(mip_1D), mip_1D.shape
    )[0] + 1
    upper = mip_1D.shape[0] - np.unravel_index(
        np.argmax(np.flip(mip_1D)), mip_1D.shape
    )[0]

    return upper, bottom

def extract_front_back(mask):
    mip_2D = np.max(mask, axis=0)
    mip_1D = np.max(mip_2D, 1)
    front = np.unravel_index(
        np.argmax(mip_1D), mip_1D.shape
    )[0] + 1
    back = mip_1D.shape[0] - np.unravel_index(
        np.argmax(np.flip(mip_1D)), mip_1D.shape
    )[0]

    return front, back

def extract_left_right(mask):
    mip_2D = np.max(mask, axis=1)
    mip_1D = np.max(mip_2D, 0)
    left = np.unravel_index(
        np.argmax(mip_1D), mip_1D.shape
    )[0] + 1
    right = mip_1D.shape[0] - np.unravel_index(
        np.argmax(np.flip(mip_1D)), mip_1D.shape
    )[0]

    return left, right

def get_seg_lymph(pet_image_suv, pet_bone_seg, pet_lung_seg, pet_liver_seg):
    # init pet lesion mask
    # dilation and erosion pet_liver_mask
    pet_liver_mask = pet_liver_seg.copy()
    for i in range(2):
        pet_liver_mask = morphology.binary_dilation(pet_liver_mask, morphology.ball(1))
    # pet_liver_mask = morphology.binary_erosion(pet_liver_mask, morphology.ball(2))

    pet_bone_mask = pet_bone_seg.copy()
    pet_bone_mask = morphology.binary_erosion(pet_bone_mask, morphology.ball(1))
    pet_bone_mask = morphology.binary_dilation(pet_bone_mask, morphology.ball(1))

    # bladder
    # pet_bladder_mask = pet_bladder_seg.copy()
    # pet_bladder_mask_fine = np.zeros(pet_image_suv.shape)
    # pet_bladder_mask_fine = morphology.binary_dilation(pet_bladder_mask, morphology.ball(1))
    # pet_bladder_mask_fine = pet_bladder_mask_fine + \
    #                         np.roll(pet_bladder_mask_fine, -1, axis=1) + np.roll(pet_bladder_mask_fine, -2, axis=1) + \
    #                         np.roll(pet_bladder_mask_fine, 1, axis=1) + \
    #                         np.roll(pet_bladder_mask_fine, -1, axis=2) + np.roll(pet_bladder_mask_fine, -2, axis=2) + \
    #                         np.roll(pet_bladder_mask_fine, 1, axis=2) + np.roll(pet_bladder_mask_fine, 2, axis=2) + \
    #                         np.roll(pet_bladder_mask_fine, 1, axis=0) + \
    #                         np.roll(pet_bladder_mask_fine, -1, axis=0) + np.roll(pet_bladder_mask_fine, -2, axis=0) + \
    #                         np.roll(pet_bladder_mask_fine, -3, axis=0) + np.roll(pet_bladder_mask_fine, -4, axis=0)

    # pet_bladder_mask_fine = pet_bladder_mask_fine.astype('int8')

    # save_nii('pet_bladder_mask_fine', pet_bladder_mask_fine)

    # lung
    pet_lung_mask = pet_lung_seg.copy()

    # bone basic
    upperLung, bottomLung = extract_upper_bottom(pet_lung_mask)
    frontLung, backLung = extract_front_back(pet_lung_mask)
    leftLung, rightLung = extract_left_right(pet_lung_mask)

    upperLiver, bottomLiver = extract_upper_bottom(pet_liver_mask)
    frontLiver, backLiver = extract_front_back(pet_liver_mask)
    leftLiver, rightLiver = extract_left_right(pet_liver_mask)

    # upperBladder, bottomBladder = extract_upper_bottom(pet_bladder_mask_fine)
    # frontBladder, backBladder = extract_front_back(pet_bladder_mask_fine)
    # leftBladder, rightBladder = extract_left_right(pet_bladder_mask_fine)

    # =================================================================================
    # bone_fine
    pet_bone_mask_fine = np.zeros(pet_image_suv.shape)

    # 颅底
    bottomHead = upperLung + 10
    # 修头骨
    pet_bone_mask_head = morphology.binary_erosion(pet_bone_mask, morphology.ball(1))
    pet_bone_mask_fine[bottomHead:pet_bone_mask.shape[0], :, :] = pet_bone_mask_head[bottomHead:pet_bone_mask.shape[0],
                                                                  :, :]

    # 修颈椎
    pet_spine_mask = np.zeros(pet_bone_mask.shape)
    pet_spine_mask[bottomHead:bottomHead + 6, :, :] = pet_bone_mask[bottomHead:bottomHead + 6, :, :]
    upperSpine, bottomSpine = extract_upper_bottom(pet_spine_mask)
    frontSpine, backSpine = extract_front_back(pet_spine_mask)
    leftSpine, rightSpine = extract_left_right(pet_spine_mask)

    ############# pet_bone_mask_fine is a dilation of bone, then find the lymph area ##########################
    # add a block on the back of the neck
    pet_bone_mask_fine[bottomSpine:bottomSpine + 25, backSpine - 10:backSpine + 10, leftSpine - 3:rightSpine + 3] = 1
    # save_nii('debug_bone_mask_Spine_backfine', pet_bone_mask_fine)

    # add a block in front of the neck
    pet_bone_mask_fine[bottomSpine:bottomSpine + 25, frontSpine - 10:frontSpine + 10, leftSpine - 3:rightSpine + 3] = 1
    # save_nii('debug_bone_mask_Spine_frontfine', pet_bone_mask_fine)

    # 修脊柱
    pet_bone_mask_spine = pet_bone_mask
    for i in range(3):
        pet_bone_mask_spine = morphology.binary_dilation(pet_bone_mask_spine, morphology.ball(1))

    pet_bone_mask_spine = pet_bone_mask_spine + np.roll(pet_bone_mask_spine, -1, axis=1)

    # extract spine fine front, back, left back
    frontSpineFine, backSpineFine = extract_front_back(pet_bone_mask_spine)
    leftSpineFine, rightSpineFine = extract_left_right(pet_bone_mask_spine)
    # print('frontSpineFine, backSpineFine, leftSpineFine, rightSpineFine', frontSpineFine, backSpineFine, leftSpineFine,
    #       rightSpineFine)

    # save_nii('debug_bone_mask_spine', pet_bone_mask_spine)

    # 修盆腔
    pet_bone_mask_spine_pelvic = pet_bone_mask
    for i in range(5):
        pet_bone_mask_spine_pelvic = morphology.binary_dilation(pet_bone_mask_spine_pelvic, morphology.ball(1))

    pet_bone_mask_spine_pelvic = pet_bone_mask_spine_pelvic + np.roll(pet_bone_mask_spine_pelvic, -1, axis=1) \
                                 + np.roll(pet_bone_mask_spine_pelvic, -2, axis=1) + np.roll(pet_bone_mask_spine_pelvic,
                                                                                             -3, axis=1) \
                                 + np.roll(pet_bone_mask_spine_pelvic, -4, axis=1)

    # 顺序不能变==============================
    # 脊柱加粗
    # pet_bone_mask_fine[upperBladder + 30:bottomHead, frontLung: pet_bone_mask_spine.shape[1],
    # leftLung + 20:rightLung - 20] = \
    #     pet_bone_mask_spine[upperBladder + 30:bottomHead, frontLung: pet_bone_mask_spine.shape[1],
    #     leftLung + 20:rightLung - 20]
    pet_bone_mask_fine[5:bottomHead, frontLung: pet_bone_mask_spine.shape[1],
    leftLung + 20:rightLung - 20] = \
        pet_bone_mask_spine[5:bottomHead, frontLung: pet_bone_mask_spine.shape[1],
        leftLung + 20:rightLung - 20]

    # 脊柱加粗2
    pet_bone_mask_fine[upperLiver - 55:upperLiver, :, :] = pet_bone_mask_spine[upperLiver - 55:upperLiver, :, :]
    # save_nii('debug_bone_mask_spine2_fine', pet_bone_mask_fine)
    # 下肢加粗
    # pet_bone_mask_fine[5:upperBladder + 30, frontLung: pet_bone_mask_spine.shape[1], leftLung - 2:rightLung + 2] = \
    #     pet_bone_mask_spine[5:upperBladder + 30, frontLung: pet_bone_mask_spine.shape[1], leftLung - 2:rightLung + 2]

    pet_bone_mask_fine[5:bottomHead, frontLung: pet_bone_mask_spine.shape[1], leftLung - 2:rightLung + 2] = \
        pet_bone_mask_spine[5:bottomHead, frontLung: pet_bone_mask_spine.shape[1], leftLung - 2:rightLung + 2]

    # save_nii('debug_bone_mask_legs_fine', pet_bone_mask_fine)

    # 脊柱前食道消除
    pet_bone_mask_fine[bottomHead - 14: bottomHead + 25, bottomSpine - 35: bottomSpine - 14, leftSpine: rightSpine] = 0

    # 图像底部消除 done
    pet_bone_mask_fine[1:5, :, :] = 0

    # 去肝部骨骼
    pet_bone_mask_fine = pet_bone_mask_fine * (1 - pet_liver_mask)
    # save_nii('pet_bone_mask_fine', pet_bone_mask_fine)

    # ============ bone_fine dilation to expand lymph area===============================================
    pet_bone_mask_fine = morphology.dilation(pet_bone_mask_fine, morphology.ball(4))
    # save_nii('debug_bone_mask_fine_dilation', pet_bone_mask_fine)

    # 顺序不能变==============================
    # 膀胱上面骨
    # pet_bone_mask_fine[bottomBladder:bottomBladder + 10, :, :] = pet_bone_mask_spine_pelvic[
    #                                                              bottomBladder:bottomBladder + 10, :, :]
    # # 膀胱前列腺区域
    # centerBladder = np.array([bottomBladder + upperBladder, frontBladder + backBladder, leftBladder + rightBladder])
    # centerBladder = np.round(centerBladder / 2)
    # # add a block around bladder area
    # pet_bone_mask_fine[bottomBladder - 24: (upperBladder - round(1 / 4 * (upperBladder - bottomBladder))), \
    # int(centerBladder[1]) - 24: int(centerBladder[1]) + 24, int(centerBladder[2]) - 24: int(centerBladder[2]) + 24] = 1

    # ============then add block to cover upper shoulder and ribs lymph (above bottomLiver)=============
    pet_bone_mask_fine[bottomLiver: bottomHead + 10, frontSpineFine - 5:backSpineFine,
    leftSpineFine - 5:rightSpineFine + 5] = 1
    # print('frontSpine, backSpine, leftSpine, rightSpine', frontSpine, backSpine, leftSpine, rightSpine)
    # save_nii('debug_bone_mask_shoulder_fine', pet_bone_mask_fine)

    # bone_chest
    pet_bone_mask_chest = np.zeros(pet_image_suv.shape)
    pet_bone_mask_chest[(upperLiver - round((upperLiver - bottomLiver) / 2)):upperLung, :, :] = \
        pet_bone_mask[(upperLiver - round((upperLiver - bottomLiver) / 2)):upperLung,:, :]
    pet_bone_mask_chest[:, frontLung - 20:frontLung, leftLung + 5:rightLung - 5] = 0
    pet_bone_mask_chest = pet_bone_mask_chest * (1 - pet_liver_mask)
    pet_bone_mask_chest = morphology.erosion(pet_bone_mask_chest, morphology.ball(1))
    # save_nii('pet_bone_mask_chest', np2sitk(pet_bone_mask_chest.astype(np.int16)))

    ####jz
    # print('debug: bone')
    # save_nii('debug_bone_mask_fine', pet_bone_mask_fine)
    # save_nii('debug_1_bladder_seg', 1-pet_bladder_seg)

    # ##### remove bladder part
    # pet_bone_mask_fine[pet_bladder_seg > 0] = 0

    """
        get lymph = bone_mask_fine - bone mask
        input already dilated bone mask (did not exclude organs yet)
        no heart seg, lesion normally does not grow on the heart
        """
    # print('bone_mask_fine, pet_bone_seg', pet_bone_mask_fine.shape, pet_bone_seg.shape)
    pet_lymph_mask_raw = pet_bone_mask_fine.copy()

    # pet_lymph_mask_raw = morphology.dilation(pet_lymph_mask_raw, morphology.ball(5))

    pet_lymph_mask_raw[pet_bone_seg == 1] = 0

    pet_lymph_mask_raw[pet_lung_mask == 1] = 0

    pet_lymph_mask_raw[pet_lung_mask == 1] = 0

    pet_lymph_mask_raw[pet_liver_mask == 1] = 0

    # print('pet_lymph_mask_raw', pet_lymph_mask_raw.shape)

    return pet_lymph_mask_raw



def segment_bone(ct):
    bone_seg = ct.copy()

    # normal ct
    int_LowThrLimit = 90
    int_HighThrLimit = 0

    # #### contrast enhanced CT
    # int_LowThrLimit = 200
    # int_HighThrLimit = 0

    bone_seg[bone_seg <= int_LowThrLimit] = 0
    bone_seg[bone_seg > int_HighThrLimit] = 1
    bone_seg = bone_seg.astype(np.float32)

    bone_seg = morphology.binary_erosion(bone_seg, selem=morphology.ball(1))
    bone_seg = morphology.binary_dilation(bone_seg, selem=morphology.ball(1))
    bone_seg = bone_seg.astype(np.float32)

    # pet_bone_seg = mr2pet_resample(mr_bone_seg, pad_size, pet_size)
    # save_nii('debug_petsegmentbone', pet_bone_seg, './')
    return bone_seg

def ct_bed_removal_bone(ct_file, suv_file, organ_seg):

    # cut 0.05, 0.95  margin
    # savefolder = '/dssg/home/acct-bmezzy/bmezzy-1/autopet_organ'

    # ct_file = os.path.join(self.input_path, 'ctres.nii.gz')
    # suv_file = os.path.join(self.input_path, 'pet.nii.gz')
    
    ct, origin, spacing, direction = nii2numpy(ct_file)
    # print(origin, spacing, direction)
    suv, _, _, _ = nii2numpy(suv_file)
    # print('suv max, min', max(np.ravel(suv)), min(np.ravel(suv)))
    # print('ct max, min', max(np.ravel(ct)), min(np.ravel(ct)))
    # print('pet suv shape, ', suv.shape)

    # find pet non zero mask
    # mask[suv<=0.05] = 0

    # ct_wo_bed = ct*mask

    ct_wo_bed = ct
    ct_wo_bed[suv<=0.05] = -1024

    # numpy2nii(ct_wo_bed, origin, spacing, direction, os.path.join(self.result_path, self.uuid+'CT_wo_bed.nii.gz'))
    whole_bone = segment_bone(ct_wo_bed)
    # numpy2nii(whole_bone, origin, spacing, direction, os.path.join(self.result_path, self.uuid+'bone.nii.gz'))

    ### get lymph area
    #### lung:1, kidney:2, liver:3, heart:4, rib:5,
    # organ_file = os.path.join(self.result_path, self.uuid+'.nii.gz')
    # organ_seg, _, _, _ = nii2numpy(organ_file)
    # print('organseg shape, ', organ_seg.shape)
    lung_seg = np.zeros(organ_seg.shape)
    liver_seg = np.zeros(organ_seg.shape)
    lung_seg[organ_seg==1] = 1
    liver_seg[organ_seg==3] = 1

    lymph_mask_raw = get_seg_lymph(suv, whole_bone, lung_seg, liver_seg)
    # numpy2nii(lymph_mask_raw, origin, spacing, direction, os.path.join(self.result_path, self.uuid+'lymph.nii.gz'))

    return whole_bone, lymph_mask_raw



def get_pet_suv_img(pet_internsity_image, corresponding_patient_parser):
    patient_suv_calculator = SUVCalculator(corresponding_patient_parser)
    pet_image_suv = patient_suv_calculator.calculate_suv(
        pet_internsity_image)

    return pet_image_suv


