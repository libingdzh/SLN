import torch
import pickle
from Nii_utils import NiiDataRead, NiiDataRead_2D
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from monai.transforms import Compose, Rand2DElasticd, Rand3DElasticd, RandRotated, RandGaussianNoised, Rand3DElastic, RandFlipd, ToTensor
import os
from torch.utils.data import Dataset, DataLoader
from skimage import transform
from utils import masked_Zscore_norm
import glob
from monai.utils import set_determinism

set_determinism(seed=9)


mean_std_dict = {'Age': [48.68514851, 10.91070313],
                 'Tumor size US length': [2.787544554,1.68241208], 'Tumor size US width': [2.213643564, 1.29107745],
                 'Tumor size MG': [2.706930693, 1.602328592], 'Tumor size MRI': [2.644158416, 1.542471507]}
#u
def Extract_clinical_features(metadata_df, ID):
    # print(ID)
    Clinical_features = []
    affected_side = str(metadata_df.loc[metadata_df['ID'] == ID, 'Laterality'].values[0])
    if 'Left' in affected_side and 'Right' not in affected_side:
        Clinical_features.append(torch.tensor([1, 0]))
    elif 'Right' in affected_side and 'Left' not in affected_side:
        Clinical_features.append(torch.tensor([0, 1]))
    else:
        Clinical_features.append(torch.tensor([0, 0]))

    num_tumor = str(metadata_df.loc[metadata_df['ID'] == ID, 'Multifocality'].values[0])
    if 'Unifocal' in num_tumor and 'Multifocal' not in num_tumor and 'Multicentric' not in num_tumor:
        Clinical_features.append(torch.tensor([1, 0, 0]))
    elif 'Multifocal' in num_tumor and 'Multicentric' not in num_tumor and 'Unifocal' not in num_tumor:
        Clinical_features.append(torch.tensor([0, 1, 0]))
    elif 'Multicentric' in num_tumor and 'Multifocal' not in num_tumor and 'Unifocal' not in num_tumor:
        Clinical_features.append(torch.tensor([0, 0, 1]))
    else:
        Clinical_features.append(torch.tensor([0, 0, 0]))

    Preoperative_mass_position = str(metadata_df.loc[metadata_df['ID'] == ID, 'Location across nine quadrants'].values[0])
    P_mass_p_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    if 'UQ' in Preoperative_mass_position:
        P_mass_p_list[0] = 1
    if 'OQ' in Preoperative_mass_position:
        P_mass_p_list[1] = 1
    if 'LQ' in Preoperative_mass_position:
        P_mass_p_list[2] = 1
    if 'IQ' in Preoperative_mass_position:
        P_mass_p_list[3] = 1
    if 'UOQ' in Preoperative_mass_position:
        P_mass_p_list[4] = 1
    if 'LOQ' in Preoperative_mass_position:
        P_mass_p_list[5] = 1
    if 'LIQ' in Preoperative_mass_position:
        P_mass_p_list[6] = 1
    if 'UIQ' in Preoperative_mass_position:
        P_mass_p_list[7] = 1
    if 'central portion' in Preoperative_mass_position:
        P_mass_p_list[8] = 1
    Clinical_features.append(torch.tensor(P_mass_p_list))

    cT_stage = str(metadata_df.loc[metadata_df['ID'] == ID, 'cT'].values[0])
    if 'T1' in cT_stage:
        Clinical_features.append(torch.tensor([0, 1, 0, 0, 0]))
    elif 'T2' in cT_stage:
        Clinical_features.append(torch.tensor([0, 0, 1, 0, 0]))
    elif 'T3' in cT_stage:
        Clinical_features.append(torch.tensor([0, 0, 0, 1, 0]))
    elif 'T4' in cT_stage:
        Clinical_features.append(torch.tensor([0, 0, 0, 0, 1]))
    else:
        Clinical_features.append(torch.tensor([0, 0, 0, 0, 0]))

    N_stage = str(metadata_df.loc[metadata_df['ID'] == ID, 'Axillary US'].values[0])
    if 'Negative' in N_stage:
        Clinical_features.append(torch.tensor([1, 0, 0, 0]))
    elif 'Suspicious' in N_stage:
        Clinical_features.append(torch.tensor([0, 1, 0, 0]))
    else:
        Clinical_features.append(torch.tensor([0, 0, 0, 0]))

    calcify_MG = str(metadata_df.loc[metadata_df['ID'] == ID, 'Calcification'].values[0])
    if 'Yes' in calcify_MG:
        Clinical_features.append(torch.tensor([1]))
    elif 'No' in calcify_MG:
        Clinical_features.append(torch.tensor([-1]))
    else:
        Clinical_features.append(torch.tensor([0]))

    density_MG = str(metadata_df.loc[metadata_df['ID'] == ID, 'Breast density'].values[0])
    if 'A' in density_MG:
        Clinical_features.append(torch.tensor([1, 0, 0, 0]))
    elif 'B' in density_MG:
        Clinical_features.append(torch.tensor([0, 1, 0, 0]))
    elif 'C' in density_MG:
        Clinical_features.append(torch.tensor([0, 0, 1, 0]))
    elif 'D' in density_MG:
        Clinical_features.append(torch.tensor([0, 0, 0, 1]))
    else:
        Clinical_features.append(torch.tensor([0, 0, 0, 0]))

    for feature_name in ['Age', 'Tumor size US length', 'Tumor size US width', 'Tumor size MG',
                         'Tumor size MRI']:
        feature_value = float(metadata_df.loc[metadata_df['ID'] == ID, feature_name].values[0])
        if np.isnan(feature_value):
            Clinical_features.append(torch.tensor([mean_std_dict[feature_name][0]]))
        else:
            feature_value = (feature_value - mean_std_dict[feature_name][0]) / mean_std_dict[feature_name][1]
            Clinical_features.append(torch.tensor([feature_value]))
    return torch.cat(Clinical_features)
# u
def Extract_pathology_features(metadata_df, ID):
    Clinical_features = []

    pathological_types = str(metadata_df.loc[metadata_df['ID'] == ID, 'Histological type'].values[0])
    if 'Invasive ductal carcinoma' in pathological_types:
        Clinical_features.append(torch.tensor([1, 0, 0, 0, 0, 0, 0, 0]))
    elif 'Invasive lobular carcinoma' in pathological_types:
        Clinical_features.append(torch.tensor([0, 1, 0, 0, 0, 0, 0, 0]))
    elif 'Invasive carcinoma' in pathological_types:
        Clinical_features.append(torch.tensor([0, 0, 0, 1, 0, 0, 0, 0]))
    elif 'Invasive micropapillary carcinoma' in pathological_types:
        Clinical_features.append(torch.tensor([0, 0, 0, 0, 0, 1, 0, 0]))
    elif 'Ductal carcinoma in situ' in pathological_types:
        Clinical_features.append(torch.tensor([0, 0, 0, 0, 0, 0, 1, 0]))
    else:
        Clinical_features.append(torch.tensor([0, 0, 0, 0, 0, 0, 0, 0]))

    histological_grading = str(metadata_df.loc[metadata_df['ID'] == ID, 'Histological grade'].values[0])
    if 'I' in histological_grading and 'II' not in histological_grading and 'III' not in histological_grading:
        Clinical_features.append(torch.tensor([1, 0, 0]))
    elif 'II' in histological_grading and 'III' not in histological_grading and 'I' not in histological_grading:
        Clinical_features.append(torch.tensor([0, 1, 0]))
    elif 'III' in histological_grading and 'I' not in histological_grading and 'II' not in histological_grading:
        Clinical_features.append(torch.tensor([0, 0, 1]))
    else:
        Clinical_features.append(torch.tensor([0, 0, 0]))

    ER = str(metadata_df.loc[metadata_df['ID'] == ID, 'ER'].values[0])
    if 'Negative' in ER:
        Clinical_features.append(torch.tensor([1, 0]))
    elif 'Positive' in ER:
        Clinical_features.append(torch.tensor([0, 1]))
    else:
        Clinical_features.append(torch.tensor([0, 0]))
    
    PR = str(metadata_df.loc[metadata_df['ID'] == ID, 'PR'].values[0])
    if 'Negative' in PR:
        Clinical_features.append(torch.tensor([1, 0]))
    elif 'Positive' in PR:
        Clinical_features.append(torch.tensor([0, 1]))
    else:
        Clinical_features.append(torch.tensor([0, 0]))
    
    HER2 = str(metadata_df.loc[metadata_df['ID'] == ID, 'HER2'].values[0])
    if 'Negative' in HER2:
        Clinical_features.append(torch.tensor([1, 0]))
    elif 'Positive' in HER2:
        Clinical_features.append(torch.tensor([0, 1]))
    else:
        Clinical_features.append(torch.tensor([0, 0]))

    LVIQF = str(metadata_df.loc[metadata_df['ID'] == ID, 'LVI'].values[0])
    if 'Invisible' in LVIQF:
        Clinical_features.append(torch.tensor([1, 0]))
    elif 'Visible' in LVIQF:
        Clinical_features.append(torch.tensor([0, 1]))
    else:
        Clinical_features.append(torch.tensor([0, 0]))

    for feature_name in ['Ki67']:
        feature_value = str(metadata_df.loc[metadata_df['ID'] == ID, feature_name].values[0])
        if '<20%' in feature_value or '0' in feature_value:
            Clinical_features.append(torch.tensor([float(0)]))
        elif '>=20%' in feature_value or '1' in feature_value:
            Clinical_features.append(torch.tensor([float(1)]))
        else:
            Clinical_features.append(torch.tensor([float(0.5)]))
    return torch.cat(Clinical_features)

# u
class Dataset_imgonlyMR(Dataset):
    def __init__(self, data_dir, split_path,data_set='train', augment=True):
        self.data_dir = data_dir
        split_data = pickle.load(open(split_path, 'rb'))
        self.ID_list = split_data[data_set][0]
        self.label_list = split_data[data_set][1]
        self.augment = augment

        if augment:
            self.transforms_2D = Compose([
                Rand2DElasticd(keys=['img', 'mask'], spacing=(20, 20), magnitude_range=(1, 2), mode="nearest",
                               padding_mode="zeros", prob=0.2),
                RandGaussianNoised(keys=['img'], mean=0.0, std=0.1, prob=0.1),
                RandRotated(keys=['img', 'mask'], range_x=np.pi / 360 * 30, prob=0.3, keep_size=True,
                           mode="nearest", padding_mode="zeros"),
            ])
            self.transforms_3D = Compose([
                Rand3DElasticd(keys=['img', 'mask'], sigma_range=(5, 7), magnitude_range=(10, 20), mode="nearest",
                               padding_mode="zeros", prob=0.2),
                RandGaussianNoised(keys=['img'], mean=0.0, std=0.1, prob=0.1),
                RandRotated(keys=['img', 'mask'], range_z=np.pi / 360 * 30, prob=0.3, keep_size=True,
                            mode="nearest", padding_mode="zeros"),
            ])
        self.len = len(self.ID_list)

    def __getitem__(self, idx):
        ID = self.ID_list[idx]
        label = self.label_list[idx]
        print(ID)
        # MR DCE
        # print('MR DCE')
        MR_DCE, _, _, _ = NiiDataRead(glob.glob(
            os.path.join(self.data_dir, 'MR+MG', label, ID, '{}_DCE_N4.nii*'.format(ID)))[0])
        if MR_DCE.max() < 0:
            MR_DCE = masked_Zscore_norm(MR_DCE, mask_min=np.min(MR_DCE), mask_max=np.max(MR_DCE), percentile_min=None,
                                        percentile_max=99.5)
        else:
            MR_DCE = masked_Zscore_norm(MR_DCE, mask_min=0, mask_max=np.max(MR_DCE), percentile_min=None,
                                        percentile_max=99.5)
        MR_DCE_GTV, _, _, _ = NiiDataRead(glob.glob(
            os.path.join(self.data_dir, 'MR+MG', label, ID, '{}_DCE_GTV.nii*'.format(ID)))[0])
        MR_DCE_GTV[MR_DCE_GTV <= 0.5] = 0
        MR_DCE_GTV[MR_DCE_GTV > 0.5] = 1
        MR_DCE_GTVnd1, _, _, _ = NiiDataRead(glob.glob(
            os.path.join(self.data_dir, 'MR+MG', label, ID, '{}_DCE_GTVnd1.nii*'.format(ID)))[0])
        MR_DCE_GTVnd1[MR_DCE_GTVnd1 <= 0.5] = 0
        MR_DCE_GTVnd1[MR_DCE_GTVnd1 > 0.5] = 1
        img_MR_DCE = MR_DCE[np.newaxis, ...]
        mask_MR_DCE = np.concatenate((MR_DCE_GTV[np.newaxis, ...], MR_DCE_GTVnd1[np.newaxis, ...]), axis=0)
        img_MR_DCE = transform.resize(img_MR_DCE, (1, 144, 320, 320), order=0, mode='constant', clip=False,
                                          preserve_range=True, anti_aliasing=False)
        mask_MR_DCE = transform.resize(mask_MR_DCE, (2, 144, 320, 320), order=0, mode='constant', clip=False,
                                           preserve_range=True, anti_aliasing=False)
        # print(img_MR_DCE.shape, mask_MR_DCE.shape)

        if self.augment:

            augmented_MR_DCE = self.transforms_3D({'img': img_MR_DCE, 'mask': mask_MR_DCE})
            img_MR_DCE = np.concatenate((augmented_MR_DCE['img'], augmented_MR_DCE['mask']), axis=0)
        else:
            img_MR_DCE = np.concatenate((img_MR_DCE, mask_MR_DCE), axis=0)

        img_MR_DCE = torch.from_numpy(img_MR_DCE)
        if label == 'Positive':
            label = torch.tensor(1)
        else:
            label = torch.tensor(0)
        return ID, img_MR_DCE, label

    def __len__(self):
        return self.len

# u
class Dataset_imgonlyUS(Dataset):
    def __init__(self, data_dir, split_path,data_set='train', augment=True):
        self.data_dir = data_dir
        split_data = pickle.load(open(split_path, 'rb'))
        self.ID_list = split_data[data_set][0]
        self.label_list = split_data[data_set][1]
        self.augment = augment

        if augment:
            self.transforms_2D = Compose([
                Rand2DElasticd(keys=['img', 'mask'], spacing=(20, 20), magnitude_range=(1, 2), mode="nearest",
                               padding_mode="zeros", prob=0.2),
                RandGaussianNoised(keys=['img'], mean=0.0, std=0.1, prob=0.1),
                RandRotated(keys=['img', 'mask'], range_x=np.pi / 360 * 30, prob=0.3, keep_size=True,
                           mode="nearest", padding_mode="zeros"),
            ])
            self.transforms_3D = Compose([
                Rand3DElasticd(keys=['img', 'mask'], sigma_range=(5, 7), magnitude_range=(10, 20), mode="nearest",
                               padding_mode="zeros", prob=0.2),
                RandGaussianNoised(keys=['img'], mean=0.0, std=0.1, prob=0.1),
                RandRotated(keys=['img', 'mask'], range_z=np.pi / 360 * 30, prob=0.3, keep_size=True,
                            mode="nearest", padding_mode="zeros"),
            ])
        self.len = len(self.ID_list)

    def __getitem__(self, idx):
        ID = self.ID_list[idx]
        label = self.label_list[idx]
        print(ID)

        # US H
        # print('US H')
        US_H, _, _, _ = NiiDataRead_2D(
            glob.glob(os.path.join(self.data_dir, 'US', ID, '{}_H.nii*'.format(ID)))[0])
        # US_H = US_H
        US_H = masked_Zscore_norm(US_H, mask_min=0, mask_max=255, percentile_min=None, percentile_max=None)
        US_H_GTV, _, _, _ = NiiDataRead_2D(
            glob.glob(os.path.join(self.data_dir, 'US', ID, '{}_H_GTV.nii*'.format(ID)))[0])
        US_H_GTV[US_H_GTV <= 0.5] = 0
        US_H_GTV[US_H_GTV > 0.5] = 1
        img_US_H = transform.resize(US_H, (384, 512), order=0, mode='constant', clip=False,
                                    preserve_range=True, anti_aliasing=False)
        mask_US_H = transform.resize(US_H_GTV, (384, 512), order=0, mode='constant', clip=False,
                                     preserve_range=True, anti_aliasing=False)
        img_US_H = img_US_H[np.newaxis, ...]
        mask_US_H = mask_US_H[np.newaxis, ...]
        # print(img_US_H.shape, mask_US_H.shape)

        # US V
        # print('US V')
        US_V, _, _, _ = NiiDataRead_2D(
            glob.glob(os.path.join(self.data_dir, 'US', ID, '{}_V.nii*'.format(ID)))[0])
        # US_V = US_V
        US_V = masked_Zscore_norm(US_V, mask_min=0, mask_max=255, percentile_min=None, percentile_max=None)
        US_V_GTV, _, _, _ = NiiDataRead_2D(
            glob.glob(os.path.join(self.data_dir, 'US', ID, '{}_V_GTV.nii*'.format(ID)))[0])
        US_V_GTV[US_V_GTV <= 0.5] = 0
        US_V_GTV[US_V_GTV > 0.5] = 1
        img_US_V = transform.resize(US_V, (384, 512), order=0, mode='constant', clip=False,
                                    preserve_range=True, anti_aliasing=False)
        mask_US_V = transform.resize(US_V_GTV, (384, 512), order=0, mode='constant', clip=False,
                                     preserve_range=True, anti_aliasing=False)
        img_US_V = img_US_V[np.newaxis, ...]
        mask_US_V = mask_US_V[np.newaxis, ...]
        # print(img_US_V.shape, mask_US_V.shape)

        if self.augment:
            augmented_US_H = self.transforms_2D({'img': img_US_H, 'mask': mask_US_H})
            img_US_H = np.concatenate((augmented_US_H['img'], augmented_US_H['mask']), axis=0)

            augmented_US_V = self.transforms_2D({'img': img_US_V, 'mask': mask_US_V})
            img_US_V = np.concatenate((augmented_US_V['img'], augmented_US_V['mask']), axis=0)
        else:
            img_US_H = np.concatenate((img_US_H, mask_US_H), axis=0)
            img_US_V = np.concatenate((img_US_V, mask_US_V), axis=0)

        img_US_H = torch.from_numpy(img_US_H)
        img_US_V = torch.from_numpy(img_US_V)
        if label == 'Positive':
            label = torch.tensor(1)
        else:
            label = torch.tensor(0)
        return ID, img_US_H, img_US_V, label

    def __len__(self):
        return self.len

# u
class Dataset_imgonlyMGUS(Dataset):
    def __init__(self, data_dir, split_path,data_set='train', augment=True):
        self.data_dir = data_dir
        split_data = pickle.load(open(split_path, 'rb'))
        self.ID_list = split_data[data_set][0]
        self.label_list = split_data[data_set][1]
        self.augment = augment

        if augment:
            self.transforms_2D = Compose([
                Rand2DElasticd(keys=['img', 'mask'], spacing=(20, 20), magnitude_range=(1, 2), mode="nearest",
                               padding_mode="zeros", prob=0.2),
                RandGaussianNoised(keys=['img'], mean=0.0, std=0.1, prob=0.1),
                RandRotated(keys=['img', 'mask'], range_x=np.pi / 360 * 30, prob=0.3, keep_size=True,
                           mode="nearest", padding_mode="zeros"),
            ])
            self.transforms_3D = Compose([
                Rand3DElasticd(keys=['img', 'mask'], sigma_range=(5, 7), magnitude_range=(10, 20), mode="nearest",
                               padding_mode="zeros", prob=0.2),
                RandGaussianNoised(keys=['img'], mean=0.0, std=0.1, prob=0.1),
                RandRotated(keys=['img', 'mask'], range_z=np.pi / 360 * 30, prob=0.3, keep_size=True,
                            mode="nearest", padding_mode="zeros"),
            ])
        self.len = len(self.ID_list)

    def __getitem__(self, idx):
        ID = self.ID_list[idx]
        label = self.label_list[idx]
        print(ID)

        # MG CC
        # print('MG CC')
        MG_CC, _, _, _ = NiiDataRead_2D(glob.glob(
            os.path.join(self.data_dir, 'MR+MG', label, ID, 'MG', '*{}_*CC.nii*'.format(ID)))[0])
        if MG_CC.max() <= 4095:
            MG_CC = masked_Zscore_norm(MG_CC, mask_min=0, mask_max=np.max(MG_CC), percentile_min=0.5,
                                        percentile_max=99.5)
        else:
            MG_CC = masked_Zscore_norm(MG_CC, mask_min=5000, mask_max=10000, percentile_min=0.5,
                                       percentile_max=99.5)
        MG_CC_GTV, _, _, _ = NiiDataRead_2D(glob.glob(
            os.path.join(self.data_dir, 'MR+MG', label, ID, 'MG', '*{}_*CC_GTV.nii*'.format(ID)))[0])
        MG_CC_GTV[MG_CC_GTV <= 0.5] = 0
        MG_CC_GTV[MG_CC_GTV > 0.5] = 1
        MG_CC = transform.resize(MG_CC, (640, 512), order=0, mode='constant', clip=False,
                                      preserve_range=True, anti_aliasing=False)
        MG_CC_GTV = transform.resize(MG_CC_GTV, (640, 512), order=0, mode='constant', clip=False,
                                       preserve_range=True, anti_aliasing=False)
        img_MG_CC = MG_CC[np.newaxis, ...]
        mask_MG_CC = MG_CC_GTV[np.newaxis, ...]
        # print(img_MG_CC.shape, mask_MG_CC.shape)

        # MG MLO
        # print('MG MLO')
        MG_MLO, _, _, _ = NiiDataRead_2D(glob.glob(
            os.path.join(self.data_dir, 'MR+MG', label, ID, 'MG', '*{}_*MLO.nii*'.format(ID)))[0])
        if MG_MLO.max() <= 4095:
            MG_MLO = masked_Zscore_norm(MG_MLO, mask_min=0, mask_max=np.max(MG_MLO), percentile_min=0.5,
                                       percentile_max=99.5)
        else:
            MG_MLO = masked_Zscore_norm(MG_MLO, mask_min=5000, mask_max=10000, percentile_min=0.5,
                                       percentile_max=99.5)
        MG_MLO_GTV, _, _, _ = NiiDataRead_2D(glob.glob(
            os.path.join(self.data_dir, 'MR+MG', label, ID, 'MG', '*{}_*MLO_GTV.nii*'.format(ID)))[0])
        MG_MLO_GTV[MG_MLO_GTV <= 0.5] = 0
        MG_MLO_GTV[MG_MLO_GTV > 0.5] = 1
        MG_MLO_GTVnd1, _, _, _ = NiiDataRead_2D(glob.glob(
            os.path.join(self.data_dir, 'MR+MG', label, ID, 'MG', '*{}_*MLO_GTVnd1.nii*'.format(ID)))[0])
        MG_MLO_GTVnd1[MG_MLO_GTVnd1 <= 0.5] = 0
        MG_MLO_GTVnd1[MG_MLO_GTVnd1 > 0.5] = 1
        MG_MLO = transform.resize(MG_MLO, (640, 512), order=0, mode='constant', clip=False,
                                     preserve_range=True, anti_aliasing=False)
        MG_MLO_GTV = transform.resize(MG_MLO_GTV, (640, 512), order=0, mode='constant', clip=False,
                                      preserve_range=True, anti_aliasing=False)
        MG_MLO_GTVnd1 = transform.resize(MG_MLO_GTVnd1, (640, 512), order=0, mode='constant', clip=False,
                                      preserve_range=True, anti_aliasing=False)
        img_MG_MLO = MG_MLO[np.newaxis, ...]
        mask_MG_MLO = np.concatenate((MG_MLO_GTV[np.newaxis, ...], MG_MLO_GTVnd1[np.newaxis, ...]), axis=0)
        # print(img_MG_MLO.shape, mask_MG_MLO.shape)

        # US H
        # print('US H')
        US_H, _, _, _ = NiiDataRead_2D(glob.glob(os.path.join(self.data_dir, 'US', ID, '{}_H.nii*'.format(ID)))[0])
        # US_H = US_H
        US_H = masked_Zscore_norm(US_H, mask_min=0, mask_max=255, percentile_min=None, percentile_max=None)
        US_H_GTV, _, _, _ = NiiDataRead_2D(
            glob.glob(os.path.join(self.data_dir, 'US', ID, '{}_H_GTV.nii*'.format(ID)))[0])
        US_H_GTV[US_H_GTV <= 0.5] = 0
        US_H_GTV[US_H_GTV > 0.5] = 1
        img_US_H = transform.resize(US_H, (384, 512), order=0, mode='constant', clip=False,
                                    preserve_range=True, anti_aliasing=False)
        mask_US_H = transform.resize(US_H_GTV, (384, 512), order=0, mode='constant', clip=False,
                                     preserve_range=True, anti_aliasing=False)
        img_US_H = img_US_H[np.newaxis, ...]
        mask_US_H = mask_US_H[np.newaxis, ...]
        # print(img_US_H.shape, mask_US_H.shape)

        # US V
        # print('US V')
        US_V, _, _, _ = NiiDataRead_2D(glob.glob(os.path.join(self.data_dir, 'US', ID, '{}_V.nii*'.format(ID)))[0])
        # US_V = US_V
        US_V = masked_Zscore_norm(US_V, mask_min=0, mask_max=255, percentile_min=None, percentile_max=None)
        US_V_GTV, _, _, _ = NiiDataRead_2D(
            glob.glob(os.path.join(self.data_dir, 'US', ID, '{}_V_GTV.nii*'.format(ID)))[0])
        US_V_GTV[US_V_GTV <= 0.5] = 0
        US_V_GTV[US_V_GTV > 0.5] = 1
        img_US_V = transform.resize(US_V, (384, 512), order=0, mode='constant', clip=False,
                                    preserve_range=True, anti_aliasing=False)
        mask_US_V = transform.resize(US_V_GTV, (384, 512), order=0, mode='constant', clip=False,
                                     preserve_range=True, anti_aliasing=False)
        img_US_V = img_US_V[np.newaxis, ...]
        mask_US_V = mask_US_V[np.newaxis, ...]
        # print(img_US_V.shape, mask_US_V.shape)

        if self.augment:

            augmented_MG_CC = self.transforms_2D({'img': img_MG_CC, 'mask': mask_MG_CC})
            img_MG_CC = np.concatenate((augmented_MG_CC['img'], augmented_MG_CC['mask']), axis=0)

            augmented_MG_MLO = self.transforms_2D({'img': img_MG_MLO, 'mask': mask_MG_MLO})
            img_MG_MLO = np.concatenate((augmented_MG_MLO['img'], augmented_MG_MLO['mask']), axis=0)

            augmented_US_H = self.transforms_2D({'img': img_US_H, 'mask': mask_US_H})
            img_US_H = np.concatenate((augmented_US_H['img'], augmented_US_H['mask']), axis=0)

            augmented_US_V = self.transforms_2D({'img': img_US_V, 'mask': mask_US_V})
            img_US_V = np.concatenate((augmented_US_V['img'], augmented_US_V['mask']), axis=0)
        else:
            img_MG_CC = np.concatenate((img_MG_CC, mask_MG_CC), axis=0)
            img_MG_MLO = np.concatenate((img_MG_MLO, mask_MG_MLO), axis=0)
            img_US_H = np.concatenate((img_US_H, mask_US_H), axis=0)
            img_US_V = np.concatenate((img_US_V, mask_US_V), axis=0)


        img_MG_CC = torch.from_numpy(img_MG_CC)
        img_MG_MLO = torch.from_numpy(img_MG_MLO)
        img_US_H = torch.from_numpy(img_US_H)
        img_US_V = torch.from_numpy(img_US_V)
        if label == 'Positive':
            label = torch.tensor(1)
        else:
            label = torch.tensor(0)
        return ID, img_MG_CC, img_MG_MLO, img_US_H, img_US_V, label

    def __len__(self):
        return self.len


# u
class Dataset_imgonlyMG(Dataset):
    def __init__(self, data_dir, split_path,data_set='train', augment=True):
        self.data_dir = data_dir
        split_data = pickle.load(open(split_path, 'rb'))
        self.ID_list = split_data[data_set][0]
        self.label_list = split_data[data_set][1]
        self.augment = augment

        if augment:
            self.transforms_2D = Compose([
                Rand2DElasticd(keys=['img', 'mask'], spacing=(20, 20), magnitude_range=(1, 2), mode="nearest",
                               padding_mode="zeros", prob=0.2),
                RandGaussianNoised(keys=['img'], mean=0.0, std=0.1, prob=0.1),
                RandRotated(keys=['img', 'mask'], range_x=np.pi / 360 * 30, prob=0.3, keep_size=True,
                           mode="nearest", padding_mode="zeros"),
            ])
            self.transforms_3D = Compose([
                Rand3DElasticd(keys=['img', 'mask'], sigma_range=(5, 7), magnitude_range=(10, 20), mode="nearest",
                               padding_mode="zeros", prob=0.2),
                RandGaussianNoised(keys=['img'], mean=0.0, std=0.1, prob=0.1),
                RandRotated(keys=['img', 'mask'], range_z=np.pi / 360 * 30, prob=0.3, keep_size=True,
                            mode="nearest", padding_mode="zeros"),
            ])
        self.len = len(self.ID_list)

    def __getitem__(self, idx):
        ID = self.ID_list[idx]
        label = self.label_list[idx]
        print(ID)

        # MG CC
        # print('MG CC')
        MG_CC, _, _, _ = NiiDataRead_2D(glob.glob(
            os.path.join(self.data_dir, 'MR+MG', label, ID, 'MG', '*{}_*CC.nii*'.format(ID)))[0])
        if MG_CC.max() <= 4095:
            MG_CC = masked_Zscore_norm(MG_CC, mask_min=0, mask_max=np.max(MG_CC), percentile_min=0.5,
                                        percentile_max=99.5)
        else:
            MG_CC = masked_Zscore_norm(MG_CC, mask_min=5000, mask_max=10000, percentile_min=0.5,
                                       percentile_max=99.5)
        MG_CC_GTV, _, _, _ = NiiDataRead_2D(glob.glob(
            os.path.join(self.data_dir, 'MR+MG', label, ID, 'MG', '*{}_*CC_GTV.nii*'.format(ID)))[0])
        MG_CC_GTV[MG_CC_GTV <= 0.5] = 0
        MG_CC_GTV[MG_CC_GTV > 0.5] = 1
        MG_CC = transform.resize(MG_CC, (640, 512), order=0, mode='constant', clip=False,
                                      preserve_range=True, anti_aliasing=False)
        MG_CC_GTV = transform.resize(MG_CC_GTV, (640, 512), order=0, mode='constant', clip=False,
                                       preserve_range=True, anti_aliasing=False)
        img_MG_CC = MG_CC[np.newaxis, ...]
        mask_MG_CC = MG_CC_GTV[np.newaxis, ...]
        # print(img_MG_CC.shape, mask_MG_CC.shape)

        # MG MLO
        # print('MG MLO')
        MG_MLO, _, _, _ = NiiDataRead_2D(glob.glob(
            os.path.join(self.data_dir, 'MR+MG', label, ID, 'MG', '*{}_*MLO.nii*'.format(ID)))[0])
        if MG_MLO.max() <= 4095:
            MG_MLO = masked_Zscore_norm(MG_MLO, mask_min=0, mask_max=np.max(MG_MLO), percentile_min=0.5,
                                       percentile_max=99.5)
        else:
            MG_MLO = masked_Zscore_norm(MG_MLO, mask_min=5000, mask_max=10000, percentile_min=0.5,
                                       percentile_max=99.5)
        MG_MLO_GTV, _, _, _ = NiiDataRead_2D(glob.glob(
            os.path.join(self.data_dir, 'MR+MG', label, ID, 'MG', '*{}_*MLO_GTV.nii*'.format(ID)))[0])
        MG_MLO_GTV[MG_MLO_GTV <= 0.5] = 0
        MG_MLO_GTV[MG_MLO_GTV > 0.5] = 1
        MG_MLO_GTVnd1, _, _, _ = NiiDataRead_2D(glob.glob(
            os.path.join(self.data_dir, 'MR+MG', label, ID, 'MG', '*{}_*MLO_GTVnd1.nii*'.format(ID)))[0])
        MG_MLO_GTVnd1[MG_MLO_GTVnd1 <= 0.5] = 0
        MG_MLO_GTVnd1[MG_MLO_GTVnd1 > 0.5] = 1
        MG_MLO = transform.resize(MG_MLO, (640, 512), order=0, mode='constant', clip=False,
                                     preserve_range=True, anti_aliasing=False)
        MG_MLO_GTV = transform.resize(MG_MLO_GTV, (640, 512), order=0, mode='constant', clip=False,
                                      preserve_range=True, anti_aliasing=False)
        MG_MLO_GTVnd1 = transform.resize(MG_MLO_GTVnd1, (640, 512), order=0, mode='constant', clip=False,
                                      preserve_range=True, anti_aliasing=False)
        img_MG_MLO = MG_MLO[np.newaxis, ...]
        mask_MG_MLO = np.concatenate((MG_MLO_GTV[np.newaxis, ...], MG_MLO_GTVnd1[np.newaxis, ...]), axis=0)
        # print(img_MG_MLO.shape, mask_MG_MLO.shape)

        if self.augment:

            augmented_MG_CC = self.transforms_2D({'img': img_MG_CC, 'mask': mask_MG_CC})
            img_MG_CC = np.concatenate((augmented_MG_CC['img'], augmented_MG_CC['mask']), axis=0)

            augmented_MG_MLO = self.transforms_2D({'img': img_MG_MLO, 'mask': mask_MG_MLO})
            img_MG_MLO = np.concatenate((augmented_MG_MLO['img'], augmented_MG_MLO['mask']), axis=0)
        else:
            img_MG_CC = np.concatenate((img_MG_CC, mask_MG_CC), axis=0)
            img_MG_MLO = np.concatenate((img_MG_MLO, mask_MG_MLO), axis=0)


        img_MG_CC = torch.from_numpy(img_MG_CC)
        img_MG_MLO = torch.from_numpy(img_MG_MLO)
        if label == 'Positive':
            label = torch.tensor(1)
        else:
            label = torch.tensor(0)
        return ID, img_MG_CC, img_MG_MLO, label

    def __len__(self):
        return self.len

# u
class Dataset_imgonly(Dataset):
    def __init__(self, data_dir, split_path,data_set='train', augment=True):
        self.data_dir = data_dir
        split_data = pickle.load(open(split_path, 'rb'))
        self.ID_list = split_data[data_set][0]
        self.label_list = split_data[data_set][1]
        self.augment = augment

        if augment:
            self.transforms_2D = Compose([
                Rand2DElasticd(keys=['img', 'mask'], spacing=(20, 20), magnitude_range=(1, 2), mode="nearest",
                               padding_mode="zeros", prob=0.2),
                RandGaussianNoised(keys=['img'], mean=0.0, std=0.1, prob=0.1),
                RandRotated(keys=['img', 'mask'], range_x=np.pi / 360 * 30, prob=0.3, keep_size=True,
                           mode="nearest", padding_mode="zeros"),
            ])
            self.transforms_3D = Compose([
                Rand3DElasticd(keys=['img', 'mask'], sigma_range=(5, 7), magnitude_range=(10, 20), mode="nearest",
                               padding_mode="zeros", prob=0.2),
                RandGaussianNoised(keys=['img'], mean=0.0, std=0.1, prob=0.1),
                RandRotated(keys=['img', 'mask'], range_z=np.pi / 360 * 30, prob=0.3, keep_size=True,
                            mode="nearest", padding_mode="zeros"),
            ])
        self.len = len(self.ID_list)

    def __getitem__(self, idx):
        ID = self.ID_list[idx]
        label = self.label_list[idx]
        print(ID)
        # MR DCE
        # print('MR DCE')
        MR_DCE, _, _, _ = NiiDataRead(glob.glob(
            os.path.join(self.data_dir, 'MR+MG', label, ID, '{}_DCE_N4.nii*'.format(ID)))[0])
        if MR_DCE.max() < 0:
            MR_DCE = masked_Zscore_norm(MR_DCE, mask_min=np.min(MR_DCE), mask_max=np.max(MR_DCE), percentile_min=None,
                                        percentile_max=99.5)
        else:
            MR_DCE = masked_Zscore_norm(MR_DCE, mask_min=0, mask_max=np.max(MR_DCE), percentile_min=None,
                                        percentile_max=99.5)
        MR_DCE_GTV, _, _, _ = NiiDataRead(glob.glob(
            os.path.join(self.data_dir, 'MR+MG', label, ID, '{}_DCE_GTV.nii*'.format(ID)))[0])
        MR_DCE_GTV[MR_DCE_GTV <= 0.5] = 0
        MR_DCE_GTV[MR_DCE_GTV > 0.5] = 1
        MR_DCE_GTVnd1, _, _, _ = NiiDataRead(glob.glob(
            os.path.join(self.data_dir, 'MR+MG', label, ID, '{}_DCE_GTVnd1.nii*'.format(ID)))[0])
        MR_DCE_GTVnd1[MR_DCE_GTVnd1 <= 0.5] = 0
        MR_DCE_GTVnd1[MR_DCE_GTVnd1 > 0.5] = 1
        img_MR_DCE = MR_DCE[np.newaxis, ...]
        mask_MR_DCE = np.concatenate((MR_DCE_GTV[np.newaxis, ...], MR_DCE_GTVnd1[np.newaxis, ...]), axis=0)
        img_MR_DCE = transform.resize(img_MR_DCE, (1, 144, 320, 320), order=0, mode='constant', clip=False,
                                          preserve_range=True, anti_aliasing=False)
        mask_MR_DCE = transform.resize(mask_MR_DCE, (2, 144, 320, 320), order=0, mode='constant', clip=False,
                                           preserve_range=True, anti_aliasing=False)
        # print(img_MR_DCE.shape, mask_MR_DCE.shape)

        # MG CC
        # print('MG CC')
        MG_CC, _, _, _ = NiiDataRead_2D(glob.glob(
            os.path.join(self.data_dir, 'MR+MG', label, ID, 'MG', '*{}_*CC.nii*'.format(ID)))[0])
        if MG_CC.max() <= 4095:
            MG_CC = masked_Zscore_norm(MG_CC, mask_min=0, mask_max=np.max(MG_CC), percentile_min=0.5,
                                        percentile_max=99.5)
        else:
            MG_CC = masked_Zscore_norm(MG_CC, mask_min=5000, mask_max=10000, percentile_min=0.5,
                                       percentile_max=99.5)
        MG_CC_GTV, _, _, _ = NiiDataRead_2D(glob.glob(
            os.path.join(self.data_dir, 'MR+MG', label, ID, 'MG', '*{}_*CC_GTV.nii*'.format(ID)))[0])
        MG_CC_GTV[MG_CC_GTV <= 0.5] = 0
        MG_CC_GTV[MG_CC_GTV > 0.5] = 1
        MG_CC = transform.resize(MG_CC, (640, 512), order=0, mode='constant', clip=False,
                                      preserve_range=True, anti_aliasing=False)
        MG_CC_GTV = transform.resize(MG_CC_GTV, (640, 512), order=0, mode='constant', clip=False,
                                       preserve_range=True, anti_aliasing=False)
        img_MG_CC = MG_CC[np.newaxis, ...]
        mask_MG_CC = MG_CC_GTV[np.newaxis, ...]
        # print(img_MG_CC.shape, mask_MG_CC.shape)

        # MG MLO
        # print('MG MLO')
        MG_MLO, _, _, _ = NiiDataRead_2D(glob.glob(
            os.path.join(self.data_dir, 'MR+MG', label, ID, 'MG', '*{}_*MLO.nii*'.format(ID)))[0])
        if MG_MLO.max() <= 4095:
            MG_MLO = masked_Zscore_norm(MG_MLO, mask_min=0, mask_max=np.max(MG_MLO), percentile_min=0.5,
                                       percentile_max=99.5)
        else:
            MG_MLO = masked_Zscore_norm(MG_MLO, mask_min=5000, mask_max=10000, percentile_min=0.5,
                                       percentile_max=99.5)
        MG_MLO_GTV, _, _, _ = NiiDataRead_2D(glob.glob(
            os.path.join(self.data_dir, 'MR+MG', label, ID, 'MG', '*{}_*MLO_GTV.nii*'.format(ID)))[0])
        MG_MLO_GTV[MG_MLO_GTV <= 0.5] = 0
        MG_MLO_GTV[MG_MLO_GTV > 0.5] = 1
        MG_MLO_GTVnd1, _, _, _ = NiiDataRead_2D(glob.glob(
            os.path.join(self.data_dir, 'MR+MG', label, ID, 'MG', '*{}_*MLO_GTVnd1.nii*'.format(ID)))[0])
        MG_MLO_GTVnd1[MG_MLO_GTVnd1 <= 0.5] = 0
        MG_MLO_GTVnd1[MG_MLO_GTVnd1 > 0.5] = 1
        MG_MLO = transform.resize(MG_MLO, (640, 512), order=0, mode='constant', clip=False,
                                     preserve_range=True, anti_aliasing=False)
        MG_MLO_GTV = transform.resize(MG_MLO_GTV, (640, 512), order=0, mode='constant', clip=False,
                                      preserve_range=True, anti_aliasing=False)
        MG_MLO_GTVnd1 = transform.resize(MG_MLO_GTVnd1, (640, 512), order=0, mode='constant', clip=False,
                                      preserve_range=True, anti_aliasing=False)
        img_MG_MLO = MG_MLO[np.newaxis, ...]
        mask_MG_MLO = np.concatenate((MG_MLO_GTV[np.newaxis, ...], MG_MLO_GTVnd1[np.newaxis, ...]), axis=0)
        # print(img_MG_MLO.shape, mask_MG_MLO.shape)

        # US H
        # print('US H')
        US_H, _, _, _ = NiiDataRead_2D(glob.glob(os.path.join(self.data_dir, 'US', ID, '{}_H.nii*'.format(ID)))[0])
        # US_H = US_H
        US_H = masked_Zscore_norm(US_H, mask_min=0, mask_max=255, percentile_min=None, percentile_max=None)
        US_H_GTV, _, _, _ = NiiDataRead_2D(
            glob.glob(os.path.join(self.data_dir, 'US', ID, '{}_H_GTV.nii*'.format(ID)))[0])
        US_H_GTV[US_H_GTV <= 0.5] = 0
        US_H_GTV[US_H_GTV > 0.5] = 1
        img_US_H = transform.resize(US_H, (384, 512), order=0, mode='constant', clip=False,
                                    preserve_range=True, anti_aliasing=False)
        mask_US_H = transform.resize(US_H_GTV, (384, 512), order=0, mode='constant', clip=False,
                                     preserve_range=True, anti_aliasing=False)
        img_US_H = img_US_H[np.newaxis, ...]
        mask_US_H = mask_US_H[np.newaxis, ...]
        # print(img_US_H.shape, mask_US_H.shape)

        # US V
        # print('US V')
        US_V, _, _, _ = NiiDataRead_2D(glob.glob(os.path.join(self.data_dir, 'US', ID, '{}_V.nii*'.format(ID)))[0])
        # US_V = US_V
        US_V = masked_Zscore_norm(US_V, mask_min=0, mask_max=255, percentile_min=None, percentile_max=None)
        US_V_GTV, _, _, _ = NiiDataRead_2D(
            glob.glob(os.path.join(self.data_dir, 'US', ID, '{}_V_GTV.nii*'.format(ID)))[0])
        US_V_GTV[US_V_GTV <= 0.5] = 0
        US_V_GTV[US_V_GTV > 0.5] = 1
        img_US_V = transform.resize(US_V, (384, 512), order=0, mode='constant', clip=False,
                                    preserve_range=True, anti_aliasing=False)
        mask_US_V = transform.resize(US_V_GTV, (384, 512), order=0, mode='constant', clip=False,
                                     preserve_range=True, anti_aliasing=False)
        img_US_V = img_US_V[np.newaxis, ...]
        mask_US_V = mask_US_V[np.newaxis, ...]
        # print(img_US_V.shape, mask_US_V.shape)

        if self.augment:

            augmented_MR_DCE = self.transforms_3D({'img': img_MR_DCE, 'mask': mask_MR_DCE})
            img_MR_DCE = np.concatenate((augmented_MR_DCE['img'], augmented_MR_DCE['mask']), axis=0)

            augmented_MG_CC = self.transforms_2D({'img': img_MG_CC, 'mask': mask_MG_CC})
            img_MG_CC = np.concatenate((augmented_MG_CC['img'], augmented_MG_CC['mask']), axis=0)

            augmented_MG_MLO = self.transforms_2D({'img': img_MG_MLO, 'mask': mask_MG_MLO})
            img_MG_MLO = np.concatenate((augmented_MG_MLO['img'], augmented_MG_MLO['mask']), axis=0)

            augmented_US_H = self.transforms_2D({'img': img_US_H, 'mask': mask_US_H})
            img_US_H = np.concatenate((augmented_US_H['img'], augmented_US_H['mask']), axis=0)

            augmented_US_V = self.transforms_2D({'img': img_US_V, 'mask': mask_US_V})
            img_US_V = np.concatenate((augmented_US_V['img'], augmented_US_V['mask']), axis=0)
        else:
            img_MR_DCE = np.concatenate((img_MR_DCE, mask_MR_DCE), axis=0)
            img_MG_CC = np.concatenate((img_MG_CC, mask_MG_CC), axis=0)
            img_MG_MLO = np.concatenate((img_MG_MLO, mask_MG_MLO), axis=0)
            img_US_H = np.concatenate((img_US_H, mask_US_H), axis=0)
            img_US_V = np.concatenate((img_US_V, mask_US_V), axis=0)


        img_MR_DCE = torch.from_numpy(img_MR_DCE)
        img_MG_CC = torch.from_numpy(img_MG_CC)
        img_MG_MLO = torch.from_numpy(img_MG_MLO)
        img_US_H = torch.from_numpy(img_US_H)
        img_US_V = torch.from_numpy(img_US_V)
        if label == 'Positive':
            label = torch.tensor(1)
        else:
            label = torch.tensor(0)
        return ID, img_MR_DCE, img_MG_CC, img_MG_MLO, img_US_H, img_US_V, label

    def __len__(self):
        return self.len


# u
class Dataset_imgonlyMRMG(Dataset):
    def __init__(self, data_dir, split_path,data_set='train', augment=True):
        self.data_dir = data_dir
        split_data = pickle.load(open(split_path, 'rb'))
        self.ID_list = split_data[data_set][0]
        self.label_list = split_data[data_set][1]
        self.augment = augment

        if augment:
            self.transforms_2D = Compose([
                Rand2DElasticd(keys=['img', 'mask'], spacing=(20, 20), magnitude_range=(1, 2), mode="nearest",
                               padding_mode="zeros", prob=0.2),
                RandGaussianNoised(keys=['img'], mean=0.0, std=0.1, prob=0.1),
                RandRotated(keys=['img', 'mask'], range_x=np.pi / 360 * 30, prob=0.3, keep_size=True,
                           mode="nearest", padding_mode="zeros"),
            ])
            self.transforms_3D = Compose([
                Rand3DElasticd(keys=['img', 'mask'], sigma_range=(5, 7), magnitude_range=(10, 20), mode="nearest",
                               padding_mode="zeros", prob=0.2),
                RandGaussianNoised(keys=['img'], mean=0.0, std=0.1, prob=0.1),
                RandRotated(keys=['img', 'mask'], range_z=np.pi / 360 * 30, prob=0.3, keep_size=True,
                            mode="nearest", padding_mode="zeros"),
            ])
        self.len = len(self.ID_list)

    def __getitem__(self, idx):
        ID = self.ID_list[idx]
        label = self.label_list[idx]
        print(ID)
        # MR DCE
        # print('MR DCE')
        MR_DCE, _, _, _ = NiiDataRead(glob.glob(
            os.path.join(self.data_dir, 'MR+MG', label, ID, '{}_DCE_N4.nii*'.format(ID)))[0])
        if MR_DCE.max() < 0:
            MR_DCE = masked_Zscore_norm(MR_DCE, mask_min=np.min(MR_DCE), mask_max=np.max(MR_DCE), percentile_min=None,
                                        percentile_max=99.5)
        else:
            MR_DCE = masked_Zscore_norm(MR_DCE, mask_min=0, mask_max=np.max(MR_DCE), percentile_min=None,
                                        percentile_max=99.5)
        MR_DCE_GTV, _, _, _ = NiiDataRead(glob.glob(
            os.path.join(self.data_dir, 'MR+MG', label, ID, '{}_DCE_GTV.nii*'.format(ID)))[0])
        MR_DCE_GTV[MR_DCE_GTV <= 0.5] = 0
        MR_DCE_GTV[MR_DCE_GTV > 0.5] = 1
        MR_DCE_GTVnd1, _, _, _ = NiiDataRead(glob.glob(
            os.path.join(self.data_dir, 'MR+MG', label, ID, '{}_DCE_GTVnd1.nii*'.format(ID)))[0])
        MR_DCE_GTVnd1[MR_DCE_GTVnd1 <= 0.5] = 0
        MR_DCE_GTVnd1[MR_DCE_GTVnd1 > 0.5] = 1
        img_MR_DCE = MR_DCE[np.newaxis, ...]
        mask_MR_DCE = np.concatenate((MR_DCE_GTV[np.newaxis, ...], MR_DCE_GTVnd1[np.newaxis, ...]), axis=0)
        img_MR_DCE = transform.resize(img_MR_DCE, (1, 144, 320, 320), order=0, mode='constant', clip=False,
                                          preserve_range=True, anti_aliasing=False)
        mask_MR_DCE = transform.resize(mask_MR_DCE, (2, 144, 320, 320), order=0, mode='constant', clip=False,
                                           preserve_range=True, anti_aliasing=False)
        # print(img_MR_DCE.shape, mask_MR_DCE.shape)

        # MG CC
        # print('MG CC')
        MG_CC, _, _, _ = NiiDataRead_2D(glob.glob(
            os.path.join(self.data_dir, 'MR+MG', label, ID, 'MG', '*{}_*CC.nii*'.format(ID)))[0])
        if MG_CC.max() <= 4095:
            MG_CC = masked_Zscore_norm(MG_CC, mask_min=0, mask_max=np.max(MG_CC), percentile_min=0.5,
                                        percentile_max=99.5)
        else:
            MG_CC = masked_Zscore_norm(MG_CC, mask_min=5000, mask_max=10000, percentile_min=0.5,
                                       percentile_max=99.5)
        MG_CC_GTV, _, _, _ = NiiDataRead_2D(glob.glob(
            os.path.join(self.data_dir, 'MR+MG', label, ID, 'MG', '*{}_*CC_GTV.nii*'.format(ID)))[0])
        MG_CC_GTV[MG_CC_GTV <= 0.5] = 0
        MG_CC_GTV[MG_CC_GTV > 0.5] = 1
        MG_CC = transform.resize(MG_CC, (640, 512), order=0, mode='constant', clip=False,
                                      preserve_range=True, anti_aliasing=False)
        MG_CC_GTV = transform.resize(MG_CC_GTV, (640, 512), order=0, mode='constant', clip=False,
                                       preserve_range=True, anti_aliasing=False)
        img_MG_CC = MG_CC[np.newaxis, ...]
        mask_MG_CC = MG_CC_GTV[np.newaxis, ...]
        # print(img_MG_CC.shape, mask_MG_CC.shape)

        # MG MLO
        # print('MG MLO')
        MG_MLO, _, _, _ = NiiDataRead_2D(glob.glob(
            os.path.join(self.data_dir, 'MR+MG', label, ID, 'MG', '*{}_*MLO.nii*'.format(ID)))[0])
        if MG_MLO.max() <= 4095:
            MG_MLO = masked_Zscore_norm(MG_MLO, mask_min=0, mask_max=np.max(MG_MLO), percentile_min=0.5,
                                       percentile_max=99.5)
        else:
            MG_MLO = masked_Zscore_norm(MG_MLO, mask_min=5000, mask_max=10000, percentile_min=0.5,
                                       percentile_max=99.5)
        MG_MLO_GTV, _, _, _ = NiiDataRead_2D(glob.glob(
            os.path.join(self.data_dir, 'MR+MG', label, ID, 'MG', '*{}_*MLO_GTV.nii*'.format(ID)))[0])
        MG_MLO_GTV[MG_MLO_GTV <= 0.5] = 0
        MG_MLO_GTV[MG_MLO_GTV > 0.5] = 1
        MG_MLO_GTVnd1, _, _, _ = NiiDataRead_2D(glob.glob(
            os.path.join(self.data_dir, 'MR+MG', label, ID, 'MG', '*{}_*MLO_GTVnd1.nii*'.format(ID)))[0])
        MG_MLO_GTVnd1[MG_MLO_GTVnd1 <= 0.5] = 0
        MG_MLO_GTVnd1[MG_MLO_GTVnd1 > 0.5] = 1
        MG_MLO = transform.resize(MG_MLO, (640, 512), order=0, mode='constant', clip=False,
                                     preserve_range=True, anti_aliasing=False)
        MG_MLO_GTV = transform.resize(MG_MLO_GTV, (640, 512), order=0, mode='constant', clip=False,
                                      preserve_range=True, anti_aliasing=False)
        MG_MLO_GTVnd1 = transform.resize(MG_MLO_GTVnd1, (640, 512), order=0, mode='constant', clip=False,
                                      preserve_range=True, anti_aliasing=False)
        img_MG_MLO = MG_MLO[np.newaxis, ...]
        mask_MG_MLO = np.concatenate((MG_MLO_GTV[np.newaxis, ...], MG_MLO_GTVnd1[np.newaxis, ...]), axis=0)
        # print(img_MG_MLO.shape, mask_MG_MLO.shape)

        if self.augment:

            augmented_MR_DCE = self.transforms_3D({'img': img_MR_DCE, 'mask': mask_MR_DCE})
            img_MR_DCE = np.concatenate((augmented_MR_DCE['img'], augmented_MR_DCE['mask']), axis=0)

            augmented_MG_CC = self.transforms_2D({'img': img_MG_CC, 'mask': mask_MG_CC})
            img_MG_CC = np.concatenate((augmented_MG_CC['img'], augmented_MG_CC['mask']), axis=0)

            augmented_MG_MLO = self.transforms_2D({'img': img_MG_MLO, 'mask': mask_MG_MLO})
            img_MG_MLO = np.concatenate((augmented_MG_MLO['img'], augmented_MG_MLO['mask']), axis=0)

        else:
            img_MR_DCE = np.concatenate((img_MR_DCE, mask_MR_DCE), axis=0)
            img_MG_CC = np.concatenate((img_MG_CC, mask_MG_CC), axis=0)
            img_MG_MLO = np.concatenate((img_MG_MLO, mask_MG_MLO), axis=0)


        img_MR_DCE = torch.from_numpy(img_MR_DCE)
        img_MG_CC = torch.from_numpy(img_MG_CC)
        img_MG_MLO = torch.from_numpy(img_MG_MLO)
        if label == 'Positive':
            label = torch.tensor(1)
        else:
            label = torch.tensor(0)
        return ID, img_MR_DCE, img_MG_CC, img_MG_MLO, label

    def __len__(self):
        return self.len


# u
class Dataset_imgonlyMRUS(Dataset):
    def __init__(self, data_dir, split_path,data_set='train', augment=True):
        self.data_dir = data_dir
        split_data = pickle.load(open(split_path, 'rb'))
        self.ID_list = split_data[data_set][0]
        self.label_list = split_data[data_set][1]
        self.augment = augment

        if augment:
            self.transforms_2D = Compose([
                Rand2DElasticd(keys=['img', 'mask'], spacing=(20, 20), magnitude_range=(1, 2), mode="nearest",
                               padding_mode="zeros", prob=0.2),
                RandGaussianNoised(keys=['img'], mean=0.0, std=0.1, prob=0.1),
                RandRotated(keys=['img', 'mask'], range_x=np.pi / 360 * 30, prob=0.3, keep_size=True,
                           mode="nearest", padding_mode="zeros"),
            ])
            self.transforms_3D = Compose([
                Rand3DElasticd(keys=['img', 'mask'], sigma_range=(5, 7), magnitude_range=(10, 20), mode="nearest",
                               padding_mode="zeros", prob=0.2),
                RandGaussianNoised(keys=['img'], mean=0.0, std=0.1, prob=0.1),
                RandRotated(keys=['img', 'mask'], range_z=np.pi / 360 * 30, prob=0.3, keep_size=True,
                            mode="nearest", padding_mode="zeros"),
            ])
        self.len = len(self.ID_list)

    def __getitem__(self, idx):
        ID = self.ID_list[idx]
        label = self.label_list[idx]
        print(ID)
        # MR DCE
        # print('MR DCE')
        MR_DCE, _, _, _ = NiiDataRead(glob.glob(
            os.path.join(self.data_dir, 'MR+MG', label, ID, '{}_DCE_N4.nii*'.format(ID)))[0])
        if MR_DCE.max() < 0:
            MR_DCE = masked_Zscore_norm(MR_DCE, mask_min=np.min(MR_DCE), mask_max=np.max(MR_DCE), percentile_min=None,
                                        percentile_max=99.5)
        else:
            MR_DCE = masked_Zscore_norm(MR_DCE, mask_min=0, mask_max=np.max(MR_DCE), percentile_min=None,
                                        percentile_max=99.5)
        MR_DCE_GTV, _, _, _ = NiiDataRead(glob.glob(
            os.path.join(self.data_dir, 'MR+MG', label, ID, '{}_DCE_GTV.nii*'.format(ID)))[0])
        MR_DCE_GTV[MR_DCE_GTV <= 0.5] = 0
        MR_DCE_GTV[MR_DCE_GTV > 0.5] = 1
        MR_DCE_GTVnd1, _, _, _ = NiiDataRead(glob.glob(
            os.path.join(self.data_dir, 'MR+MG', label, ID, '{}_DCE_GTVnd1.nii*'.format(ID)))[0])
        MR_DCE_GTVnd1[MR_DCE_GTVnd1 <= 0.5] = 0
        MR_DCE_GTVnd1[MR_DCE_GTVnd1 > 0.5] = 1
        img_MR_DCE = MR_DCE[np.newaxis, ...]
        mask_MR_DCE = np.concatenate((MR_DCE_GTV[np.newaxis, ...], MR_DCE_GTVnd1[np.newaxis, ...]), axis=0)
        img_MR_DCE = transform.resize(img_MR_DCE, (1, 144, 320, 320), order=0, mode='constant', clip=False,
                                          preserve_range=True, anti_aliasing=False)
        mask_MR_DCE = transform.resize(mask_MR_DCE, (2, 144, 320, 320), order=0, mode='constant', clip=False,
                                           preserve_range=True, anti_aliasing=False)
        # print(img_MR_DCE.shape, mask_MR_DCE.shape)

        # US H
        # print('US H')
        US_H, _, _, _ = NiiDataRead_2D(
            glob.glob(os.path.join(self.data_dir, 'US', ID, '{}_H.nii*'.format(ID)))[0])
        # US_H = US_H
        US_H = masked_Zscore_norm(US_H, mask_min=0, mask_max=255, percentile_min=None, percentile_max=None)
        US_H_GTV, _, _, _ = NiiDataRead_2D(
            glob.glob(os.path.join(self.data_dir, 'US', ID, '{}_H_GTV.nii*'.format(ID)))[0])
        US_H_GTV[US_H_GTV <= 0.5] = 0
        US_H_GTV[US_H_GTV > 0.5] = 1
        img_US_H = transform.resize(US_H, (384, 512), order=0, mode='constant', clip=False,
                                    preserve_range=True, anti_aliasing=False)
        mask_US_H = transform.resize(US_H_GTV, (384, 512), order=0, mode='constant', clip=False,
                                     preserve_range=True, anti_aliasing=False)
        img_US_H = img_US_H[np.newaxis, ...]
        mask_US_H = mask_US_H[np.newaxis, ...]
        # print(img_US_H.shape, mask_US_H.shape)

        # US V
        # print('US V')
        US_V, _, _, _ = NiiDataRead_2D(
            glob.glob(os.path.join(self.data_dir, 'US', ID, '{}_V.nii*'.format(ID)))[0])
        # US_V = US_V
        US_V = masked_Zscore_norm(US_V, mask_min=0, mask_max=255, percentile_min=None, percentile_max=None)
        US_V_GTV, _, _, _ = NiiDataRead_2D(
            glob.glob(os.path.join(self.data_dir, 'US', ID, '{}_V_GTV.nii*'.format(ID)))[0])
        US_V_GTV[US_V_GTV <= 0.5] = 0
        US_V_GTV[US_V_GTV > 0.5] = 1
        img_US_V = transform.resize(US_V, (384, 512), order=0, mode='constant', clip=False,
                                    preserve_range=True, anti_aliasing=False)
        mask_US_V = transform.resize(US_V_GTV, (384, 512), order=0, mode='constant', clip=False,
                                     preserve_range=True, anti_aliasing=False)
        img_US_V = img_US_V[np.newaxis, ...]
        mask_US_V = mask_US_V[np.newaxis, ...]
        # print(img_US_V.shape, mask_US_V.shape)

        if self.augment:

            augmented_MR_DCE = self.transforms_3D({'img': img_MR_DCE, 'mask': mask_MR_DCE})
            img_MR_DCE = np.concatenate((augmented_MR_DCE['img'], augmented_MR_DCE['mask']), axis=0)

            augmented_US_H = self.transforms_2D({'img': img_US_H, 'mask': mask_US_H})
            img_US_H = np.concatenate((augmented_US_H['img'], augmented_US_H['mask']), axis=0)

            augmented_US_V = self.transforms_2D({'img': img_US_V, 'mask': mask_US_V})
            img_US_V = np.concatenate((augmented_US_V['img'], augmented_US_V['mask']), axis=0)
        else:
            img_MR_DCE = np.concatenate((img_MR_DCE, mask_MR_DCE), axis=0)
            img_US_H = np.concatenate((img_US_H, mask_US_H), axis=0)
            img_US_V = np.concatenate((img_US_V, mask_US_V), axis=0)


        img_MR_DCE = torch.from_numpy(img_MR_DCE)
        img_US_H = torch.from_numpy(img_US_H)
        img_US_V = torch.from_numpy(img_US_V)
        if label == 'Positive':
            label = torch.tensor(1)
        else:
            label = torch.tensor(0)
        return ID, img_MR_DCE, img_US_H, img_US_V, label

    def __len__(self):
        return self.len


# u
class Dataset_img_clinical(Dataset):
    def __init__(self, data_dir, split_path,data_set='train', augment=True):

        self.data_dir = data_dir
        split_data = pickle.load(open(split_path, 'rb'))
        self.ID_list = split_data[data_set]
        self.augment = augment
        self.df = pd.read_excel(os.path.join(data_dir,'CP_info.xlsx'), sheet_name='Sheet1', dtype={'ID': str})
        self.df['ID'] = self.df['ID'].astype(str)

        if augment:
            self.transforms_2D = Compose([
                Rand2DElasticd(keys=['img', 'mask'], spacing=(20, 20), magnitude_range=(1, 2), mode="nearest",
                               padding_mode="zeros", prob=0.2),
                RandGaussianNoised(keys=['img'], mean=0.0, std=0.1, prob=0.1),
                RandRotated(keys=['img', 'mask'], range_x=np.pi / 360 * 30, prob=0.3, keep_size=True,
                           mode="nearest", padding_mode="zeros"),
            ])
            self.transforms_3D = Compose([
                Rand3DElasticd(keys=['img', 'mask'], sigma_range=(5, 7), magnitude_range=(10, 20), mode="nearest",
                               padding_mode="zeros", prob=0.2),
                RandGaussianNoised(keys=['img'], mean=0.0, std=0.1, prob=0.1),
                RandRotated(keys=['img', 'mask'], range_z=np.pi / 360 * 30, prob=0.3, keep_size=True,
                            mode="nearest", padding_mode="zeros"),
            ])
        self.len = len(self.ID_list)

    def __getitem__(self, idx):

        ID = self.ID_list[idx]['ID']
        label = self.ID_list[idx]['label']
        distance = self.ID_list[idx]['distance']
        clinical =[]
        clinical.append(torch.tensor([distance]))
        clinical = torch.cat(clinical)
        print(ID)
        # MR DCE
        # print('MR DCE')
        # print('------------------')
        # print(ID)
        MR_DCE, _, _, _ = NiiDataRead(glob.glob(
            os.path.join(self.data_dir, 'MR+MG', label, ID, '{}_DCE_N4.nii*'.format(ID)))[0])
        if MR_DCE.max() < 0:
            MR_DCE = masked_Zscore_norm(MR_DCE, mask_min=np.min(MR_DCE), mask_max=np.max(MR_DCE), percentile_min=None,
                                        percentile_max=99.5)
        else:
            MR_DCE = masked_Zscore_norm(MR_DCE, mask_min=0, mask_max=np.max(MR_DCE), percentile_min=None,
                                        percentile_max=99.5)
        MR_DCE_GTV, _, _, _ = NiiDataRead(glob.glob(
            os.path.join(self.data_dir, 'MR+MG', label, ID, '{}_DCE_GTV.nii*'.format(ID)))[0])
        MR_DCE_GTV[MR_DCE_GTV <= 0.5] = 0
        MR_DCE_GTV[MR_DCE_GTV > 0.5] = 1
        MR_DCE_GTVnd1, _, _, _ = NiiDataRead(glob.glob(
            os.path.join(self.data_dir, 'MR+MG', label, ID, '{}_DCE_GTVnd1.nii*'.format(ID)))[0])
        MR_DCE_GTVnd1[MR_DCE_GTVnd1 <= 0.5] = 0
        MR_DCE_GTVnd1[MR_DCE_GTVnd1 > 0.5] = 1
        img_MR_DCE = MR_DCE[np.newaxis, ...]
        mask_MR_DCE = np.concatenate((MR_DCE_GTV[np.newaxis, ...], MR_DCE_GTVnd1[np.newaxis, ...]), axis=0)
        img_MR_DCE = transform.resize(img_MR_DCE, (1, 144, 320, 320), order=0, mode='constant', clip=False,
                                          preserve_range=True, anti_aliasing=False)
        mask_MR_DCE = transform.resize(mask_MR_DCE, (2, 144, 320, 320), order=0, mode='constant', clip=False,
                                           preserve_range=True, anti_aliasing=False)
        # print(img_MR_DCE.shape, mask_MR_DCE.shape)
        # print('++++++++++++++++++++++++')
        # MG CC
        # print('MG CC')
        MG_CC, _, _, _ = NiiDataRead_2D(glob.glob(
            os.path.join(self.data_dir, 'MR+MG', label, ID, 'MG', '*{}_*CC.nii*'.format(ID)))[0])
        if MG_CC.max() <= 4095:
            MG_CC = masked_Zscore_norm(MG_CC, mask_min=0, mask_max=np.max(MG_CC), percentile_min=0.5,
                                        percentile_max=99.5)
        else:
            MG_CC = masked_Zscore_norm(MG_CC, mask_min=5000, mask_max=10000, percentile_min=0.5,
                                       percentile_max=99.5)
        MG_CC_GTV, _, _, _ = NiiDataRead_2D(glob.glob(
            os.path.join(self.data_dir, 'MR+MG', label, ID, 'MG', '*{}_*CC_GTV.nii*'.format(ID)))[0])
        MG_CC_GTV[MG_CC_GTV <= 0.5] = 0
        MG_CC_GTV[MG_CC_GTV > 0.5] = 1
        MG_CC = transform.resize(MG_CC, (640, 512), order=0, mode='constant', clip=False,
                                      preserve_range=True, anti_aliasing=False)
        MG_CC_GTV = transform.resize(MG_CC_GTV, (640, 512), order=0, mode='constant', clip=False,
                                       preserve_range=True, anti_aliasing=False)
        img_MG_CC = MG_CC[np.newaxis, ...]
        mask_MG_CC = MG_CC_GTV[np.newaxis, ...]
        # print(img_MG_CC.shape, mask_MG_CC.shape)

        # MG MLO
        # print('MG MLO')
        MG_MLO, _, _, _ = NiiDataRead_2D(glob.glob(
            os.path.join(self.data_dir, 'MR+MG', label, ID, 'MG', '*{}_*MLO.nii*'.format(ID)))[0])
        if MG_MLO.max() <= 4095:
            MG_MLO = masked_Zscore_norm(MG_MLO, mask_min=0, mask_max=np.max(MG_MLO), percentile_min=0.5,
                                       percentile_max=99.5)
        else:
            MG_MLO = masked_Zscore_norm(MG_MLO, mask_min=5000, mask_max=10000, percentile_min=0.5,
                                       percentile_max=99.5)
        MG_MLO_GTV, _, _, _ = NiiDataRead_2D(glob.glob(
            os.path.join(self.data_dir, 'MR+MG', label, ID, 'MG', '*{}_*MLO_GTV.nii*'.format(ID)))[0])
        MG_MLO_GTV[MG_MLO_GTV <= 0.5] = 0
        MG_MLO_GTV[MG_MLO_GTV > 0.5] = 1
        MG_MLO_GTVnd1, _, _, _ = NiiDataRead_2D(glob.glob(
            os.path.join(self.data_dir, 'MR+MG', label, ID, 'MG', '*{}_*MLO_GTVnd1.nii*'.format(ID)))[0])
        MG_MLO_GTVnd1[MG_MLO_GTVnd1 <= 0.5] = 0
        MG_MLO_GTVnd1[MG_MLO_GTVnd1 > 0.5] = 1
        MG_MLO = transform.resize(MG_MLO, (640, 512), order=0, mode='constant', clip=False,
                                     preserve_range=True, anti_aliasing=False)
        MG_MLO_GTV = transform.resize(MG_MLO_GTV, (640, 512), order=0, mode='constant', clip=False,
                                      preserve_range=True, anti_aliasing=False)
        MG_MLO_GTVnd1 = transform.resize(MG_MLO_GTVnd1, (640, 512), order=0, mode='constant', clip=False,
                                      preserve_range=True, anti_aliasing=False)
        img_MG_MLO = MG_MLO[np.newaxis, ...]
        mask_MG_MLO = np.concatenate((MG_MLO_GTV[np.newaxis, ...], MG_MLO_GTVnd1[np.newaxis, ...]), axis=0)
        # print(img_MG_MLO.shape, mask_MG_MLO.shape)

        # clinical and patology info
        Clinical_features = Extract_clinical_features(self.df, ID)
        Clinical_features=torch.cat((Clinical_features,clinical), dim=0)
        pathology_features = Extract_pathology_features(self.df, ID)

        # US H
        # print('US H')
        US_H, _, _, _ = NiiDataRead_2D(
            glob.glob(os.path.join(self.data_dir, 'US', ID, '{}_H.nii*'.format(ID)))[0])
        # US_H = US_H
        US_H = masked_Zscore_norm(US_H, mask_min=0, mask_max=255, percentile_min=None, percentile_max=None)
        US_H_GTV, _, _, _ = NiiDataRead_2D(
            glob.glob(os.path.join(self.data_dir, 'US', ID, '{}_H_GTV.nii*'.format(ID)))[0])
        US_H_GTV[US_H_GTV <= 0.5] = 0
        US_H_GTV[US_H_GTV > 0.5] = 1
        # US_H_GTV = np.max(US_H_GTV, axis=0, keepdims=True)
        img_US_H = transform.resize(US_H, (384, 512), order=0, mode='constant', clip=False,
                                    preserve_range=True, anti_aliasing=False)
        mask_US_H = transform.resize(US_H_GTV, (384, 512), order=0, mode='constant', clip=False,
                                     preserve_range=True, anti_aliasing=False)
        img_US_H = img_US_H[np.newaxis, ...]
        mask_US_H = mask_US_H[np.newaxis, ...]
        # print(img_US_H.shape, mask_US_H.shape)

        # US V
        # print('US V')
        US_V, _, _, _ = NiiDataRead_2D(
            glob.glob(os.path.join(self.data_dir, 'US', ID, '{}_V.nii*'.format(ID)))[0])
        # US_V = US_V
        US_V = masked_Zscore_norm(US_V, mask_min=0, mask_max=255, percentile_min=None, percentile_max=None)
        US_V_GTV, _, _, _ = NiiDataRead_2D(
            glob.glob(os.path.join(self.data_dir, 'US', ID, '{}_V_GTV.nii*'.format(ID)))[0])
        US_V_GTV[US_V_GTV <= 0.5] = 0
        US_V_GTV[US_V_GTV > 0.5] = 1
        # US_V_GTV = np.max(US_V_GTV, axis=0, keepdims=True)
        img_US_V = transform.resize(US_V, (384, 512), order=0, mode='constant', clip=False,
                                    preserve_range=True, anti_aliasing=False)
        mask_US_V = transform.resize(US_V_GTV, (384, 512), order=0, mode='constant', clip=False,
                                     preserve_range=True, anti_aliasing=False)
        img_US_V = img_US_V[np.newaxis, ...]
        mask_US_V = mask_US_V[np.newaxis, ...]
        # print(img_US_V.shape, mask_US_V.shape)

        if self.augment:

            augmented_MR_DCE = self.transforms_3D({'img': img_MR_DCE, 'mask': mask_MR_DCE})
            img_MR_DCE = np.concatenate((augmented_MR_DCE['img'], augmented_MR_DCE['mask']), axis=0)

            augmented_MG_CC = self.transforms_2D({'img': img_MG_CC, 'mask': mask_MG_CC})
            img_MG_CC = np.concatenate((augmented_MG_CC['img'], augmented_MG_CC['mask']), axis=0)

            augmented_MG_MLO = self.transforms_2D({'img': img_MG_MLO, 'mask': mask_MG_MLO})
            img_MG_MLO = np.concatenate((augmented_MG_MLO['img'], augmented_MG_MLO['mask']), axis=0)

            augmented_US_H = self.transforms_2D({'img': img_US_H, 'mask': mask_US_H})
            img_US_H = np.concatenate((augmented_US_H['img'], augmented_US_H['mask']), axis=0)

            augmented_US_V = self.transforms_2D({'img': img_US_V, 'mask': mask_US_V})
            img_US_V = np.concatenate((augmented_US_V['img'], augmented_US_V['mask']), axis=0)
            
            Clinical_features += torch.randn(*Clinical_features.size()) * 0.05
            pathology_features += torch.randn(*pathology_features.size()) * 0.05
        else:
            img_MR_DCE = np.concatenate((img_MR_DCE, mask_MR_DCE), axis=0)
            img_MG_CC = np.concatenate((img_MG_CC, mask_MG_CC), axis=0)
            img_MG_MLO = np.concatenate((img_MG_MLO, mask_MG_MLO), axis=0)
            img_US_H = np.concatenate((img_US_H, mask_US_H), axis=0)
            img_US_V = np.concatenate((img_US_V, mask_US_V), axis=0)


        img_MR_DCE = torch.from_numpy(img_MR_DCE)
        img_MG_CC = torch.from_numpy(img_MG_CC)
        img_MG_MLO = torch.from_numpy(img_MG_MLO)
        img_US_H = torch.from_numpy(img_US_H)
        img_US_V = torch.from_numpy(img_US_V)
        if label == 'Positive':
            label = torch.tensor(1)
        else:
            label = torch.tensor(0)
        return ID, img_MR_DCE, img_MG_CC, img_MG_MLO, img_US_H, img_US_V, Clinical_features, pathology_features, label

    def __len__(self):
        return self.len
# u
class Dataset_img_clinical_notdistance(Dataset):
    def __init__(self, data_dir, split_path, data_set='train', augment=True):

        self.data_dir = data_dir
        split_data = pickle.load(open(split_path, 'rb'))
        self.ID_list = split_data[data_set]
        self.augment = augment
        self.df = pd.read_excel(os.path.join(data_dir, 'CP_info.xlsx'),
                                sheet_name='Sheet1', dtype={'ID': str})
        self.df['ID'] = self.df['ID'].astype(str)

        if augment:
            self.transforms_2D = Compose([
                Rand2DElasticd(keys=['img', 'mask'], spacing=(20, 20), magnitude_range=(1, 2), mode="nearest",
                               padding_mode="zeros", prob=0.2),
                RandGaussianNoised(keys=['img'], mean=0.0, std=0.1, prob=0.1),
                RandRotated(keys=['img', 'mask'], range_x=np.pi / 360 * 30, prob=0.3, keep_size=True,
                            mode="nearest", padding_mode="zeros"),
            ])
            self.transforms_3D = Compose([
                Rand3DElasticd(keys=['img', 'mask'], sigma_range=(5, 7), magnitude_range=(10, 20), mode="nearest",
                               padding_mode="zeros", prob=0.2),
                RandGaussianNoised(keys=['img'], mean=0.0, std=0.1, prob=0.1),
                RandRotated(keys=['img', 'mask'], range_z=np.pi / 360 * 30, prob=0.3, keep_size=True,
                            mode="nearest", padding_mode="zeros"),
            ])
        self.len = len(self.ID_list)

    def __getitem__(self, idx):

        ID = self.ID_list[idx]['ID']
        label = self.ID_list[idx]['label']
        print(ID)
        # MR DCE
        # print('MR DCE')
        MR_DCE, _, _, _ = NiiDataRead(glob.glob(
            os.path.join(self.data_dir, 'MR+MG', label, ID, '{}_DCE_N4.nii*'.format(ID)))[0])
        if MR_DCE.max() < 0:
            MR_DCE = masked_Zscore_norm(MR_DCE, mask_min=np.min(MR_DCE), mask_max=np.max(MR_DCE), percentile_min=None,
                                        percentile_max=99.5)
        else:
            MR_DCE = masked_Zscore_norm(MR_DCE, mask_min=0, mask_max=np.max(MR_DCE), percentile_min=None,
                                        percentile_max=99.5)
        MR_DCE_GTV, _, _, _ = NiiDataRead(glob.glob(
            os.path.join(self.data_dir, 'MR+MG', label, ID, '{}_DCE_GTV.nii*'.format(ID)))[0])
        MR_DCE_GTV[MR_DCE_GTV <= 0.5] = 0
        MR_DCE_GTV[MR_DCE_GTV > 0.5] = 1
        MR_DCE_GTVnd1, _, _, _ = NiiDataRead(glob.glob(
            os.path.join(self.data_dir, 'MR+MG', label, ID, '{}_DCE_GTVnd1.nii*'.format(ID)))[0])
        MR_DCE_GTVnd1[MR_DCE_GTVnd1 <= 0.5] = 0
        MR_DCE_GTVnd1[MR_DCE_GTVnd1 > 0.5] = 1
        img_MR_DCE = MR_DCE[np.newaxis, ...]
        mask_MR_DCE = np.concatenate((MR_DCE_GTV[np.newaxis, ...], MR_DCE_GTVnd1[np.newaxis, ...]), axis=0)
        img_MR_DCE = transform.resize(img_MR_DCE, (1, 144, 320, 320), order=0, mode='constant', clip=False,
                                      preserve_range=True, anti_aliasing=False)
        mask_MR_DCE = transform.resize(mask_MR_DCE, (2, 144, 320, 320), order=0, mode='constant', clip=False,
                                       preserve_range=True, anti_aliasing=False)
        # print(img_MR_DCE.shape, mask_MR_DCE.shape)

        # MG CC
        # print('MG CC')
        MG_CC, _, _, _ = NiiDataRead_2D(glob.glob(
            os.path.join(self.data_dir, 'MR+MG', label, ID, 'MG', '*{}_*CC.nii*'.format(ID)))[0])
        if MG_CC.max() <= 4095:
            MG_CC = masked_Zscore_norm(MG_CC, mask_min=0, mask_max=np.max(MG_CC), percentile_min=0.5,
                                       percentile_max=99.5)
        else:
            MG_CC = masked_Zscore_norm(MG_CC, mask_min=5000, mask_max=10000, percentile_min=0.5,
                                       percentile_max=99.5)
        MG_CC_GTV, _, _, _ = NiiDataRead_2D(glob.glob(
            os.path.join(self.data_dir, 'MR+MG', label, ID, 'MG', '*{}_*CC_GTV.nii*'.format(ID)))[0])
        MG_CC_GTV[MG_CC_GTV <= 0.5] = 0
        MG_CC_GTV[MG_CC_GTV > 0.5] = 1
        MG_CC = transform.resize(MG_CC, (640, 512), order=0, mode='constant', clip=False,
                                 preserve_range=True, anti_aliasing=False)
        MG_CC_GTV = transform.resize(MG_CC_GTV, (640, 512), order=0, mode='constant', clip=False,
                                     preserve_range=True, anti_aliasing=False)
        img_MG_CC = MG_CC[np.newaxis, ...]
        mask_MG_CC = MG_CC_GTV[np.newaxis, ...]
        # print(img_MG_CC.shape, mask_MG_CC.shape)

        # MG MLO
        # print('MG MLO')
        MG_MLO, _, _, _ = NiiDataRead_2D(glob.glob(
            os.path.join(self.data_dir, 'MR+MG', label, ID, 'MG', '*{}_*MLO.nii*'.format(ID)))[0])
        if MG_MLO.max() <= 4095:
            MG_MLO = masked_Zscore_norm(MG_MLO, mask_min=0, mask_max=np.max(MG_MLO), percentile_min=0.5,
                                        percentile_max=99.5)
        else:
            MG_MLO = masked_Zscore_norm(MG_MLO, mask_min=5000, mask_max=10000, percentile_min=0.5,
                                        percentile_max=99.5)
        MG_MLO_GTV, _, _, _ = NiiDataRead_2D(glob.glob(
            os.path.join(self.data_dir, 'MR+MG', label, ID, 'MG', '*{}_*MLO_GTV.nii*'.format(ID)))[0])
        MG_MLO_GTV[MG_MLO_GTV <= 0.5] = 0
        MG_MLO_GTV[MG_MLO_GTV > 0.5] = 1
        MG_MLO_GTVnd1, _, _, _ = NiiDataRead_2D(glob.glob(
            os.path.join(self.data_dir, 'MR+MG', label, ID, 'MG', '*{}_*MLO_GTVnd1.nii*'.format(ID)))[0])
        MG_MLO_GTVnd1[MG_MLO_GTVnd1 <= 0.5] = 0
        MG_MLO_GTVnd1[MG_MLO_GTVnd1 > 0.5] = 1
        MG_MLO = transform.resize(MG_MLO, (640, 512), order=0, mode='constant', clip=False,
                                  preserve_range=True, anti_aliasing=False)
        MG_MLO_GTV = transform.resize(MG_MLO_GTV, (640, 512), order=0, mode='constant', clip=False,
                                      preserve_range=True, anti_aliasing=False)
        MG_MLO_GTVnd1 = transform.resize(MG_MLO_GTVnd1, (640, 512), order=0, mode='constant', clip=False,
                                         preserve_range=True, anti_aliasing=False)
        img_MG_MLO = MG_MLO[np.newaxis, ...]
        mask_MG_MLO = np.concatenate((MG_MLO_GTV[np.newaxis, ...], MG_MLO_GTVnd1[np.newaxis, ...]), axis=0)
        # print(img_MG_MLO.shape, mask_MG_MLO.shape)

        # clinical and patology info
        Clinical_features = Extract_clinical_features(self.df, ID)
        pathology_features = Extract_pathology_features(self.df, ID)

        # US H
        # print('US H')
        US_H, _, _, _ = NiiDataRead_2D(
            glob.glob(os.path.join(self.data_dir, 'US', ID, '{}_H.nii*'.format(ID)))[0])
        # US_H = US_H
        US_H = masked_Zscore_norm(US_H, mask_min=0, mask_max=255, percentile_min=None, percentile_max=None)
        US_H_GTV, _, _, _ = NiiDataRead_2D(
            glob.glob(os.path.join(self.data_dir, 'US', ID, '{}_H_GTV.nii*'.format(ID)))[0])
        US_H_GTV[US_H_GTV <= 0.5] = 0
        US_H_GTV[US_H_GTV > 0.5] = 1
        img_US_H = transform.resize(US_H, (384, 512), order=0, mode='constant', clip=False,
                                    preserve_range=True, anti_aliasing=False)
        mask_US_H = transform.resize(US_H_GTV, (384, 512), order=0, mode='constant', clip=False,
                                     preserve_range=True, anti_aliasing=False)
        img_US_H = img_US_H[np.newaxis, ...]
        mask_US_H = mask_US_H[np.newaxis, ...]
        # print(img_US_H.shape, mask_US_H.shape)

        # US V
        # print('US V')
        US_V, _, _, _ = NiiDataRead_2D(
            glob.glob(os.path.join(self.data_dir, 'US', ID, '{}_V.nii*'.format(ID)))[0])
        # US_V = US_V
        US_V = masked_Zscore_norm(US_V, mask_min=0, mask_max=255, percentile_min=None, percentile_max=None)
        US_V_GTV, _, _, _ = NiiDataRead_2D(
            glob.glob(os.path.join(self.data_dir, 'US', ID, '{}_V_GTV.nii*'.format(ID)))[0])
        US_V_GTV[US_V_GTV <= 0.5] = 0
        US_V_GTV[US_V_GTV > 0.5] = 1
        img_US_V = transform.resize(US_V, (384, 512), order=0, mode='constant', clip=False,
                                    preserve_range=True, anti_aliasing=False)
        mask_US_V = transform.resize(US_V_GTV, (384, 512), order=0, mode='constant', clip=False,
                                     preserve_range=True, anti_aliasing=False)
        img_US_V = img_US_V[np.newaxis, ...]
        mask_US_V = mask_US_V[np.newaxis, ...]
        # print(img_US_V.shape, mask_US_V.shape)

        if self.augment:

            augmented_MR_DCE = self.transforms_3D({'img': img_MR_DCE, 'mask': mask_MR_DCE})
            img_MR_DCE = np.concatenate((augmented_MR_DCE['img'], augmented_MR_DCE['mask']), axis=0)

            augmented_MG_CC = self.transforms_2D({'img': img_MG_CC, 'mask': mask_MG_CC})
            img_MG_CC = np.concatenate((augmented_MG_CC['img'], augmented_MG_CC['mask']), axis=0)

            augmented_MG_MLO = self.transforms_2D({'img': img_MG_MLO, 'mask': mask_MG_MLO})
            img_MG_MLO = np.concatenate((augmented_MG_MLO['img'], augmented_MG_MLO['mask']), axis=0)

            augmented_US_H = self.transforms_2D({'img': img_US_H, 'mask': mask_US_H})
            img_US_H = np.concatenate((augmented_US_H['img'], augmented_US_H['mask']), axis=0)

            augmented_US_V = self.transforms_2D({'img': img_US_V, 'mask': mask_US_V})
            img_US_V = np.concatenate((augmented_US_V['img'], augmented_US_V['mask']), axis=0)

            Clinical_features += torch.randn(*Clinical_features.size()) * 0.05
            pathology_features += torch.randn(*pathology_features.size()) * 0.05
        else:
            img_MR_DCE = np.concatenate((img_MR_DCE, mask_MR_DCE), axis=0)
            img_MG_CC = np.concatenate((img_MG_CC, mask_MG_CC), axis=0)
            img_MG_MLO = np.concatenate((img_MG_MLO, mask_MG_MLO), axis=0)
            img_US_H = np.concatenate((img_US_H, mask_US_H), axis=0)
            img_US_V = np.concatenate((img_US_V, mask_US_V), axis=0)

        img_MR_DCE = torch.from_numpy(img_MR_DCE)
        img_MG_CC = torch.from_numpy(img_MG_CC)
        img_MG_MLO = torch.from_numpy(img_MG_MLO)
        img_US_H = torch.from_numpy(img_US_H)
        img_US_V = torch.from_numpy(img_US_V)
        if label == 'Positive':
            label = torch.tensor(1)
        else:
            label = torch.tensor(0)
        return ID, img_MR_DCE, img_MG_CC, img_MG_MLO, img_US_H, img_US_V, Clinical_features, pathology_features, label

    def __len__(self):
        return self.len
# u
class Dataset_img_onlyclinical(Dataset):
    def __init__(self, data_dir, split_path, data_set='train', augment=True):

        self.data_dir = data_dir
        split_data = pickle.load(open(split_path, 'rb'))
        self.ID_list = split_data[data_set]
        self.augment = augment
        self.df = pd.read_excel(os.path.join(data_dir, 'CP_info.xlsx'),
                                sheet_name='Sheet1', dtype={'ID': str})
        self.df['ID'] = self.df['ID'].astype(str)

        if augment:
            self.transforms_2D = Compose([
                Rand2DElasticd(keys=['img', 'mask'], spacing=(20, 20), magnitude_range=(1, 2), mode="nearest",
                               padding_mode="zeros", prob=0.2),
                RandGaussianNoised(keys=['img'], mean=0.0, std=0.1, prob=0.1),
                RandRotated(keys=['img', 'mask'], range_x=np.pi / 360 * 30, prob=0.3, keep_size=True,
                            mode="nearest", padding_mode="zeros"),
            ])
            self.transforms_3D = Compose([
                Rand3DElasticd(keys=['img', 'mask'], sigma_range=(5, 7), magnitude_range=(10, 20), mode="nearest",
                               padding_mode="zeros", prob=0.2),
                RandGaussianNoised(keys=['img'], mean=0.0, std=0.1, prob=0.1),
                RandRotated(keys=['img', 'mask'], range_z=np.pi / 360 * 30, prob=0.3, keep_size=True,
                            mode="nearest", padding_mode="zeros"),
            ])
        self.len = len(self.ID_list)

    def __getitem__(self, idx):

        ID = self.ID_list[idx]['ID']
        label = self.ID_list[idx]['label']
        distance = self.ID_list[idx]['distance']
        clinical = []
        clinical.append(torch.tensor([distance]))
        clinical = torch.cat(clinical)
        print(ID)

        # clinical and patology info
        Clinical_features = Extract_clinical_features(self.df, ID)
        Clinical_features = torch.cat((Clinical_features, clinical), dim=0)
        pathology_features = Extract_pathology_features(self.df, ID)

        if self.augment:
            Clinical_features += torch.randn(*Clinical_features.size()) * 0.05
            pathology_features += torch.randn(*pathology_features.size()) * 0.05

        if label == 'Positive':
            label = torch.tensor(1)
        else:
            label = torch.tensor(0)
        return ID, Clinical_features, pathology_features, label

    def __len__(self):
        return self.len
# u
class Dataset_img_onlyclinicalnotdistance(Dataset):
    def __init__(self, data_dir, split_path, data_set='train', augment=True):

        self.data_dir = data_dir
        split_data = pickle.load(open(split_path, 'rb'))
        self.ID_list = split_data[data_set]
        self.augment = augment
        self.df = pd.read_excel(os.path.join(data_dir, 'CP_info.xlsx'),
                                sheet_name='Sheet1', dtype={'ID': str})
        self.df['ID'] = self.df['ID'].astype(str)

        if augment:
            self.transforms_2D = Compose([
                Rand2DElasticd(keys=['img', 'mask'], spacing=(20, 20), magnitude_range=(1, 2), mode="nearest",
                               padding_mode="zeros", prob=0.2),
                RandGaussianNoised(keys=['img'], mean=0.0, std=0.1, prob=0.1),
                RandRotated(keys=['img', 'mask'], range_x=np.pi / 360 * 30, prob=0.3, keep_size=True,
                            mode="nearest", padding_mode="zeros"),
            ])
            self.transforms_3D = Compose([
                Rand3DElasticd(keys=['img', 'mask'], sigma_range=(5, 7), magnitude_range=(10, 20), mode="nearest",
                               padding_mode="zeros", prob=0.2),
                RandGaussianNoised(keys=['img'], mean=0.0, std=0.1, prob=0.1),
                RandRotated(keys=['img', 'mask'], range_z=np.pi / 360 * 30, prob=0.3, keep_size=True,
                            mode="nearest", padding_mode="zeros"),
            ])
        self.len = len(self.ID_list)

    def __getitem__(self, idx):

        ID = self.ID_list[idx]['ID']
        label = self.ID_list[idx]['label']
        print(ID)

        # clinical and patology info
        Clinical_features = Extract_clinical_features(self.df, ID)
        pathology_features = Extract_pathology_features(self.df, ID)

        if self.augment:
            Clinical_features += torch.randn(*Clinical_features.size()) * 0.05
            pathology_features += torch.randn(*pathology_features.size()) * 0.05

        if label == 'Positive':
            label = torch.tensor(1)
        else:
            label = torch.tensor(0)
        return ID, Clinical_features, pathology_features, label

    def __len__(self):
        return self.len


if __name__ == '__main__':
    print('--------------------')