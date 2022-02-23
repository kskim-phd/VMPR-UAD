import os
from tqdm import tqdm
import cv2
import numpy as np
import nibabel as nib
import glob
import pydicom as dcm
import matplotlib.pyplot as plt
import pickle

thr_q=50
view_name=['l_c', 'l_a', 'l_s', 'r_c', 'r_a', 'r_s']

patient = glob.glob('../dataset/*.nii.gz')
cam_path='../result/cammap/'
savedir='../result_map/'
maskfactory='../projection_data/'
path ='../dataset/'
mask_3Dpath = '../dataset/segmentation_mask/'

def load_total_view(patient_name):
    anomalys = []
    view_mask = []
    for name in view_name:
        anomaly = np.load(cam_path+patient_name+name+'.npy')
        mask = cv2.imread(maskfactory + patient_name+ name+'_mask.png',0)
        mask[mask > 0] = 1
        anomaly= cv2.resize(anomaly, mask.shape[::-1])
        if '_a' in name:
            mask=mask[::-1, :]
            anomaly=anomaly[::-1, :]
        anomalys.append(anomaly)
        view_mask.append(mask)
    return anomalys,view_mask

def min_max_total_normalize(maps,masks,thr):
    min_map =[]
    for anomap,mask in zip(maps,masks):
        min_map.append(anomap[mask!=0])
    min_vals = np.percentile(np.concatenate(min_map),thr)
    max_vals = np.max(np.concatenate(min_map))
    norm_maps=[]
    for anomap,mask in zip(maps,masks):
        anomap[mask==0]=min_vals
        anomap[anomap<min_vals]=min_vals
        anomap=anomap-min_vals
        anomap=anomap/max_vals
        norm_maps.append(anomap)
    return norm_maps
def generate_3D(maps):

    position=[1,0,2]
    lung_3D = []
    for views in [maps[:3],maps[3:]]:
        lung_frame = np.zeros((views[0].shape[0], views[1].shape[0], views[1].shape[1]))
        for idx,map in enumerate(views):
            for iters in range(lung_frame.shape[position[idx]]):
                if position[idx]==2:
                    lung_frame[:, :, iters] += map
                elif position[idx]==1:
                    lung_frame[:, iters, :] += map
                else:
                    lung_frame[iters, :, :] += map
        lung_3D.append(lung_frame)
    return lung_3D

def start_point(mask_3D,l_lung,r_lung,thr):
    bin_arr = np.zeros(mask_3D.shape)
    for u_idx in range(mask_3D.shape[0]):
        if 2 in mask_3D[u_idx, :, :]:
            left_up = u_idx
            break
    for f_idx in range(mask_3D.shape[1]):
        if 2 in mask_3D[:, f_idx, :]:
            left_f = f_idx
            break
    for s_idx in range(mask_3D.shape[2]):
        if 2 in mask_3D[:, :, s_idx]:
            left_side = s_idx
            break

    for u_idx in range(mask_3D.shape[0]):
        if 1 in mask_3D[u_idx, :, :]:
            right_up = u_idx
            break
    for f_idx in range(mask_3D.shape[1]):
        if 1 in mask_3D[:, f_idx, :]:
            right_f = f_idx
            break
    for s_idx in range(mask_3D.shape[2]):
        if 1 in mask_3D[:, :, s_idx]:
            right_side = s_idx
            break
    bin_arr[left_up:left_up + l_lung.shape[0], left_f:left_f + l_lung.shape[1],
    left_side:left_side + l_lung.shape[2]] = l_lung

    bin_arr[right_up:right_up + r_lung.shape[0], right_f:right_f + r_lung.shape[1],
    right_side:right_side + r_lung.shape[2]] = r_lung
    bin_arr[mask_3D == 0] = 0
    min_val = np.percentile(bin_arr[np.nonzero(bin_arr)].ravel(), thr)
    bin_arr[bin_arr < min_val] = min_val
    bin_arr = np.array((bin_arr - bin_arr.min()) / (bin_arr.max() - bin_arr.min()) * 255, dtype='uint8')
    return bin_arr

def slice_wise_save_map(dataarray, anomalymap_3D, patient_n):
    os.makedirs(savedir+patient_n ,exist_ok=True)
    for map_idx in range(anomalymap_3D.shape[0]):
        heat = cv2.applyColorMap(anomalymap_3D[map_idx, :, :].copy(), cv2.COLORMAP_JET)
        X = np.expand_dims(dataarray[map_idx, :, :].copy(), axis=2)
        concat_2d = np.concatenate([X, X, X], axis=2)
        heat_img = cv2.addWeighted(concat_2d, 0.7, heat, 0.3, 0)
        cv2.imwrite(savedir + patient_n + '/slice_' + str(map_idx) + '.png', heat_img)


for name in tqdm(patient):
    patientname= name.replace('.nii.gz','').split('/')[-1]
    dataarray = nib.load(name).get_fdata()
    dataarray = np.array(dataarray)[::-1, :, :]
    dataarray[dataarray < -1024] = -1024
    dataarray[dataarray > 1024] = 1024
    dataarray=np.array((dataarray - dataarray.min()) / (dataarray.max() - dataarray.min()) * 255, dtype='uint8')
    anomaly_view,mask_view=load_total_view(patientname)

    anomaly_view = min_max_total_normalize(anomaly_view,mask_view,thr_q)

    maskitem = glob.glob(mask_3Dpath + patientname+'_mask.nii.gz')[0]
    maskdata = nib.load(maskitem).get_fdata()
    maskdata = maskdata[::-1, :, :]

    left_lung, right_lung = generate_3D(anomaly_view)

    anomalymap_3D = start_point(maskdata, left_lung, right_lung,thr_q)

    slice_wise_save_map(dataarray, anomalymap_3D, patientname)



