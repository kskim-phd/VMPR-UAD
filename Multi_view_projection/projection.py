import nibabel as nib
import matplotlib.pyplot as plt
import glob, os, cv2
import numpy as np
from tqdm import tqdm

def crop_area(img,mask):
    for idx in range(0, mask.shape[1]):
        if mask[:, idx].sum() != 0:
            l_idx = idx
            break
    for idx in range(mask.shape[1] - 1, 0, -1):
        if mask[:, idx].sum() != 0:
            r_idx = idx
            break
    img = img[:, l_idx:r_idx + 1, ]
    mask = mask[:, l_idx:r_idx + 1]
    for idx in range(0, mask.shape[0]):
        if mask[idx, :].sum() != 0:
            l_idx = idx
            break
    for idx in range(mask.shape[0] - 1, 0, -1):
        if mask[idx, :].sum() != 0:
            r_idx = idx
            break
    img = img[l_idx:r_idx + 1, :]
    mask = mask[l_idx:r_idx + 1, :]
    return img,mask


def normalize(volume, max=0,min=-800):
    volume[volume < min] = min
    volume[volume > max] = max
    volume=(volume-min)/(max-min)

    return volume
iter=2
savepath = '../projection_data/'
projection_name = ['r_a', 'r_c', 'r_s', 'l_a', 'l_c', 'l_s']
maskpath = '../dataset/segmentation_mask/'
patientpath = '../dataset/'
names = glob.glob(patientpath+'*.nii.gz')
os.makedirs(savepath,exist_ok=True)

for name in tqdm(names):

    dataarray=nib.load(name).get_fdata()
    dataarray[dataarray < -1024] = -1024
    dataarray = normalize(dataarray)

    dir_name = name.split('/')[-1].replace('.nii.gz', '')
    maskdata = nib.load(maskpath + name.split('/')[-1].replace('.nii.gz', '_mask.nii.gz')).get_fdata()

    kernel = np.ones((3, 3))
    for idx in range(maskdata.shape[0]):
        for _ in range(iter):
            try:
                maskdata[idx, :, :] = cv2.erode(maskdata[idx, :, :], kernel, iterations=1)
            except:
                pass
    leftmask = np.zeros(maskdata.shape)
    leftmask[maskdata == 2] = 1
    rightmask = np.zeros(maskdata.shape)
    rightmask[maskdata == 1] = 1
    right = dataarray.copy()  # * rightmask
    left = dataarray.copy()  # * leftmask
    right[rightmask != 1] = right.min()
    left[leftmask != 1] = left.min()
    projection_list=[]
    for idx in range(3):
        projection_list.append(np.max(right.copy(), axis=idx)[::-1, :])

    for idx in range(3):
        projection_list.append(np.max(left.copy(), axis=idx)[::-1, :])


    rightmask = np.array(rightmask, dtype='uint8')
    leftmask = np.array(leftmask, dtype='uint8')
    projection_masklist = []
    for idx in range(3):
        mask=np.max(rightmask.copy(), axis=idx)[::-1, :]
        mask[mask>0]=1
        projection_masklist.append(mask)

    for idx in range(3):
        mask = np.max(leftmask.copy(), axis=idx)[::-1, :]
        mask[mask>0]=1
        projection_masklist.append(mask)


    patient_name = name.split('/')[-1].replace('.nii.gz', '')

    for position, img, mask in zip(projection_name,projection_list,projection_masklist):
        img, mask = crop_area(img, mask)
        plt.imsave(savepath  + patient_name + position+'.png', img, cmap='gray')
        plt.imsave(savepath + patient_name + position+'_mask.png', mask, cmap='gray')
