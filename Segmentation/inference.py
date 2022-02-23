from lungmask import mask
import glob,os
import nibabel as nib
from tqdm import tqdm
data_dir ='../dataset/'
datas=glob.glob(data_dir+'*.nii.gz')
savedir=data_dir+'segmentation_mask/'
os.makedirs(savedir,exist_ok=True)
for data in tqdm(datas):

    matrixs = nib.load(data).get_fdata()
    matrixs[matrixs<-1024]=-1024
    segmentation = mask.apply(matrixs)
    segmentation_mask = nib.Nifti1Image(segmentation, None)
    nib.save(segmentation_mask,savedir+data.replace('.nii.gz','_mask.nii.gz').split('/')[-1])
