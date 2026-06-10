"""
Creates the perturbed datasets. 
"""
import os 
import numpy as np 
from PIL import Image
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb

# From MIST vs BCI inter-dataset HSV spread
hue_shift = 30
sat_shift = 0.25 
val_shift = 0.05 

def perturbation(orig_dir, pert_dir):
    origfiles = [f for f in os.listdir(orig_dir) if f.endswith(('.png', '.jpg'))] 
    for origfile in origfiles:
        # get hsv image 
        filepath = os.path.join(orig_dir, origfile) 
        orig_rgb = np.array(Image.open(filepath).convert('RGB'), dtype=np.float32) / 255.0
        orig_hsv = rgb_to_hsv(orig_rgb) 
        H, S, V = orig_hsv[:, :, 0], orig_hsv[:, :, 1], orig_hsv[:, :, 2]
        
        # define perturbations
        hue_pert = np.random.uniform(-hue_shift, hue_shift) / 360.0
        sat_pert = np.random.uniform(1-sat_shift, 1+sat_shift)
        val_pert = np.random.uniform(1-val_shift, 1+val_shift)
        
        # apply shift
        H = (H + hue_pert) % 1.0
        S = np.clip(S * sat_pert, 0.0, 1.0)
        V = np.clip(V * val_pert, 0.0, 1.0)
        
        pert_hsv = np.stack([H,S,V], axis=-1) # combine channels
        pert_rgb = (hsv_to_rgb(pert_hsv) * 255.0).astype(np.uint8)
        
        # save to dir 
        savepath =  os.path.join(pert_dir, origfile)
        Image.fromarray(pert_rgb).save(savepath)

MIST_TEST = 'raw_data/MIST/HER2/TrainValAB/valA'
BCI_TEST  = 'raw_data/BCI/BCI_dataset/HE/test'
for dataset in [MIST_TEST, BCI_TEST]: 
    pert_dir = dataset + '_' + 'HSV'  # name of the folder we create
    if os.path.exists(pert_dir + '_done.npy'):
        continue
    os.makedirs(pert_dir, exist_ok = True) 
    print(f"Perturbing {dataset} -> {pert_dir}") 
    perturbation(dataset, pert_dir)
    np.save(pert_dir + '_done.npy', np.array([1]))