"""
Measures the H, S, V variation between MIST and BCI H&E images. 
(1) Mask out background (low saturation). 
(2) Get median for each image for H, S, V. 
(3) Get mean / sd for the per-image means.
"""
import os, sys
import numpy as np
from pathlib import Path
from PIL import Image
from matplotlib.colors import rgb_to_hsv
from tqdm import tqdm
from scipy.stats import wassersteiyou n_distance

MIST_DIR = '/scratch/users/arshmeet/virtual_stain/raw_data/MIST/HER2/TrainValAB/trainA'
BCI_DIR  = '/scratch/users/arshmeet/virtual_stain/raw_data/BCI/BCI_dataset/HE/train'

def hsv_medians(dir, n):
    """
    Per image H, S, V medians.
    """
    imgfiles = [f for f in os.listdir(dir) if f.endswith(('.png', '.jpg'))]
    np.random.seed(42) 
    imgfiles = np.random.choice(imgfiles, n, replace = False) 
    H = []
    S = [] 
    V = []
    for imgfile in tqdm(imgfiles):
        # open image
        filepath = os.path.join(dir, imgfile) 
        img_rgb = np.array(Image.open(filepath).convert('RGB'), dtype=np.float32) / 255.0
        img_hsv = rgb_to_hsv(img_rgb)
        # seperate each channel
        hue, sat, val = img_hsv[:, :, 0], img_hsv[:, :, 1], img_hsv[:, :, 2] 
        # mask out white background + get HSV medians 
        mask = (sat > 0.1)
        # check if the mask is all white (or 90 percent white), background patch 
        if mask.sum() < 0.1 * img_rgb.shape[0] * img_rgb.shape[1]: 
            continue 
        H.append(np.median(hue[mask]))
        S.append(np.median(sat[mask]))
        V.append(np.median(val[mask]))
    return H, S, V

# 3896 Training images in BCI, more in MIST, use max number 3896
MH, MS, MV = hsv_medians(MIST_DIR, 3896)
BH, BS, BV = hsv_medians(BCI_DIR, 3896)
np.save('/scratch/users/arshmeet/virtual_stain/logs/hsv_medians.npy',
        {'MH': MH, 'MS': MS, 'MV': MV, 'BH': BH, 'BS': BS, 'BV': BV}) # for future ref 

def abs_diff_means(med1, med2):
    return abs(np.nanmean(med1) - np.nanmean(med2))
def pooled_sd(med1, med2):
    return np.sqrt((np.nanstd(med1)**2 + np.nanstd(med2)**2)/2)

print(f"Difference in Average of Medians:")
print(f"H: {abs_diff_means(MH, BH)*360:.1f}°, S: {abs_diff_means(MS, BS):.1f}, V: {abs_diff_means(MV, BV):.1f}")
print(f"Pooled SD:")
print(f"H: {pooled_sd(MH, BH)*360:.1f}°, S: {pooled_sd(MS, BS):.3f}, V: {pooled_sd(MV, BV):.3f}")
print(f"Wasserstein Distance:")
print(f"H: {wasserstein_distance(MH, BH)*360:.1f}°, S: {wasserstein_distance(MS, BS)*100:.1f}%, V: {wasserstein_distance(MV, BV)*100:.1f}%")