import os
import random
import torch
import numpy as np
from skimage import io
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
from tqdm import tqdm
from torchvision.transforms.functional import to_tensor
from pytorch_fid.fid_score import calculate_activation_statistics, calculate_frechet_distance
from pytorch_fid.inception import InceptionV3
from cellpose import models
from scipy.stats import spearmanr
import sys

from util.perceptual import PerceptualHashValue
sys.path.insert(0, '/scratch/users/arshmeet/virtual_stain/TDKstain')
from preprocess.get_nuclei_map import get_ihc_channel, get_h_nuclei, get_dab_nuclei, get_nuclei_map
from preprocess.get_dab_mask import get_dab_mask

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--pred_dir', required=True)
parser.add_argument('--targ_dir', required=True)
parser.add_argument('--num_eval', type=int, default=0, help='Num images to sample for BIM.')
args = parser.parse_args()
pred_dir = args.pred_dir
targ_dir = args.targ_dir

img_list = [f for f in os.listdir(pred_dir) if f.endswith(('.png', '.jpg'))]
img_format = '.' + img_list[0].split('.')[-1]
img_list = [f.replace('.png', '').replace('.jpg', '') for f in img_list]
random.seed(0)
random.shuffle(img_list)
# BIM on a subsample for efficiency. 
img_list_bim = img_list
if args.num_eval > 0:
    img_list_bim = img_list[:args.num_eval]

#________________________________________________________________________________________#
# PHV statistics
device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
layers = ['layer_1', 'layer_2', 'layer_3', 'layer_4']
PHV = PerceptualHashValue(
        T=0.01, network='resnet50', layers=layers, 
        resize=False, resize_mode='bilinear',
        instance_normalized=False).to(device)
all_phv = []
for i in tqdm(img_list):
    fake = io.imread(os.path.join(pred_dir, i + img_format))
    real = io.imread(os.path.join(targ_dir, i + img_format))

    fake = to_tensor(fake).to(device)
    real = to_tensor(real).to(device)

    phv_list = PHV(fake, real)
    all_phv.append(phv_list)
all_phv = np.array(all_phv)
all_phv = np.mean(all_phv, axis=0)
res_str = ''
for layer, value in zip(layers, all_phv):
    res_str += f'{layer}: {value:.4f} '
print(res_str)
print(np.round(all_phv, 4))

#________________________________________________________________________________________#
# FID statistics
device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
num_avail_cpus = len(os.sched_getaffinity(0))
num_workers = min(num_avail_cpus, 8)

real_paths = [os.path.join(targ_dir, f + img_format) for f in img_list]
fake_paths = [os.path.join(pred_dir, f + img_format) for f in img_list]
print(f"Total number of images: {len(real_paths)}")

dims = 2048
block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
model = InceptionV3([block_idx]).to(device)

m1, s1 = calculate_activation_statistics(real_paths, model, batch_size=10, dims=dims,
                                    device=device, num_workers=num_workers)

m2, s2 = calculate_activation_statistics(fake_paths, model, batch_size=10, dims=dims,
                                    device=device, num_workers=num_workers)

fid_value = calculate_frechet_distance(m1, s1, m2, s2)

print(f'FID: {fid_value:.2f}')

#________________________________________________________________________________________#
# KID statistics
command = f'python3 util/kid_score.py --true {targ_dir} --fake {pred_dir}'
os.system(command)

#________________________________________________________________________________________#
# PSNR and SSIM statistics
psnr = []
ssim = []
for i in tqdm(img_list):
    fake = io.imread(os.path.join(pred_dir, i + img_format))
    real = io.imread(os.path.join(targ_dir, i + img_format))
    PSNR = peak_signal_noise_ratio(fake, real)
    psnr.append(PSNR)
    SSIM = structural_similarity(fake, real, channel_axis=-1)
    ssim.append(SSIM)
average_psnr = sum(psnr)/len(psnr)
average_ssim = sum(ssim)/len(ssim)
print(pred_dir)
print("The average psnr is " + str(average_psnr))
print("The average ssim is " + str(average_ssim))
print(f"{average_psnr:.4f} {average_ssim:.4f}")

##########################################################################################
#________________________________________________________________________________________#
# Arshmeet's Additions: 
# Biologically-Informed Metrics
# NDC - nuclei density consistency 
# MSIC - membrane staining intensity consistency
#________________________________________________________________________________________#
##########################################################################################

# load models once.
nuclei_model = models.CellposeModel(gpu=True, pretrained_model='nuclei') # extracts nuceli only 
cyto_model = models.CellposeModel(gpu=True, pretrained_model='cyto2') # exactract full cell 

def helper(ihc_rgb): # extracts information
    # (1) split H (blue, hematotoxylin) and DAB (brown) channels
    ihc_h, ihc_h_rgb, ihc_dab, ihc_dab_rgb = get_ihc_channel(ihc_rgb)
    # (2) Get DAB binary mask: most dark saturated-stained
    ihc_dab_s, ihc_dab_rgb, ihc_dab_mask = get_dab_mask(ihc_rgb)
    # (3) Get nuceli from H channel, get nuclei from DAB channel, combine
    ihc_h_seg, ihc_h_mask, ihc_h_nuclei = get_h_nuclei(ihc_h, nuclei_model)
    ihc_dab_mask, ihc_dab_nuclei = get_dab_nuclei(ihc_dab_mask, cyto_model)
    nuclei = ihc_h_nuclei + ihc_dab_nuclei
    num_nuclei = len(nuclei)
    # (4) build the density map
    map = get_nuclei_map(ihc_rgb, nuclei)
    # return nuclei densty map, nuclei count
    return map, num_nuclei

def compute_bim(fake, real): 
    # get real/fake DAB density maps
    fake_map, fake_count = helper(fake) 
    fake_map = fake_map.astype(np.float64) / 255.0
    real_map, real_count = helper(real) 
    real_map = real_map.astype(np.float64) / 255.0

    # NDC --------------------------------------------------------------------
    # count error
    if real_count == 0 and fake_count == 0: 
        ndc = 1 # both agree, no nuclei
    elif real_count == 0 and fake_count != 0:
        ndc = 0 # model hallucinated nuclei 
    elif real_count != 0 and fake_count == 0: 
        ndc = 0 # model failed to catch any 
    else:
        ndc = 1 - abs(fake_count - real_count) / (real_count + 1e-6) 
        ndc = float(np.clip(ndc, 0, 1))

    # MSIC -------------------------------------------------------------------
    # get saturation for the real and fake 
    fake_dab_s, _, _ = get_dab_mask(fake)
    real_dab_s, _, _ = get_dab_mask(real) 
    # taken from the get_dab_mask.py code 
    # sweep over different saturation thresholds, create maskss
    errors = [] 
    r_reals = [] 
    r_fakes = [] 
    for t in np.arange(0.1, 0.5+0.02, 0.02):
        r_fake = np.mean(fake_dab_s > t) 
        r_real = np.mean(real_dab_s > t) 
        errors.append(abs(r_fake - r_real))
        r_reals.append(r_real) 
        r_fakes.append(r_fake)
    msic = 1 - np.mean(errors) / (np.mean(r_reals) + np.mean(r_fakes) + 1e-6)

    # Spearman -----------------------------------------------------------------
    # compute spearman on flattened 1d
    scorr, _ = spearmanr(fake_map.flatten(), real_map.flatten())
    scorr = float(scorr) if not np.isnan(scorr) else np.nan
    
    return ndc, msic, scorr
    
# BIM - bioligically informed metrics 
ndcs = [] 
msics = [] 
scorrs = [] 

for i in tqdm(img_list_bim):
    fake = io.imread(os.path.join(pred_dir, i + img_format))
    real = io.imread(os.path.join(targ_dir, i + img_format))
    # .ndim tells how many dimensions - 3 for RGB 
    # .shape[2] == 4 means RGBA 
    if fake.ndim == 3 and fake.shape[2] == 4:
        fake = fake[:, :, :3]
    if real.ndim == 3 and real.shape[2] == 4: 
        real = real[:, :, :3]
    ndc, msic, scorr = compute_bim(fake, real)
    print(f"NDC: {ndc:.4f}, MSIC: {msic:.4f}, SCorr: {scorr:.4f}")
    ndcs.append(ndc) 
    msics.append(msic)
    scorrs.append(scorr)
    
print(f"NDC Count: {np.nanmean(ndcs):.4f} +/- {np.nanstd(ndcs):.4f}")
print(f"MSIC: {np.nanmean(msics):.4f} +/- {np.nanstd(msics):.4f}")
print(f"Spearman Correlation: {np.nanmean(scorrs):.4f} +/- {np.nanstd(scorrs):.4f}")