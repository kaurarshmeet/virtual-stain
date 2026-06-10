import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage import io

BASE = '/scratch/users/arshmeet/virtual_stain/results'

def load(path):
    img = io.imread(path)
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]
    return img

# ── Figure 1: Generalizability ──────────────────────────────────────────────
# Worst model: dair_bci_lam0_2 on MIST
# Compare against: asp_mist on MIST (best in-dist)
worst_img = '2M2103108_13_23'
best_img  = '81M2100356_25_26'

gen_dir   = f'{BASE}/dair_bci_lam0_2_on_mist/dair_bci_lam0_2/val_25/images'
ref_dir   = f'{BASE}/asp_mist_on_mist/asp_mist/val_25/images'

rows = [
    ('Worst (NDC=0.00, MSIC=0.03, SDC=0.04)', worst_img),
    ('Best  (NDC=0.99, MSIC=0.87, SDC=0.46)', best_img),
]
cols = ['H\&E Input', 'Real IHC', 'dair\_bci λ=0.2\n(worst model)', 'asp\_mist\n(best model)']

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle('Figure 1: Generalizability — dair_bci_lam0_2 on MIST', fontsize=13, fontweight='bold')

for r, (label, name) in enumerate(rows):
    imgs = [
        load(f'{gen_dir}/real_A/{name}.png'),
        load(f'{gen_dir}/real_B/{name}.png'),
        load(f'{gen_dir}/fake_B/{name}.png'),
        load(f'{ref_dir}/fake_B/{name}.png'),
    ]
    for c, img in enumerate(imgs):
        ax = axes[r, c]
        ax.imshow(img)
        ax.axis('off')
        if r == 0:
            ax.set_title(cols[c], fontsize=10, fontweight='bold')
    axes[r, 0].set_ylabel(label, fontsize=9, labelpad=8)

plt.tight_layout()
plt.savefig('/scratch/users/arshmeet/virtual_stain/figures/fig1_generalizability.png', dpi=150, bbox_inches='tight')
plt.close()
print('Figure 1 saved.')

# ── Figure 2: Robustness ─────────────────────────────────────────────────────
# Model: dair_bci_lam0_2, clean BCI vs HSV BCI
worst_img_r = '00459_test_1+'
best_img_r  = '00687_test_3+'

clean_dir = f'{BASE}/dair_bci_lam0_2_on_bci/dair_bci_lam0_2/test_25/images'
hsv_dir   = f'{BASE}/dair_bci_lam0_2_on_bci_hsv/dair_bci_lam0_2/test_25/images'

rows2 = [
    ('Worst (NDC=0.00, MSIC=0.06, SDC=-0.28)', worst_img_r),
    ('Best  (NDC=1.00, MSIC=0.68, SDC=0.67)',  best_img_r),
]
cols2 = ['H\&E (clean)', 'H\&E (HSV perturbed)', 'Real IHC', 'Output (clean)', 'Output (HSV)']

fig2, axes2 = plt.subplots(2, 5, figsize=(20, 8))
fig2.suptitle('Figure 2: Robustness — dair_bci_lam0_2 on BCI (clean vs HSV perturbed)', fontsize=13, fontweight='bold')

for r, (label, name) in enumerate(rows2):
    imgs = [
        load(f'{clean_dir}/real_A/{name}.png'),
        load(f'{hsv_dir}/real_A/{name}.png'),
        load(f'{clean_dir}/real_B/{name}.png'),
        load(f'{clean_dir}/fake_B/{name}.png'),
        load(f'{hsv_dir}/fake_B/{name}.png'),
    ]
    for c, img in enumerate(imgs):
        ax = axes2[r, c]
        ax.imshow(img)
        ax.axis('off')
        if r == 0:
            ax.set_title(cols2[c], fontsize=10, fontweight='bold')
    axes2[r, 0].set_ylabel(label, fontsize=9, labelpad=8)

plt.tight_layout()
plt.savefig('/scratch/users/arshmeet/virtual_stain/figures/fig2_robustness.png', dpi=150, bbox_inches='tight')
plt.close()
print('Figure 2 saved.')
