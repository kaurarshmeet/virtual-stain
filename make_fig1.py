import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage import io

BASE = '/scratch/users/arshmeet/virtual_stain/results'

def load(path):
    img = io.imread(path)
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]
    return img

patch    = '2M2103108_13_23'
bci_dir  = f'{BASE}/asp_bci_on_mist/asp_bci/val_25/images'
mist_dir = f'{BASE}/asp_mist_on_mist/asp_mist/val_25/images'

imgs = [
    load(f'{bci_dir}/real_A/{patch}.png'),
    load(f'{bci_dir}/real_B/{patch}.png'),
    load(f'{bci_dir}/fake_B/{patch}.png'),
    load(f'{mist_dir}/fake_B/{patch}.png'),
]

titles = [
    'H&E Input',
    'Real IHC (Ground Truth)',
    'asp_bci on MIST  (cross-dataset)',
    'asp_mist on MIST  (in-distribution)',
]

scores = [None, None, (0.000, 0.039, 0.009), (0.635, 0.116, 0.105)]

fig, axes = plt.subplots(4, 1, figsize=(10, 36))

for i, (ax, img, title, sc) in enumerate(zip(axes, imgs, titles, scores)):
    ax.imshow(img)
    ax.axis('off')
    ax.set_title(title, fontsize=38, fontweight='bold', pad=16)

    if sc is not None:
        label = f'NDC = {sc[0]:.2f}\nMSIC = {sc[1]:.2f}\nSDC  = {sc[2]:.2f}'
        bg = '#ffaaaa' if i == 2 else '#aaffaa'
        ax.text(0.02, 0.02, label,
                transform=ax.transAxes,
                fontsize=36, fontweight='bold',
                verticalalignment='bottom',
                linespacing=1.6,
                bbox=dict(boxstyle='round,pad=0.6', facecolor=bg, alpha=0.92, edgecolor='black', linewidth=2))

plt.tight_layout(h_pad=4)
plt.savefig('/scratch/users/arshmeet/virtual_stain/figures/fig1_generalizability_v6.png',
            dpi=180, bbox_inches='tight')
plt.close()
print('done')
