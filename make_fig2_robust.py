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

patch     = '00467_test_1+'
clean_dir = f'{BASE}/dair_bci_lam0_2_on_bci/dair_bci_lam0_2/test_25/images'
hsv_dir   = f'{BASE}/dair_bci_lam0_2_on_bci_hsv/dair_bci_lam0_2/test_25/images'

inputs  = [
    load(f'{clean_dir}/real_A/{patch}.png'),
    load(f'{hsv_dir}/real_A/{patch}.png'),
    load(f'{clean_dir}/real_B/{patch}.png'),
]
outputs = [
    load(f'{clean_dir}/fake_B/{patch}.png'),
    load(f'{hsv_dir}/fake_B/{patch}.png'),
]

input_titles  = ['H&E Input (clean)', 'H&E Input (HSV perturbed)', 'Real IHC (Ground Truth)']
output_titles = ['Model Output (clean)', 'Model Output (HSV perturbed)', '']

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Robustness: dair_bci λ=0.2 — Clean vs. HSV-Perturbed Input',
             fontsize=32, fontweight='bold', y=1.01)

for c, (img, title) in enumerate(zip(inputs, input_titles)):
    ax = axes[0, c]
    ax.imshow(img)
    ax.axis('off')
    ax.set_title(title, fontsize=28, fontweight='bold', pad=10)

for c in range(3):
    ax = axes[1, c]
    if c < 2:
        ax.imshow(outputs[c])
        ax.axis('off')
        ax.set_title(output_titles[c], fontsize=28, fontweight='bold', pad=10)
        ndc = 0.987 if c == 0 else 0.000
        bg  = '#aaffaa' if c == 0 else '#ffaaaa'
        ax.text(0.02, 0.02, f'NDC = {ndc:.2f}',
                transform=ax.transAxes,
                fontsize=36, fontweight='bold',
                verticalalignment='bottom',
                bbox=dict(boxstyle='round,pad=0.5', facecolor=bg, alpha=0.92,
                          edgecolor='black', linewidth=2))
    else:
        ax.axis('off')

plt.tight_layout(h_pad=3, w_pad=2)
plt.savefig('/scratch/users/arshmeet/virtual_stain/figures/fig2_robustness_v2.png',
            dpi=180, bbox_inches='tight')
plt.close()
print('done')
