# Project Contributions — Detailed Breakdown
*(Claude-generated from session transcripts. Source: `/Users/arshmeetkaur/Desktop/05-30.md` and `/Users/arshmeetkaur/Desktop/06-05.md`)*

---

## 1. What the Base Code Already Had

### ASP — Adaptive Supervised PatchNCE (Lin et al., MICCAI 2023)
The entire `AdaptiveSupervisedPatchNCE/` directory is upstream baseline code, kept unmodified for reference. Our working codebase is `AdaptiveSupervisedPatchNCE_DAIR/`, which was forked from it.

Pre-existing components in the baseline:
- **`models/cpt_model.py`** — CPTModel implementing the full training loop with: GAN loss (`loss_G_GAN`, `loss_D`), PatchNCE contrastive loss (`loss_NCE`), Gaussian Pyramid perceptual loss (`loss_GP`), and Adaptive Supervised PatchNCE loss (`loss_ASP`). The `optimize_parameters()` method and all loss weighting (`lambda_GAN`, `lambda_NCE`, `lambda_asp`) were pre-existing.
- **`models/networks.py`** — Generator (ResNet 6-block), NLayer discriminator, MLP feature projector, all network definitions
- **`train.py` / `test.py`** — Full training and inference entry points, argument parsing, epoch loop, checkpoint saving (`save_epoch_freq=5`)
- **`experiments/mist_launcher.py`** — Training defaults: `n_epochs=30`, `n_epochs_decay=10`, `netG=resnet_6blocks`, `lambda_GAN=1.0`, `lambda_NCE=10.0`, `lambda_gp=10.0`, `lambda_asp=10.0`, `asp_loss_mode='lambda_linear'`, `batch_size=1`, `load_size=1024`, `crop_size=512`
- **Dataset loaders, image utilities, visualizer** — all pre-existing

### TDKstain (Peng et al.)
Pre-existing in `TDKstain/preprocess/`:
- **`get_ihc_channel(img)`** — Vahadane color deconvolution separating H&E stain channels from IHC RGB input
- **`get_h_nuclei(h_conc)`** — Hematoxylin channel nuclei extraction using Otsu threshold + morphological cleanup
- **`get_dab_nuclei(dab_conc)`** — DAB channel nuclei extraction (same pipeline, different channel)
- **`get_nuclei_map(labeled)`** — Spatial nuclei density map generation (Gaussian-smoothed centroid map for Spearman correlation)
- **`get_dab_mask(dab_conc, threshold)`** — Threshold-sweep DAB staining mask for MSIC computation

We borrowed these utilities entirely; our contribution was composing them into scalar BIM metrics and adding DAIR-SQ.

---

## 2. What We Implemented (User + Claude)

### DAIR-SQ Loss (`AdaptiveSupervisedPatchNCE_DAIR/models/cpt_model.py`)
Added to `optimize_parameters()` in CPTModel:
- **HSV perturbation augmentation**: For each training batch, a perturbed copy of the input H&E is generated with random hue shift (±0.0833), saturation scale (×[0.75, 1.25]), and value scale (×[0.95, 1.05])
- **DAIR-SQ regularizer**: `loss_DAIR = (√L_orig − √L_pert)²` where `L_orig` and `L_pert` are the generator's total losses on the original and perturbed inputs respectively. Applied only to the generator.
- **λ hyperparameter**: `lambda_dair` controls regularization strength; ablated over {0.2, 1, 5, 20, 50, 100}

### BIM Evaluation Pipeline (`evaluate.py`)
Entirely new file. Key components:
- **`compute_ndc(real_ihc, fake_ihc)`** — Calls TDKstain's `get_h_nuclei` on both images, counts labeled nuclei regions (`N_real`, `N_fake`), returns `min(N_real, N_fake) / max(N_real, N_fake)`. Filters nuclei by area (80–8000 px) and circularity.
- **`compute_msic(real_ihc, fake_ihc)`** — Runs a threshold sweep (0–255 in steps of 5) on both DAB masks from `get_dab_mask`, computes Pearson correlation between the two threshold-vs-positive-area curves
- **`compute_sdc(real_ihc, fake_ihc)`** — Generates spatial nuclei density maps via `get_nuclei_map`, computes Spearman rank correlation between real and virtual density maps
- **`compute_bim(real_ihc, fake_ihc)`** — Wrapper returning `{ndc, msic, sdc}` for a single patch pair
- **`helper(results_dir)`** — Main evaluation loop: uses `random.seed(0)` shuffle to iterate over patches in a fixed order (matching SLURM job output order), calls `compute_bim` per patch, writes per-image scores and dataset-level means to log

### `her2_classifier.py` — BIM Biological Validation
New file. Trains a ResNet18 HER2 scoring classifier (0/1+/2+/3+) on real BCI IHC patches, then evaluates it on virtual IHC from each model. Used as a biological proxy: if a model's virtual IHC passes the classifier at the same rate as real IHC, the stain is biologically plausible. Submitted as SLURM job 27777913.

### SLURM Infrastructure
- All SLURM scripts to run code on server written by Claude.

---

## 3. Our Contributions

### Conceptual and Intellectual

**Problem definition**
- Identified that existing virtual staining work evaluates only in-distribution performance and ignores cross-dataset and robustness failure modes
- Framed the paper around two evaluation axes: generalizability (train on X, test on Y) and robustness (test under image perturbations)
- Chose the specific perturbation type: HSV shifts, motivated by the fact that scanner color profiles vary across institutions — HSV perturbation is a realistic proxy for acquisition variability

**DAIR-SQ concept**
- Proposed the idea of training with augmented inputs and penalizing output instability as the mechanism for learning robustness
- Decided against adding Peng et al.'s auxiliary losses (`L_nuclei`, `L_membrane`) to the training objective — kept DAIR-SQ as the sole training-time contribution and used BIM purely for evaluation

**BIM concept**
- Originated the idea that evaluation should use biologically meaningful signals (nuclei counts, membrane staining) rather than pixel-level statistics
- Proposed the NDC + MSIC two-component structure explicitly before Claude had suggested either component. 
- Named NDC (Nuclei Density Consistency); directed the choice of Spearman correlation for spatial structure (SDC)
- Identified TDKstain as the right preprocessing foundation

**Experimental design**
- Chose BCI and MIST as the two datasets for cross-dataset evaluation
- Designed the λ ablation sweep {0.2, 1, 5, 20, 50, 100}
- Directed HER2 classifier validation as a biological sanity check

### Execution and Oversight

- Managed all SLURM job submissions: ~12 training jobs, ~20 inference jobs, BIM evaluation jobs
- Monitored training logs, caught the early stopping misconfiguration that was killing jobs prematurely at epoch 15, and directed retraining
- Identified the NaN bug in BIM evaluation and directed debugging
- Reconnected SSH ControlMaster sessions each time the Sherlock connection dropped (required multiple times throughout both sessions)
- Directed all figure iteration: layout, font sizes, panel order, label content, color choices
- Laid out the full paper and poster in Overleaf; pasted and integrated all LaTeX

### Analysis

- Interpreted all result tables and identified key findings:
  - λ=0.2 is the optimal DAIR-SQ strength — lower λ means less regularization, but too much hurts in-distribution performance
  - DAIR-SQ improves robustness (NDC under HSV) while maintaining near-baseline generalizability
  - Cross-dataset failure traces to color deconvolution breakdown under out-of-distribution color spaces
  - Robustness failure = color shortcut learning, not morphological understanding
- Chose patch `2M2103108_13_23` framing for the generalizability figure and directed the "worst model vs. best in-distribution" comparison structure
- Chose patch `00467_test_1+` for the robustness figure (NDC 0.987→0.000 under HSV)

---

## 4. Claude's Contributions

### Code

**DAIR-SQ implementation**
- Wrote the HSV augmentation function and `loss_DAIR` computation in `cpt_model.py`
- Wired λ into the training loop and model `__init__`

**BIM metric formalization and implementation**
- Formalized NDC, MSIC, SDC as concrete equations from the user's concept
- Named MSIC (Membrane Staining Intensity Consistency) and SDC (Spearman Density Correlation)
- Wrote the complete `compute_ndc`, `compute_msic`, `compute_sdc`, `compute_bim`, and `helper` functions in `evaluate.py`
- Fixed `get_dab_mask.py` after it was corrupted by earlier edits — rewrote from scratch using the original logic

**Infrastructure**
- Wrote all SLURM training, inference, and evaluation scripts
- Wrote `her2_classifier.py` with ResNet18 fine-tuning and BIM-score injection
- Handled all file transfer via hex-encoding workaround (`xxd -p` → `binascii.unhexlify`) when SCP/stdin piping failed

**Results extraction**
- Queried Sherlock log files using `strings` + `grep`/`awk` to extract all metric values after garbled log format (no newlines, mixed SLURM stdout)
- Reconstructed `random.seed(0)` shuffle order from `evaluate.py` to map per-image BIM scores back to patch filenames
- Identified worst-performing patches for both qualitative figures

### Tables
All tables were built by Claude from raw log extractions. Iterative formatting was done collaboratively:

| Table | Content | Notes |
|---|---|---|
| BIM Validation | NDC/MSIC/SDC for ASP-BCI and ASP-MIST vs. real IHC, plus HER2 classifier F1 | Directional bolding: FID lower=better, all others higher=better |
| λ Ablation | FID/PSNR/SSIM/PHV/NDC/MSIC/SDC for all 6 DAIR-MIST λ values + ASP baseline | Δ% change rows; bold = best per column |
| Generalizability | Cross-dataset performance: BCI models on MIST, MIST models on BCI, in-distribution baselines | Structured to show cross-dataset penalty |
| Robustness | Clean vs. HSV-perturbed NDC/MSIC/SDC for DAIR and ASP models | Shows DAIR-SQ's robustness benefit |
| Data Splits | BCI/MIST train/test patch counts from Sherlock `ls \| wc -l` | BCI: 3,896 train / 977 test; MIST: 4,642 train / 1,000 val |

Table formatting went through multiple LaTeX iterations: removing `\usepackage{multirow}` and `\usepackage{booktabs}` to match poster's plain `\hline` style, adjusting `\colwidth` and `fontsize` to fit the Gemini poster column width, removing path column from data splits table.

### Figures
All figure generation scripts were written by Claude, uploaded via hex-encoding (`xxd -p` pipeline), and run on Sherlock in `bim_env` (Python 3.9, skimage 0.24, matplotlib). Results pulled to Desktop via base64.

**`fig1_generalizability_v6.png`** (`make_fig1_v6.py`)
- 4 panels stacked vertically: H&E input → Real IHC → `asp_bci` on MIST (cross-dataset, NDC=0.00) → `asp_mist` on MIST (in-distribution, NDC=0.64)
- Patch: `2M2103108_13_23` — identified as worst by reconstructing `random.seed(0)` shuffle and ranking `dair_bci_lam0_2`'s per-patch NDC scores
- NDC labels on each panel; large font; tight layout

**`fig2_robustness_v2.png`** (`make_fig2_robust.py`)
- 4 panels: H&E (clean) → H&E (HSV-perturbed) → model output on clean (NDC=0.99) → model output on perturbed (NDC=0.00)
- Patch: `00467_test_1+` — identified as largest NDC drop (0.987→0.000) across BCI test set
- Shows visually identical inputs producing radically different outputs

**`fig_ndc.png`** (`make_fig_ndc.py`)
- Explanation figure for NDC calculation on a single patch
- 4 columns × 2 rows: Virtual IHC / Real IHC × IHC Patch / H Channel / DAB Channel / Nuclei Mask
- Arrows connecting columns; count boxes (N_virt, N_real) feeding into NDC result box
- Iterated multiple times: fixed overlapping arrows (MSIC version), adjusted box sizing, font scaling (base FS=42)
- User later asked to remove the NDC formula from the figure and increase font sizes

**`fig_msic.png`** (`make_fig_msic.py`)
- SDC visualization showing spatial nuclei density maps (Gaussian-smoothed centroid heatmaps) for real vs. virtual IHC
- Scatter plot with Spearman ρ annotated in result box

**`fig_sdc.png`** (`make_fig_sdc.py`)
- SDC scatter: real density map pixel values vs. virtual density map pixel values
- Spearman ρ and p-value annotated; result box placed inside scatter plot area (fixed in later iteration to avoid overlap with arrows)

**`fig_data_grid.png`** (`make_fig_data_grid.py`)
- 2×2 grid: BCI H&E (top-left), BCI IHC (top-right), MIST H&E (bottom-left), MIST IHC (bottom-right)
- Patches: `00000_train_1+` (BCI) and `100M2004069_10_12` (MIST)
- Iterated: started as 2×2, temporarily tried 1×4 horizontal strip, reverted to 2×2 square
- Used for poster Data block

**`fig_mode_workflow2.png`** (`make_fig_workflow2.py`)
- Pipeline workflow diagram showing training mode vs. inference mode
- Training: H&E → Generator → Virtual IHC, with DAIR-SQ branch (HSV perturb → Generator → compare)
- Inference: H&E → Generator → Virtual IHC only
- Based on a reference image the user provided; Claude reproduced it in matplotlib with `FancyBboxPatch` and annotated arrows
- Iterated: user asked for all text much larger; then asked for "HSV Shift" font specifically smaller

---

## 5. Source Transcripts

| File | Dates | Lines | Content |
|---|---|---|---|
| `05-30.md` | May 30 – June 4, 2026 | ~59,900 | Model training, BIM development, evaluation pipeline, result extraction |
| `06-05.md` | June 5 – June 9, 2026 | ~6,975 | Result tables, qualitative figures, poster figures, poster layout |