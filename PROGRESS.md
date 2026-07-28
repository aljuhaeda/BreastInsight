# BreastInsight — Progress

## Status
Stable, research/educational. No live demo — local notebook + trained
model artifact.

## Done
- 3-layer CNN classifying breast ultrasound images (BUSI dataset) into
  normal / benign / malignant.
- Fixed a real data-integrity bug: the original pipeline trained on
  1,578 files from `Dataset_BUSI_with_GT/` that were supposed to be 780
  real images — the other 798 were segmentation masks mixed in as
  ordinary training data, invalidating the reported 87% accuracy claim.
- Fixed a second, independent bug: true/predicted labels for the
  classification report were collected from two separately-shuffled
  dataset iterations, making the notebook's own report self-contradictory
  (46% vs. 70% on the same data).
- Retrained on a masks-excluded `clean_dataset/`; labels now collected
  in one aligned pass.
- Real, honest result reported: 69% validation accuracy, with per-class
  recall broken out (benign 0.91, malignant 0.47, normal 0.14) instead
  of one flattering headline number.
- Fixed a double-normalization bug in the README's inference code
  snippet (model's first layer already rescales 0-255 input).
- README explicitly states "not for clinical use."

## In progress
- Nothing currently active.

## Known issues / honest limitations
- Malignant recall (0.47) is not screening-grade — misses over half of
  malignant cases in validation. Stated plainly in README, not hidden.
- Small, imbalanced dataset (780 images: 437 benign / 210 malignant /
  133 normal) — normal-class recall is only 0.14 as a direct result.
- Not validated on multi-site data or different imaging equipment.
- No live demo app exists for this repo (unlike the other ML projects).

## Verification log
- 2026-07-23: git working tree clean, no pending diff. `/security-review`
  skill checked — N/A, diff-based and nothing to review.
  `BreastInsight.h5` confirmed to have a valid HDF5 file signature (not
  corrupted/truncated); `model.fix/` SavedModel directory structure
  intact (`saved_model.pb`, `variables/`, metadata present).
  **Not verified**: full model load + inference test — no TensorFlow
  environment available locally to run it. Stated honestly rather than
  assumed working; would need `pip install tensorflow` and an actual
  `model.predict()` call to fully confirm.
- 2026-07-29: TensorFlow environment built (`py -3.11` venv,
  `pip install tensorflow numpy pillow scikit-learn`). `BreastInsight.h5`
  loads cleanly; architecture matches the README exactly (3-layer CNN,
  `Rescaling(1./255)` first layer, `224×224×3` input, 3-class output,
  6.4M params).
- 2026-07-29 (full re-verification): downloaded the real BUSI dataset via
  the Kaggle API, rebuilt `clean_dataset/` using the notebook's exact
  mask-filtering logic (780 files: 437 benign / 210 malignant / 133
  normal — matches README exactly), reproduced the notebook's exact
  `image_dataset_from_directory` validation split (`validation_split=0.2,
  seed=123`, 156 validation files), and ran the real trained
  `BreastInsight.h5` through `classification_report` and
  `model.evaluate()`. **Result matches the README exactly**: 69.2%
  accuracy, recall benign 0.91 / malignant 0.47 / normal 0.14. The
  headline accuracy claim is now genuinely, independently reproduced —
  not just internally consistent with the notebook's saved output.
  `Dataset_BUSI_with_GT/` and `clean_dataset/` are gitignored (not
  committed — too large, and reproducible from Kaggle on demand).

## Next up
- Nothing outstanding — the one open verification item is now closed.

## Verification log (continued)
- 2026-07-28: fresh audit found `git log -- model.fix` shows that
  directory was added in the very first commit and never touched by the
  later data-integrity fix commit — it's the broken pre-fix model, not
  an alternate checkpoint as README implied. README now warns against
  using it for inference. Also fixed a filename mismatch in
  `BreastInsight.ipynb`: the model-save cell wrote `uas.h5` (untracked,
  never produced elsewhere in the notebook) while the load-model demo
  cell a few cells later loaded `BreastInsight.h5` — running the
  notebook top-to-bottom as the README instructs never actually
  exercised the just-trained model. Save cell now writes
  `BreastInsight.h5` to match.
