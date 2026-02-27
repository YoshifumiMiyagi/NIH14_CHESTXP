# NIH14_CHESTXP

Pediatric CXR sex classification with optional preprocessing and thorax cropping using lung masks.  
Supports multiple-seed evaluation and deterministic mode.

---

## Setup

```bash
git clone https://github.com/YoshifumiMiyagi/NIH14_CHESTXP.git
cd NIH14_CHESTXP
```

### Check CLI options

```bash
python -m src.pedcxr_sex.main --help
```

---

## Data format

### `--img_npy` (required)

A numpy dict saved as `.npy` with keys:

- `images`: `(N, 512, 512)` (uint8/float32)
- `ages`: `(N,)`
- `patient_id`: `(N,)`
- `sex`: `(N,)` (binary label)

### `--lung_mask_npy` (optional, for `--do_crop`)

A numpy array saved as `.npy`:

- `masks`: `(N, 512, 512)` (uint8, 0/1)

---

## Quick Run (Multiple Seeds)

> Replace `<your_path>` with your actual path.

Set paths:

- **IMG**: image dict npy
- **MASK**: lung mask npy
- **OUT_ROOT**: output root directory

```bash
IMG="<your_path>/cxr_pediatric_images512_age_under18_patients_sex_3400.npy"
MASK="<your_path>/lung_mask_3400.npy"
OUT_ROOT="<your_path>/checkpoints_compare"

MODELS="resnet18"
SEEDS="42,43,44,45,46"
EPOCHS=10
BATCH=32
WORKERS=2
```

---

## Experiments

### 1) BASELINE

```bash
echo "===== BASELINE ====="
python -m src.pedcxr_sex.main \
  --img_npy "$IMG" \
  --out_dir "${OUT_ROOT}/runs_baseline" \
  --models "$MODELS" \
  --epochs $EPOCHS \
  --batch $BATCH \
  --workers $WORKERS \
  --use_amp \
  --seeds "$SEEDS" \
  --deterministic
```

### 2) PREPROCESS (CLAHE + enhance)

```bash
echo "===== PREPROCESS ====="
python -m src.pedcxr_sex.main \
  --img_npy "$IMG" \
  --out_dir "${OUT_ROOT}/runs_preprocess" \
  --models "$MODELS" \
  --epochs $EPOCHS \
  --batch $BATCH \
  --workers $WORKERS \
  --use_amp \
  --seeds "$SEEDS" \
  --deterministic \
  --do_preprocess
```

### 3) CROP (thorax crop using lung masks)

```bash
echo "===== CROP ====="
python -m src.pedcxr_sex.main \
  --img_npy "$IMG" \
  --out_dir "${OUT_ROOT}/runs_crop" \
  --models "$MODELS" \
  --epochs $EPOCHS \
  --batch $BATCH \
  --workers $WORKERS \
  --use_amp \
  --seeds "$SEEDS" \
  --deterministic \
  --do_crop \
  --lung_mask_npy "$MASK"
```

### 4) PREPROCESS + CROP

```bash
echo "===== PREPROCESS + CROP ====="
python -m src.pedcxr_sex.main \
  --img_npy "$IMG" \
  --out_dir "${OUT_ROOT}/runs_preprocess_crop" \
  --models "$MODELS" \
  --epochs $EPOCHS \
  --batch $BATCH \
  --workers $WORKERS \
  --use_amp \
  --seeds "$SEEDS" \
  --deterministic \
  --do_preprocess \
  --do_crop \
  --lung_mask_npy "$MASK"
```

---

## Outputs

Each run writes to its own directory, e.g. `${OUT_ROOT}/runs_crop/`.

Typical outputs:

- model checkpoint(s): `*_best*.pt` (or similar)
- metrics summary: `results.csv` (or similar)
- optional predictions: `test_probs.npy` (if enabled in code)

> Tip: separating `--out_dir` per experiment avoids overwriting checkpoints/results.

---

## Notes on reproducibility

Even with fixed seeds, deep learning can show small variations due to CUDA/AMP and dataloader nondeterminism.  
Use `--deterministic` and report final performance as **mean ± SD across multiple seeds**.
