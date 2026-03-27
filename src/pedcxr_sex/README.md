# main_blending.py

小児胸部X線（CXR）の性別分類に対して、以下を比較・統合するための実験スクリプトです。

- **CNN 単独**
- **CNN embedding + Logistic Regression**
- **Radiomics + Logistic Regression**
- **Hybrid model**
  - **blend**: CNN / embedding / radiomics の確率平均
  - **stack**: CNN / embedding / radiomics を入力としたメタ分類器

本スクリプトでは、患者IDベースでデータリークを防ぎながら train / val / test を分割し、  
**overall test AUC** に加えて **年齢ビン別 AUC**、**macro age-bin AUC** を算出します。  
また、embedding や test probability を `.npy` として保存し、後続解析にも使えるようにしています。 :contentReference[oaicite:1]{index=1}

---

## 1. Features

### 1.1 Main functions

- 患者IDベースの **StratifiedGroupKFold** による split
- 画像前処理
  - percentile clipping
  - CLAHE
  - unsharp masking
- 複数CNN backboneに対応
- **ResNet18 embedding** の抽出
- embedding に対する **Logistic Regression**
- radiomics CSV を読み込んで **Radiomics LR**
- hybrid 推論
  - `blend`
  - `stack`
- 評価指標
  - validation AUC
  - test AUC (all)
  - test AUC by age bins
  - macro AUC across age bins
- 各種 `.csv` / `.npy` 保存

画像前処理では、各画像に対して **1–99 percentile clipping → 正規化 → CLAHE → Gaussian blur を用いたシャープ化** が行われます。 :contentReference[oaicite:2]{index=2}

---

## 2. Data format

このスクリプトでは、基本的に以下の4要素を想定しています。

- `images`: `(N, H, W)` の grayscale 画像配列
- `ages`: 年齢
- `sex`: 性別ラベル
- `pids`: 患者ID

内部では `sample_id = np.arange(len(images)).astype(str)` が使われ、  
radiomics 側との対応付けにもこの **画像単位ID** が利用されます。  
そのため、radiomics CSV 側の ID 列は、この `sample_id` と一致している必要があります。 :contentReference[oaicite:3]{index=3}

### Sex label
`sex_to_int()` では以下のように変換されます。

- male / m / 1 / true → `0`
- female / f / 0 / false → `1`

未知ラベルはエラーになります。 :contentReference[oaicite:4]{index=4}

---

## 3. Train / Val / Test split

分割は **患者IDベース** に行われます。

- まず `StratifiedGroupKFold(n_splits=5)` で test を分離
- 残りの trainval を `StratifiedGroupKFold(n_splits=8)` で train / val に分離

コメント上は **「7:1:2 相当」** の設計です。  
stratification には、**年齢ビン + 性別** を用いた strata が使われています。  
これにより、年齢層と性別の分布をある程度保ちながら、患者リークを避けた評価ができます。 :contentReference[oaicite:5]{index=5}

---

## 4. Supported models

### torchvision models
- `resnet18`
- `resnet50`
- `resnet101`
- `densenet121`
- `convnext_tiny`
- `vit_b_16`
- `swin_t`

### timm models
上記以外のモデル名は `timm.create_model(..., pretrained=True)` で読み込まれます。 :contentReference[oaicite:6]{index=6}

> **Note:**  
> embedding 抽出は現時点で **ResNet18 のみ** 対応です。  
> `hybrid_mode != "none"` または `radio_csv` を使う場合、`resnet18` を使用してください。 :contentReference[oaicite:7]{index=7}

---

## 5. Hybrid learning modes

### `hybrid_mode="none"`
CNN 単独で学習・評価します。

### `hybrid_mode="blend"`
以下を確率レベルで加重平均します。

- CNN probability
- embedding LR probability
- radiomics LR probability（指定時）

radiomics なしの場合は、CNN + embedding の2系統 blend にも対応します。 :contentReference[oaicite:8]{index=8}

### `hybrid_mode="stack"`
以下を入力にしたメタ分類器で最終予測を行います。

- CNN probability
- embedding LR probability
- radiomics LR probability

---

## 6. Radiomics input

radiomics を使う場合、CSV を読み込みます。

想定仕様:

- CSV に ID 列が必要
- デフォルト ID 列名は `sample_id`
- 数値列のみ使用
- 以下の不要列は自動除外
  - `Unnamed: 0`
  - `sex`
  - `age`
  - `patient_id`

指定した ID が radiomics CSV 側に存在しない場合はエラーになります。 :contentReference[oaicite:9]{index=9}

---

## 7. Outputs

各モデルごとに以下が保存されます。

### CSV
- `results_<experiment_name>.csv`

主な列:

- `model`
- `val_auc_best`
- `test_auc_all`
- `test_auc_macro_age`
- `auc_embed_val`
- `auc_embed_test`
- `auc_radio_val`
- `auc_radio_test`
- `auc_hybrid_test`
- `auc_hybrid_macro_age`
- 学習時間 / 推論時間
- 年齢ビン別 hybrid AUC（ある場合） :contentReference[oaicite:10]{index=10}

### NPY
条件に応じて以下が保存されます。

- `emb_train_*.npy`
- `emb_val_*.npy`
- `emb_test_*.npy`
- `prob_val_embed_*.npy`
- `prob_test_embed_*.npy`
- `prob_val_radio_*.npy`
- `prob_test_radio_*.npy`
- `prob_test_blend_*.npy` or `prob_test_stack_*.npy`
- `test_prob_*.npy` :contentReference[oaicite:11]{index=11}

---

## 8. Age-bin evaluation

デフォルトの年齢ビンは以下です。

- `0–4`
- `5–9`
- `10–14`
- `15–18`

各ビンで AUC を計算し、その平均を **macro age-bin AUC** として出力します。  
小サンプルや片側クラスのみの場合は `NaN` になることがあります。 :contentReference[oaicite:12]{index=12}

---

## 9. Reproducibility

再現性確保のため、以下が実装されています。

- `random`, `numpy`, `torch` seed 固定
- `torch.cuda.manual_seed_all`
- `worker_init_fn`
- `DataLoader generator`
- optional deterministic mode
- `CUBLAS_WORKSPACE_CONFIG=":4096:8"` 設定

ただし、GPU / CUDA / AMP 使用時には完全一致が保証されない場合があります。 :contentReference[oaicite:13]{index=13}

---

## 10. Example workflow

### A. CNN only
1. CXR を読み込む
2. 前処理
3. train / val / test split
4. CNN を学習
5. overall AUC / age-bin AUC を評価

### B. CNN + embedding
1. ResNet18 を学習
2. penultimate layer の embedding を抽出
3. embedding に Logistic Regression を適用
4. CNN 単独 vs embedding の性能を比較

### C. CNN + embedding + radiomics
1. radiomics CSV を `sample_id` で対応付け
2. radiomics LR を学習
3. `blend` または `stack` で統合
4. hybrid AUC を評価

---

## 11. Practical notes

### 11.1 Embedding extraction limitation
embedding は **ResNet18 限定** です。  
他 backbone を使う場合、現状では hybrid 機能はそのままでは使えません。 :contentReference[oaicite:14]{index=14}

### 11.2 ID alignment is critical
radiomics と CNN 側の対応は **sample_id ベース** なので、  
CSV 側の行順ではなく **ID一致** が重要です。  
ここがずれると hybrid 性能は無意味になります。 :contentReference[oaicite:15]{index=15}

### 11.3 Patient-level split
train / val / test は **patient_id ベース** で分割されます。  
画像単位で分けると leakage の原因になるので注意してください。 :contentReference[oaicite:16]{index=16}

---

## 12. Example command template

実際の引数定義は手元のスクリプトに合わせて確認してください。  
概念的には以下のような実行を想定しています。

```bash
python src/pedcxr_sex/main_blending.py \
  --img_npy /path/to/images.npy \
  --age_npy /path/to/ages.npy \
  --sex_npy /path/to/sex.npy \
  --pid_npy /path/to/pids.npy \
  --models resnet18 \
  --epochs 10 \
  --batch 32 \
  --workers 2 \
  --seed 42 \
  --hybrid_mode blend \
  --radio_csv /path/to/radiomics.csv \
  --radio_id_col sample_id \
  --save_root checkpoints_compare
