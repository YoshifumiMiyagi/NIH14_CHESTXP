# main_pseudo_retrain.py

`main_pseudo.py` は、小児胸部X線画像を用いた性別二値分類のための学習スクリプトです。
主解析は **Baseline** と **Baseline + Pseudo Retrain** の比較で、pseudo-label を後段で微調整するのではなく、`train + pseudo` を用いて**最初から再学習**する設計をサポートします。

この README は、スクリプトの目的、入出力、主要オプション、実行例、結果の見方を簡潔にまとめたものです。

---

## 1. 目的

本スクリプトの目的は、以下を評価することです。

* 小児CXRから性別を二値分類できるか
* 年齢群別の性能がどう異なるか
* pseudo-label を追加学習データとして用いたとき、性能が改善するか
* その改善が安定して再現するか

特に本版では、pseudo-label の効果をより解釈しやすくするため、**fine-tuning ではなく retrain** を主眼に置いています。

---

## 2. 特徴

* `StratifiedGroupKFold` による **patient-level split**
* `train / val / test` の分割
* 年齢 bin ごとの AUC 評価
* optional な画像前処理（CLAHE + enhance）
* optional な肺野マスクに基づく thorax crop
* external pseudo pool の読込
* pseudo pool と main dataset の patient overlap 除去
* `Baseline`
* `Baseline + Pseudo Retrain`
* multi-seed 実行と mean ± SD 集計

---

## 3. 入力データ形式

### 3.1 main dataset (`--img_npy`)

`np.load(..., allow_pickle=True).item()` で読める辞書形式を想定しています。

必要キー:

* `images` : `(N, H, W)` の画像配列
* `ages` : 年齢配列
* `patient_id` : 患者ID配列
* `sex` : 性別配列 (`M/F`, `male/female` など)

例:

```python
{
    "images": np.ndarray,      # shape = (N, 512, 512)
    "ages": np.ndarray,
    "patient_id": np.ndarray,
    "sex": np.ndarray,
}
```

### 3.2 pseudo pool (`--pseudo_img_npy`)

pseudo-label 用の外部データです。形式は main dataset と同じです。

必要キー:

* `images`
* `ages`
* `patient_id`
* `sex`

`sex` は pseudo-label 自体には不要ですが、`pseudo_acc_vs_gt` を確認したい場合に使います。

---

## 4. データ分割

本スクリプトでは、患者単位のリークを避けるため、`patient_id` を group として用いています。

* まず `StratifiedGroupKFold(n_splits=test_splits)` で `trainval / test` に分割
* 次に `trainval` を `StratifiedGroupKFold(n_splits=val_splits)` で `train / val` に分割

つまり、**train / val / test は patient-level non-overlap** を前提としています。

さらに pseudo pool を用いる場合、main dataset と pseudo pool の patient overlap を除去してから学習します。

ログには以下が出力されます。

```text
[Leakage Check]
pseudo ∩ train = ...
pseudo ∩ val   = ...
pseudo ∩ test  = ...
```

---

## 5. 学習モード

### 5.1 Baseline

main dataset の `train` のみを使って学習し、best validation AUC の重みで test を評価します。

### 5.2 Baseline + Pseudo Retrain

1. Baseline モデルで pseudo pool に対する予測確率を算出
2. 高信頼サンプルのみを pseudo-label として選択
3. `train_ds + pseudo_ds` を結合
4. **新しいモデルを初期化して最初から再学習**
5. best validation AUC の重みで test を評価

この設計により、pseudo-label の効果を「追加データとしての効果」として解釈しやすくしています。

---

## 6. pseudo-label 選択ロジック

pseudo pool から、以下の条件を満たす高信頼サンプルのみを使います。

* `prob >= thr_pos`
* `prob <= thr_neg`

デフォルトでは:

* `thr_pos = 0.995`
* `thr_neg = 0.005`
* `max_n = 300`

さらに pseudo sample には通常より小さい重みを付けられます。

* `pseudo_weight = 0.25` など

これにより、誤pseudo-labelの影響を抑えます。

---

## 7. 対応モデル

torchvision モデル:

* `resnet18`
* `resnet50`
* `resnet101`
* `densenet121`
* `convnext_tiny`
* `vit_b_16`
* `swin_t`

上記以外は `timm.create_model()` に渡されます。

---

## 8. 主な引数

### 必須

* `--img_npy` : main dataset の `.npy`

### 出力

* `--out_dir` : 結果保存先

### モデル学習

* `--models` : カンマ区切りモデル名
* `--epochs` : Baseline 学習 epoch
* `--batch` : batch size
* `--workers` : DataLoader workers
* `--seed` : 単一 seed
* `--seeds` : 複数 seed（カンマ区切り）
* `--use_amp` : mixed precision
* `--deterministic` : 再現性優先モード

### 前処理 / crop

* `--do_preprocess` : CLAHE + enhance
* `--do_crop` : thorax crop
* `--lung_mask_npy` : crop 用 lung mask

### pseudo-label

* `--pseudo_enable`
* `--pseudo_img_npy`
* `--pseudo_weight`
* `--pseudo_thr_pos`
* `--pseudo_thr_neg`
* `--pseudo_max_n`
* `--pseudo_batch`
* `--pseudo_retrain` : pseudo retrain を有効化
* `--pseudo_retrain_epochs` : retrain epoch 数

---

## 9. 実行例

### 9.1 Baseline

```bash
python main_pseudo.py \
  --img_npy /path/to/main_dataset.npy \
  --out_dir /path/to/output \
  --models resnet18 \
  --epochs 10 \
  --batch 32 \
  --workers 2 \
  --seeds 42,43,44 \
  --use_amp
```

### 9.2 Baseline + Pseudo Retrain

```bash
python main_pseudo.py \
  --img_npy /path/to/main_dataset.npy \
  --pseudo_enable \
  --pseudo_img_npy /path/to/pseudo_pool.npy \
  --pseudo_retrain \
  --pseudo_retrain_epochs 10 \
  --out_dir /path/to/output \
  --models resnet18 \
  --epochs 10 \
  --batch 32 \
  --workers 2 \
  --seeds 42,43,44 \
  --use_amp
```

### 9.3 crop あり

```bash
python main_pseudo.py \
  --img_npy /path/to/main_dataset.npy \
  --do_crop \
  --lung_mask_npy /path/to/lung_mask.npy \
  --out_dir /path/to/output \
  --models resnet18 \
  --epochs 10 \
  --batch 32 \
  --workers 2 \
  --seeds 42,43,44 \
  --use_amp
```

---

## 10. 出力ファイル

主に以下が保存されます。

### seedごとの結果

* `results_<experiment>_seed42.csv`
* `results_<experiment>_seed43.csv`
* ...

### 全seed結合

* `results_<experiment>_seedRepeated.csv`

### mean ± SD 集計

* `results_<experiment>_seedRepeated_mean_sd.csv`

### test予測確率

* `test_prob_<model>.npy`

### best model

* baseline: `*_best.pth`
* pseudo retrain: `*_best_pseudo_retrain.pth`

---

## 11. 結果の見方

出力CSVには少なくとも以下が含まれます。

* `val_auc_best`
* `test_auc_all`
* `test_auc_0-4`
* `test_auc_5-9`
* `test_auc_10-14`
* `test_auc_15-18`
* `pseudo_selected_n`
* `pseudo_acc_vs_gt`
* `pseudo_male_n`
* `pseudo_female_n`

### 推奨する比較

本スクリプトでは、以下の比較を主解析として推奨します。

* Baseline vs Baseline + Pseudo Retrain
* 全体AUC
* 年齢群別AUC
* MacroAUC（各年齢群AUCの単純平均）
* multi-seed の mean ± SD

---

## 12. 解釈上の注意

### 12.1 pseudo fine-tuning との違い

本版では pseudo-label の直接効果を見やすくするため、**retrain** を重視しています。
pseudo fine-tuning は後段で決定境界を寄せる挙動を示すことがあり、分布適応の影響を受けやすいため、主解析には採用しない方針が妥当な場合があります。

### 12.2 0–4歳群の不安定性

0–4歳群は test sample 数が少なく、AUC 推定が不安定になりやすいです。
高いAUCが得られても、過剰解釈は避けてください。

### 12.3 pseudo pool の分布差

pseudo pool が main dataset/test set と分布的に近い場合、pseudo-label は単なるデータ増量ではなく、domain adaptation 的に振る舞う可能性があります。

---

## 13. 推奨される追加解析

* Baseline vs Pseudo Retrain の **paired multi-seed comparison**
* 各 age bin の **ΔAUC**
* **MacroAUC** の比較
* bootstrap CI
* pseudo selected の年齢分布・性別分布
* embedding 空間での train / test / pseudo の可視化

---

## 14. 典型的なワークフロー

1. Baseline を multi-seed で実行
2. Pseudo Retrain を同じ seeds で実行
3. 全体AUC、年齢群別AUC、MacroAUC を比較
4. pseudo selected の件数・精度・分布を確認
5. 必要なら domain adaptation 仮説を検討

---

## 15. まとめ

`main_pseudo.py` は、pseudo-label を **train data augmentation として再学習で評価する** ためのスクリプトです。
特に、pseudo-label の効果を fine-tuning による境界調整と切り分けて解釈したい場合に有用です。

主解析としては、以下を推奨します。

* Baseline
* Baseline + Pseudo Retrain
* multi-seed mean ± SD
* 年齢群別AUC
* MacroAUC

