#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, time, random, argparse
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score

import torchvision.models as tv_models
import torchvision.models as tv
import timm

import cv2
from PIL import Image


# =========================
# 0) Reproducibility
# =========================
def set_seed(seed=42, deterministic=False):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def worker_init_fn(worker_id, base_seed=42):
    s = base_seed + worker_id
    np.random.seed(s)
    random.seed(s)


# =========================
# 1) Utils: metrics
# =========================
def safe_auc(y_true, y_prob):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    if y_true.size == 0 or np.unique(y_true).size < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def auc_by_age_bins(probs, labels, ages, bins, min_n=20):
    rows = []
    for lo, hi in bins:
        idx = (ages >= lo) & (ages <= hi)
        n = int(idx.sum())
        auc = float("nan")
        if n >= min_n and np.unique(labels[idx]).size >= 2:
            auc = float(roc_auc_score(labels[idx], probs[idx]))
        rows.append((f"{lo}-{hi}", n, auc))
    return rows


# =========================
# 2) Classification preprocessing
# =========================
def preprocess_images(imgs):
    """
    imgs: (N,H,W) float32 0-1
    return imgs_sc: (N,H,W) float32 0-1
    """
    N = len(imgs)
    out = imgs.copy().astype(np.float32)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

    for i in range(N):
        x = out[i].astype(np.float32)

        p1, p99 = np.percentile(x, (1, 99))
        x = np.clip(x, p1, p99)

        x = (x - x.min()) / (x.max() - x.min() + 1e-6)

        x8 = (x * 255).astype(np.uint8)
        x = clahe.apply(x8).astype(np.float32) / 255.0

        blur = cv2.GaussianBlur(x, (0,0), 3)
        x = cv2.addWeighted(x, 1.5, blur, -0.5, 0)
        x = np.clip(x, 0, 1)

        x = (x - x.min()) / (x.max() - x.min() + 1e-6)

        out[i] = x

    return out


# =========================
# 3) Classification dataset
# =========================
def sex_to_int(s):
    ss = str(s).strip().lower()
    if ss in ["m", "male", "man", "1", "true"]:
        return 0
    if ss in ["f", "female", "woman", "0", "false"]:
        return 1
    # unknown -> raise (silent fail is dangerous)
    raise ValueError(f"Unknown sex label: {s}")


class EVACXRDataset(Dataset):
    def __init__(self, images, ages, sex):
        self.images = images
        self.ages = ages
        self.sex = sex

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img)
        img = img.float()

        if img.ndim == 2:
            img = img.unsqueeze(0)
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)

        age = self.ages[idx]
        sx  = self.sex[idx]
        sx_i = sex_to_int(sx)

        if isinstance(age, np.generic):
            age = age.item()
        if isinstance(sx, np.generic):
            sx = sx.item()

        return img, torch.tensor(age, dtype=torch.float32), torch.tensor(sx_i, dtype=torch.long)


def make_strata(ages, sex, age_bins=(0,5,10,15,19)):
    age_bin = pd.cut(ages, bins=list(age_bins), right=False, include_lowest=True)
    strata = age_bin.astype(str) + "__" + pd.Series(sex).astype(str)
    return np.asarray(strata)


def split_train_val_test(indices, strata, groups, seed=42, test_splits=5, val_splits=8):
    """
    7:1:2 相当：まず test を sgkf_test で切って、その残りを sgkf_val で train/val に切る
    """
    sgkf_test = StratifiedGroupKFold(n_splits=test_splits, shuffle=True, random_state=seed)
    trainval_idx, test_idx = next(sgkf_test.split(np.zeros(len(indices)), y=strata, groups=groups))

    groups_tv = np.asarray(groups)[trainval_idx]
    strata_tv = np.asarray(strata)[trainval_idx]

    sgkf_val = StratifiedGroupKFold(n_splits=val_splits, shuffle=True, random_state=seed)
    tr2, va2 = next(sgkf_val.split(np.zeros(len(trainval_idx)), y=strata_tv, groups=groups_tv))

    train_idx = trainval_idx[tr2]
    val_idx   = trainval_idx[va2]

    return train_idx, val_idx, test_idx


# =========================
# 4) Model builders (classification)
# =========================
def build_torchvision(name: str, num_classes=2):
    name = name.lower()

    if name == "resnet18":
        m = tv_models.resnet18(weights=tv_models.ResNet18_Weights.IMAGENET1K_V1)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m

    if name == "resnet50":
        m = tv_models.resnet50(weights=tv_models.ResNet50_Weights.IMAGENET1K_V2)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m

    if name == "resnet101":
        m = tv_models.resnet101(weights=tv_models.ResNet101_Weights.IMAGENET1K_V2)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m

    if name == "densenet121":
        m = tv_models.densenet121(weights=tv_models.DenseNet121_Weights.IMAGENET1K_V1)
        m.classifier = nn.Linear(m.classifier.in_features, num_classes)
        return m

    if name == "convnext_tiny":
        m = tv_models.convnext_tiny(weights=tv_models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, num_classes)
        return m

    if name == "vit_b_16":
        m = tv_models.vit_b_16(weights=tv_models.ViT_B_16_Weights.IMAGENET1K_V1)
        m.heads.head = nn.Linear(m.heads.head.in_features, num_classes)
        return m

    if name == "swin_t":
        m = tv_models.swin_t(weights=tv_models.Swin_T_Weights.IMAGENET1K_V1)
        m.head = nn.Linear(m.head.in_features, num_classes)
        return m

    raise ValueError(f"[torchvision] Unknown model: {name}")


def build_timm(name: str, num_classes=2):
    return timm.create_model(name, pretrained=True, num_classes=num_classes)


def build_model_any(name: str, num_classes=2):
    tv_names = {"resnet18","resnet50","resnet101","densenet121","convnext_tiny","vit_b_16","swin_t"}
    if name.lower() in tv_names:
        return build_torchvision(name, num_classes=num_classes)
    return build_timm(name, num_classes=num_classes)


# =========================
# 5) Train / Predict (classification)
# =========================
def train_one_epoch(model, loader, optimizer, criterion, device, use_amp=True):
    model.train()
    total_loss = 0.0

    device_type = "cuda" if str(device).startswith("cuda") else "cpu"
    amp_enabled = bool(use_amp and device_type == "cuda")
    scaler = torch.amp.GradScaler(device_type, enabled=amp_enabled)

    if device_type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    for x, age, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).long()

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type, enabled=amp_enabled):
            logits = model(x)
            loss = criterion(logits, y)

        if amp_enabled:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += float(loss.item())

    if device_type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    return total_loss / max(len(loader), 1), (t1 - t0)


@torch.no_grad()
def predict_probs_labels_ages(model, loader, device, use_amp=True):
    model.eval()
    device_type = "cuda" if str(device).startswith("cuda") else "cpu"
    amp_enabled = bool(use_amp and device_type == "cuda")

    probs_list, y_list, age_list = [], [], []

    if device_type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    for x, age, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.amp.autocast(device_type, enabled=amp_enabled):
            logits = model(x)
            prob = torch.softmax(logits, dim=1)[:, 1]
        probs_list.append(prob.detach().cpu().numpy())
        y_list.append(np.asarray(y))
        age_list.append(np.asarray(age))

    if device_type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    probs = np.concatenate(probs_list)
    y = np.concatenate(y_list)
    age = np.concatenate(age_list)
    return probs, y, age, (t1 - t0)


def run_one_model(name, train_loader, val_loader, test_loader, device,
                  age_bins=((0,4),(5,9),(10,14),(15,18)),
                  epochs=10, base_lr=1e-4, weight_decay=1e-4,
                  use_amp=False, save_root="checkpoints_compare", save_test_probs=True):

    save_root = Path(save_root)
    save_root.mkdir(parents=True, exist_ok=True)

    model = build_model_any(name, num_classes=2).to(device)
    lr = base_lr * 0.5 if "eva02" in name.lower() else base_lr

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_auc = -1.0
    best_path = save_root / f"{name.replace('/','_')}_best.pth"

    epoch_times = []
    val_pred_times = []

    for epoch in range(1, epochs + 1):
        loss, t_train = train_one_epoch(model, train_loader, optimizer, criterion, device, use_amp=use_amp)
        epoch_times.append(t_train)

        val_probs, val_y, val_age, t_valpred = predict_probs_labels_ages(model, val_loader, device, use_amp=use_amp)
        val_pred_times.append(t_valpred)

        val_auc = safe_auc(val_y, val_probs)

        print(f"[{name}] epoch {epoch:02d}/{epochs} "
              f"loss={loss:.4f} val_auc={val_auc:.3f} "
              f"train_time={t_train:.1f}s val_pred_time={t_valpred:.1f}s lr={lr:g}")

        if np.isfinite(val_auc) and val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), best_path)
            print("  -> saved best")

    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device))

    test_probs, test_y, test_age, t_testpred = predict_probs_labels_ages(model, test_loader, device, use_amp=use_amp)
    test_auc_all = safe_auc(test_y, test_probs)
    test_bins = auc_by_age_bins(test_probs, test_y, test_age, bins=age_bins)

    if save_test_probs:
        np.save(save_root / f"test_prob_{name.replace('/','_')}.npy", np.asarray(test_probs))

    row = {
        "model": name,
        "val_auc_best": best_val_auc,
        "test_auc_all": test_auc_all,
        "time_train_total_sec": float(np.sum(epoch_times)),
        "time_train_epoch_mean_sec": float(np.mean(epoch_times)),
        "time_val_pred_total_sec": float(np.sum(val_pred_times)),
        "time_test_pred_sec": float(t_testpred),
    }
    for bin_name, n, auc in test_bins:
        row[f"test_auc_{bin_name}"] = auc
        row[f"n_{bin_name}"] = n

    print(f"[{name}] TEST AUC(all) = {test_auc_all:.3f}")
    print("  test bins:", test_bins)

    return row


def run_experiment(exp_name, images, ages, sex, pids, model_names,
                   seed=42, batch_size=32, num_workers=2,
                   epochs=10, base_lr=1e-4, use_amp=False,
                   save_root="checkpoints_compare",
                   age_bins=((0,4),(5,9),(10,14),(15,18))):

    print("\n" + "="*100)
    print(f"EXPERIMENT: {exp_name}")
    print("="*100)

    strata = make_strata(ages, sex)
    idx_all = np.arange(len(images))

    train_idx, val_idx, test_idx = split_train_val_test(
        idx_all, strata=strata, groups=pids, seed=seed
    )
    print("N (train/val/test):", len(train_idx), len(val_idx), len(test_idx))
    print("unique patients:", len(np.unique(np.asarray(pids)[train_idx])),
                         len(np.unique(np.asarray(pids)[val_idx])),
                         len(np.unique(np.asarray(pids)[test_idx])))

    full_ds = EVACXRDataset(images=images, ages=ages, sex=sex)
    train_ds = Subset(full_ds, train_idx)
    val_ds   = Subset(full_ds, val_idx)
    test_ds  = Subset(full_ds, test_idx)

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=True, worker_init_fn=lambda wid: worker_init_fn(wid, seed),
        generator=g
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=True, worker_init_fn=lambda wid: worker_init_fn(wid, seed),
        generator=g
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=True, worker_init_fn=lambda wid: worker_init_fn(wid, seed),
        generator=g
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = []
    for mn in model_names:
        row = run_one_model(
            mn, train_loader, val_loader, test_loader, device,
            age_bins=age_bins, epochs=epochs, base_lr=base_lr,
            use_amp=use_amp, save_root=save_root, save_test_probs=True
        )
        row["experiment"] = exp_name
        results.append(row)

    df = pd.DataFrame(results)
    out_csv = Path(save_root) / f"results_{exp_name.replace(' ','_')}.csv"
    df.to_csv(out_csv, index=False)
    print("saved:", out_csv)
    return df


# =========================
# 6) Segmentation parts (lung field) - minimal refactor
# =========================
class StackSegDataset(Dataset):
    def __init__(self, X, Y, indices=None, size=512, do_aug=False):
        self.X = X
        self.Y = Y
        self.size = size
        self.do_aug = do_aug
        self.indices = np.arange(len(Y)) if indices is None else np.array(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, k):
        i = int(self.indices[k])
        img = np.asarray(self.X[i])
        mask = np.asarray(self.Y[i])
        if mask.ndim == 3:
            mask = mask[..., 0]

        img  = cv2.resize(img,  (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.size, self.size), interpolation=cv2.INTER_NEAREST)

        mask = (mask > 0).astype(np.float32)

        img  = torch.from_numpy(img).float().unsqueeze(0)
        mask = torch.from_numpy(mask).float().unsqueeze(0)

        img = (img - img.min()) / (img.max() - img.min() + 1e-6)

        if self.do_aug:
            if torch.rand(1).item() < 0.5:
                img  = torch.flip(img,  dims=[2])
                mask = torch.flip(mask, dims=[2])
            if torch.rand(1).item() < 0.8:
                gain = 1.0 + (torch.rand(1).item() - 0.5) * 0.10
                bias = (torch.rand(1).item() - 0.5) * 0.05
                img = torch.clamp(img * gain + bias, 0.0, 1.0)

        return img, mask


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.net(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.conv1 = ConvBNReLU(in_ch + skip_ch, out_ch)
        self.conv2 = ConvBNReLU(out_ch, out_ch)
    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv2(self.conv1(x))
        return x


class ResNet18UNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        m = tv.resnet18(weights=tv.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)

        self.conv1 = m.conv1
        self.bn1   = m.bn1
        self.relu  = m.relu
        self.maxpool = m.maxpool

        self.layer1 = m.layer1
        self.layer2 = m.layer2
        self.layer3 = m.layer3
        self.layer4 = m.layer4

        self.bridge = nn.Sequential(ConvBNReLU(512, 512), ConvBNReLU(512, 512))

        self.dec4 = DecoderBlock(512, 512, 256)
        self.dec3 = DecoderBlock(256, 256, 128)
        self.dec2 = DecoderBlock(128, 128, 64)
        self.dec1 = DecoderBlock(64,  64,  64)

        self.head = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        x0 = self.relu(self.bn1(self.conv1(x)))
        x1 = self.maxpool(x0)

        e1 = self.layer1(x1)
        e2 = self.layer2(e1)
        e3 = self.layer3(e2)
        e4 = self.layer4(e3)

        b  = self.bridge(e4)

        d4 = self.dec4(b,  e4)
        d3 = self.dec3(d4, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)

        out = F.interpolate(d1, scale_factor=4, mode="bilinear", align_corners=False)
        return self.head(out)


def dice_loss_logits(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits).flatten(1)
    targets = targets.flatten(1)
    inter = (probs * targets).sum(1)
    den = probs.sum(1) + targets.sum(1)
    dice = (2*inter + eps) / (den + eps)
    return (1 - dice).mean()


@torch.no_grad()
def dice_score_logits(logits, targets, thr=0.5, eps=1e-6):
    probs = torch.sigmoid(logits)
    preds = (probs > thr).float()
    inter = (preds * targets).sum(dim=(2,3))
    union = preds.sum(dim=(2,3)) + targets.sum(dim=(2,3))
    return ((2*inter + eps) / (union + eps)).mean().item()


def set_encoder_trainable(model, trainable: bool):
    enc = []
    enc += list(model.conv1.parameters())
    enc += list(model.bn1.parameters())
    enc += list(model.layer1.parameters())
    enc += list(model.layer2.parameters())
    enc += list(model.layer3.parameters())
    enc += list(model.layer4.parameters())
    for p in enc:
        p.requires_grad = trainable


def run_train_seg(model, train_loader, val_loader, device,
                  epochs=30, lr=2e-4, weight_decay=1e-4,
                  unfreeze_epoch=10, unfreeze_lr=1e-4,
                  patience=5, save_path="best_pretrain_all.pt",
                  w_dice=0.7, pos_weight=3.0, use_amp=True):

    device_type = "cuda" if str(device).startswith("cuda") else "cpu"
    amp_enabled = bool(use_amp and device_type == "cuda")
    scaler = torch.amp.GradScaler(device_type, enabled=amp_enabled)

    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))

    def loss_fn(logits, targets):
        return w_dice * dice_loss_logits(logits, targets) + (1-w_dice) * bce(logits, targets)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    model = model.to(device)

    set_encoder_trainable(model, False)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                  lr=lr, weight_decay=weight_decay)

    best = -1.0
    bad = 0

    for ep in range(1, epochs + 1):
        if ep == unfreeze_epoch + 1:
            set_encoder_trainable(model, True)
            optimizer = torch.optim.AdamW(model.parameters(), lr=unfreeze_lr, weight_decay=weight_decay)
            print(f"✅ encoder unfrozen @epoch {ep}, lr={unfreeze_lr}")

        model.train()
        tr_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type, enabled=amp_enabled):
                logits = model(xb)
                loss = loss_fn(logits, yb)

            if amp_enabled:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            tr_loss += loss.item() * xb.size(0)

        tr_loss /= max(len(train_loader.dataset), 1)

        model.eval()
        va_dices = []
        va_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                with torch.amp.autocast(device_type, enabled=amp_enabled):
                    logits = model(xb)
                    loss = loss_fn(logits, yb)
                va_losses.append(float(loss.item()))
                va_dices.append(float(dice_score_logits(logits, yb)))

        va_loss = float(np.mean(va_losses)) if va_losses else float("nan")
        va_dice = float(np.mean(va_dices)) if va_dices else float("nan")

        print(f"Epoch {ep:03d} | train_loss {tr_loss:.4f} | val_loss {va_loss:.4f} | val_dice {va_dice:.4f}")

        if np.isfinite(va_dice) and va_dice > best + 1e-4:
            best = va_dice
            bad = 0
            torch.save({"model": model.state_dict(), "best_val_dice": best}, save_path)
            print("  ✅ saved best:", best)
        else:
            bad += 1
            if bad >= patience:
                print("  ⏹ early stopping. best val dice:", best)
                break

    return str(save_path), best


# =========================
# 7) Thorax crop by lung mask
# =========================
def central_crop_bbox(H, W, scale=0.75):
    s = int(min(H, W) * scale)
    y1 = (H - s)//2
    x1 = (W - s)//2
    return x1, y1, x1+s, y1+s


def bbox_from_lung_mask_safe(mask, pad_x=0.1, pad_y=0.1, extra_down=0.1,
                            make_square=True, min_area_ratio=0.05):
    m = (mask > 0).astype(np.uint8)
    H, W = m.shape

    k = max(3, (min(H, W)//256)*2 + 3)
    kernel = np.ones((k, k), np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN,  kernel)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)

    if m.sum() < H*W*min_area_ratio:
        return central_crop_bbox(H, W, scale=0.75)

    ys, xs = np.where(m > 0)
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()

    bw = (x2 - x1 + 1)
    bh = (y2 - y1 + 1)

    px = int(bw * pad_x)
    py = int(bh * pad_y)
    pd = int(bh * extra_down)

    x1 = max(0, x1 - px)
    x2 = min(W-1, x2 + px)
    y1 = max(0, y1 - py)
    y2 = min(H-1, y2 + py + pd)

    if make_square:
        bw = x2 - x1 + 1
        bh = y2 - y1 + 1
        s = max(bw, bh)
        cx = (x1 + x2)//2
        cy = (y1 + y2)//2

        x1 = cx - s//2
        x2 = x1 + s - 1
        y1 = cy - s//2
        y2 = y1 + s - 1

        if x1 < 0: x2 -= x1; x1 = 0
        if y1 < 0: y2 -= y1; y1 = 0
        if x2 >= W:
            d = x2 - (W-1); x1 -= d; x2 = W-1
            x1 = max(0, x1)
        if y2 >= H:
            d = y2 - (H-1); y1 -= d; y2 = H-1
            y1 = max(0, y1)

    return int(x1), int(y1), int(x2+1), int(y2+1)


def crop_resize_norm(img, mask, out_size=512):
    H, W = img.shape[:2]
    x1,y1,x2,y2 = bbox_from_lung_mask_safe(mask, min_area_ratio=0.05)
    img_c = img[y1:y2, x1:x2]
    img_p = Image.fromarray(img_c.astype(np.float32)).resize((out_size, out_size), Image.BILINEAR)
    x = np.array(img_p, dtype=np.float32)
    x = (x - x.min()) / (x.max() - x.min() + 1e-6)
    return x


def make_thorax_crop_memmap(images, masks, out_path, out_size=512):
    out_path = Path(out_path)
    N = len(images)
    mm = np.memmap(out_path, dtype="float32", mode="w+", shape=(N, out_size, out_size))
    for i in range(N):
        mm[i] = crop_resize_norm(images[i], masks[i], out_size=out_size)
        if (i+1) % 200 == 0:
            print("crop done", i+1, "/", N)
    mm.flush()
    return mm


# =========================
# 8) main()
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_npy", type=str, required=True,
                        help="cxr_pediatric_images512_...npy (dict with keys images/ages/patient_id/sex)")
    parser.add_argument("--out_dir", type=str, default="checkpoints_compare")
    parser.add_argument("--models", type=str, default="resnet18",
                        help="comma separated, e.g. resnet18,convnext_tiny")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_amp", action="store_true")

    parser.add_argument("--do_preprocess", action="store_true",
                        help="apply CLAHE+enhance preprocessing")
    parser.add_argument("--do_crop", action="store_true",
                        help="apply thorax crop using precomputed lung mask")
    parser.add_argument("--lung_mask_npy", type=str, default="",
                        help="lung_mask_3400.npy (uint8 0/1), required if --do_crop")

    args = parser.parse_args()

    set_seed(args.seed, deterministic=False)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(args.img_npy, allow_pickle=True).item()
    imgs = data["images"]
    ages = data["ages"]
    pids = data["patient_id"]
    sex  = data["sex"]

    imgs = np.asarray(imgs).astype(np.float32)
    print("Loaded images:", imgs.shape, imgs.min(), imgs.max())

    if args.do_preprocess:
        print(">>> preprocessing enabled")
        imgs = preprocess_images(imgs)

    if args.do_crop:
        if not args.lung_mask_npy:
            raise ValueError("--do_crop requires --lung_mask_npy")
        masks = np.load(args.lung_mask_npy, mmap_mode="r")
        if masks.ndim == 4:
            masks = masks[..., 0]
        print("Loaded masks:", masks.shape, masks.dtype, "unique:", np.unique(masks)[:10])

        crop_path = out_dir / "thorax_crop_all_512.npy"
        print(">>> thorax cropping ->", crop_path)
        imgs = make_thorax_crop_memmap(imgs, masks, crop_path, out_size=512)
        # memmap returns view; keep as np.asarray-like
        print("Cropped images:", imgs.shape, float(np.min(imgs)), float(np.max(imgs)))

    model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    df = run_experiment(
        exp_name=("Preproc+Crop" if (args.do_preprocess and args.do_crop)
                  else "Preprocessing" if args.do_preprocess
                  else "Thoracic_Crop" if args.do_crop
                  else "Baseline"),
        images=imgs, ages=ages, sex=sex, pids=pids,
        model_names=model_names,
        seed=args.seed, batch_size=args.batch, num_workers=args.workers,
        epochs=args.epochs, base_lr=1e-4, use_amp=args.use_amp,
        save_root=str(out_dir),
        age_bins=((0,4),(5,9),(10,14),(15,18))
    )

    print("\n=== DONE ===")
    print(df)


if __name__ == "__main__":
    main()
