import argparse
import os
import time
import random
import yaml
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from typing import Optional
from dataloader import create_dataloader
from train import train_model
from models import MLPClassifier


# -------------------------
# Helpers
# -------------------------
def load_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def get_device(pref: str):
    if pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(pref)

def build_save_dir(root: str, fold_test: int, fold_inner: Optional[int]):
    if fold_inner is None:
        return os.path.join(root, f"test_{fold_test}")
    return os.path.join(root, f"test_{fold_test}", f"fold_{fold_inner}")

def build_splits_paths(split_base: str, fold_test: int, fold_inner: Optional[int]):
    """
    Deterministic — no fallback.
    If fold_inner is None → final outer model → train on train_split.json, test on test_split.json.
    If fold_inner is k → inner validation → train on train_split_k.json, validate on val_split_k.json.
    """
    test_dir = os.path.join(split_base, f"data-{fold_test}")

    # TRAIN
    if fold_inner is None:
        train_split = os.path.join(test_dir, "train_split.json")
    else:
        train_split = os.path.join(test_dir, f"train_split_{fold_inner}.json")

    if not os.path.exists(train_split):
        raise FileNotFoundError(f"Missing train split: {train_split}")

    # VAL
    if fold_inner is None:
        val_split = os.path.join(test_dir, "test_split.json")   # final test
    else:
        val_split = os.path.join(test_dir, f"val_split_{fold_inner}.json")

    if not os.path.exists(val_split):
        raise FileNotFoundError(f"Missing validation split: {val_split}")

    return train_split, val_split


def build_simclr_model(cfg, device):
    """
    Build your original ResNetSimCLR backbone (images mode), load SimCLR weights, and freeze the backbone.
    Assumes models.py contains your original ResNetSimCLR class.
    """
    from models import ResNetSimCLR  # import here to avoid hard dependency in embeddings mode

    arch = cfg["model"]["arch"]            # 'resnet18' or 'resnet50'
    out_dim = cfg["model"]["output_dim"]   # 3 classes
    dim_in = 4                             # one slice with 4 channels (precon, corticomedullary, nephrographic, excretory)

    model = ResNetSimCLR(base_model=arch, dim_in=dim_in, out_dim=out_dim).to(device)

    ckpt_path = cfg["simclr"]["checkpoint_path"]
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"SimCLR checkpoint not found at: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt.get("state_dict", ckpt)  # support raw or wrapped
    log = model.load_state_dict(state_dict, strict=False)

    # freeze all layers except final fc
    for name, p in model.named_parameters():
        if name not in ["backbone.fc.weight", "backbone.fc.bias"]:
            p.requires_grad = False

    # sanity check: only fc is trainable
    trainables = [n for n, p in model.named_parameters() if p.requires_grad]
    assert set(trainables) == {"backbone.fc.weight", "backbone.fc.bias"}, \
        f"Unexpected trainables for frozen SimCLR: {trainables}"

    return model


# -------------------------
# CLI
# -------------------------
def get_args():
    parser = argparse.ArgumentParser(description="Training entry (dual-mode)")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config")
    parser.add_argument("--fold_test", type=int, required=True, help="Outer/test fold index")
    parser.add_argument("--fold_inner", type=int, default=None, help="Inner/validation fold index (optional)")
    parser.add_argument("--seed", type=int, default=123457, help="Seed value (kept in main.py by request)")
    return parser.parse_args()


# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    args = get_args()

    # ✅ Seed stays in main.py (exact original behavior)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Load config and device
    cfg = load_config(args.config)
    device = get_device(cfg["device"])
    mode = cfg["data"]["mode"].strip().lower()  # "embeddings" or "images"

    # Resolve splits for this outer/inner fold setting
    train_split, val_split = build_splits_paths(
        split_base=cfg["data"]["split_path_base"],
        fold_test=args.fold_test,
        fold_inner=args.fold_inner
    )

    # Save directory structure (results/test_{outer}/[fold_{inner}])
    save_dir = build_save_dir(
        root=cfg["output"]["save_dir"],
        fold_test=args.fold_test,
        fold_inner=args.fold_inner
    )
    os.makedirs(os.path.join(save_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "tensorboard"), exist_ok=True)

    # Build data loaders (factory internally switches on cfg["data"]["mode"])
    train_loader = create_dataloader(
        cfg=cfg,
        split_json_path=train_split,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True
    )
    valid_loader = create_dataloader(
        cfg=cfg,
        split_json_path=val_split,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False
    )

    # Build model
    if mode == "embeddings":
        model = MLPClassifier(
            dim_in=cfg["model"]["embed_dim"],
            out_dim=cfg["model"]["output_dim"]
        ).to(device)
    elif mode == "images":
        model = build_simclr_model(cfg, device)
    else:
        raise ValueError(f"Unsupported data.mode: {mode}")

    # Optimizer / criterion
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"]
    )
    criterion = torch.nn.CrossEntropyLoss().to(device)

    # TensorBoard writer & hyperparams file (same spirit as your original)
    writer = SummaryWriter(log_dir=os.path.join(save_dir, "tensorboard"))

    # Hyperparameters record
    hyper_file = os.path.join(save_dir, "hyperparameters.txt")
    with open(hyper_file, "w") as fid:
        fid.write(f"Mode = {mode}\n")
        fid.write(f"Device = {device}\n")
        fid.write(f"Seed = {args.seed}\n")
        fid.write(f"Outer fold (test) = {args.fold_test}\n")
        fid.write(f"Inner fold (val)  = {args.fold_inner}\n")
        fid.write(f"Training epochs = {cfg['training']['epochs']}\n")
        fid.write(f"Batch size = {cfg['training']['batch_size']}\n")
        fid.write(f"Learning rate = {cfg['training']['learning_rate']}\n")
        fid.write(f"L2 (weight_decay) = {cfg['training']['weight_decay']}\n")
        if mode == "embeddings":
            fid.write(f"Embeddings path = {cfg['data']['embeddings_path']}\n")
        else:
            fid.write(f"Data path (HDF5) = {cfg['data']['data_path']}\n")
            fid.write(f"SimCLR checkpoint = {cfg['simclr']['checkpoint_path']}\n")
            fid.write(f"Backbone arch = {cfg['model']['arch']}\n")

    # Diary file for epoch-by-epoch logs (as in your original)
    print_file = open(os.path.join(save_dir, "diary.txt"), "w")

    # ---- Train (your train.py logic is preserved) ----
    t0 = time.time()
    t_acc, v_acc, best_t_acc, best_v_acc, train_auc_classes, valid_auc_classes = train_model(
        model=model,
        n_epochs=cfg["training"]["epochs"],
        train_loader=train_loader,
        valid_loader=valid_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        savepath=save_dir,
        writer=writer,
        print_file=print_file,
    )
    elapsed = time.time() - t0

    # Save final model / optimizer (train.py also saves checkpoints)
    torch.save(model.state_dict(), os.path.join(save_dir, "models", "model.pt"))
    torch.save(optimizer.state_dict(), os.path.join(save_dir, "models", "optimizer.pt"))

    # Results summary (matching your original style)
    results_file = os.path.join(save_dir, "results.txt")
    with open(results_file, "w") as fid:
        fid.write(f"Total elapsed time is: {elapsed:.3f}\n")
        fid.write(f"Training accuracy: {t_acc:.3f}\n")
        fid.write(f"Validation accuracy: {v_acc:.3f}\n")
        fid.write(f"Best training accuracy is: {best_t_acc:.3f}\n")
        fid.write(f"Best validation accuracy is: {best_v_acc:.3f}\n")
        fid.write(f"Training AUC (OvR): {train_auc_classes}\n")
        fid.write(f"Validation AUC (OvR): {valid_auc_classes}\n")
