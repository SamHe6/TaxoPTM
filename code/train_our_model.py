import argparse
import csv
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, matthews_corrcoef, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup
try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - tqdm is optional for non-interactive environments.
    tqdm = None

from ptm_head import (
    AblationPTMModel,
    compute_loss,
    load_prompt_metadata,
    maybe_drop_prompt_ids,
)


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_DIR / "data" / "processed" / "open_phospho_clean_cdhit"
DEFAULT_WINDOWS = DEFAULT_DATA_DIR / "phospho_windows_41aa.tsv"
DEFAULT_META = PROJECT_DIR / "data" / "processed" / "species_benchmarks" / "all_samples_with_taxon_group.tsv"
DEFAULT_OUT_DIR = PROJECT_DIR / "outputs" / "esm_finetune_ablation"
DEFAULT_MODEL = "facebook/esm2_t6_8M_UR50D"


RARE_TO_X = str.maketrans({"U": "X", "Z": "X", "O": "X", "B": "X", "J": "X"})


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune ESM-2 with the prompt-aware PTM head.")
    parser.add_argument("--windows_tsv", default=str(DEFAULT_WINDOWS))
    parser.add_argument("--metadata_tsv", default=str(DEFAULT_META))
    parser.add_argument("--model_dir", default=DEFAULT_MODEL)
    parser.add_argument("--init_checkpoint", default=None, help="Optional checkpoint used to warm-start matching weights.")
    parser.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--run_name", default="esm_finetune_run")

    parser.add_argument(
        "--prompt_mode",
        choices=["none", "tax_id", "taxon_group", "broad_group"],
        default="taxon_group",
    )
    parser.add_argument(
        "--prompt_fusion",
        choices=["none", "add_concat", "concat_only", "add_only"],
        default="add_concat",
    )
    parser.add_argument("--prompt_dropout", type=float, default=0.2)
    parser.add_argument("--prompt_token_dropout", type=float, default=0.3)

    parser.add_argument("--disable_site_relative", action="store_true")
    parser.add_argument("--disable_center_attention", action="store_true")
    parser.add_argument("--disable_prototype", action="store_true")
    parser.add_argument("--disable_evidence", action="store_true")

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr_head", type=float, default=1e-3)
    parser.add_argument("--lr_esm", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--scheduler", choices=["none", "cosine"], default="none")
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--early_stop_patience", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--freeze_esm", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--freeze_esm_embeddings", action="store_true")
    parser.add_argument("--freeze_esm_first_n_layers", type=int, default=0)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_val_samples", type=int, default=None)
    parser.add_argument("--max_test_samples", type=int, default=None)
    parser.add_argument("--train_split", default="train")
    parser.add_argument("--val_split", default="val")
    parser.add_argument("--test_split", default="test")
    parser.add_argument("--extra_eval_splits", nargs="*", default=[])
    parser.add_argument("--force_unk_splits", nargs="*", default=["ood"])

    parser.add_argument("--lambda_sparse", type=float, default=0.01)
    parser.add_argument("--lambda_div", type=float, default=0.01)
    parser.add_argument("--lambda_evidence", type=float, default=0.02)
    parser.add_argument("--lambda_width", type=float, default=0.03)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def clean_seq(seq):
    return str(seq).strip().upper().translate(RARE_TO_X)


def build_prompt_vocab(train_ids, prompt_by_id, prompt_mode):
    if prompt_mode == "none":
        return {"<unk>": 0}
    labels = sorted({prompt_by_id.get(str(sample_id), "other") for sample_id in train_ids})
    vocab = {"<unk>": 0}
    for label in labels:
        if label not in vocab:
            vocab[label] = len(vocab)
    return vocab


class WindowSequenceDataset(Dataset):
    def __init__(self, rows, prompt_by_id, prompt_vocab, prompt_mode, tokenizer):
        self.ids = rows["sample_id"].astype(str).tolist()
        self.seqs = [clean_seq(x) for x in rows["window"].astype(str).tolist()]
        self.labels = rows["label"].astype(int).tolist()
        prompts = []
        for sample_id in self.ids:
            if prompt_mode == "none":
                prompts.append(0)
            else:
                label = prompt_by_id.get(str(sample_id), "other")
                prompts.append(prompt_vocab.get(label, prompt_vocab["<unk>"]))
        self.prompts = prompts
        enc = tokenizer(
            self.seqs,
            return_tensors="np",
            padding=True,
            truncation=False,
            add_special_tokens=True,
            return_special_tokens_mask=True,
        )
        self.input_ids = enc["input_ids"].astype(np.int64)
        self.attention_mask = enc["attention_mask"].astype(np.int64)
        self.special_tokens_mask = enc["special_tokens_mask"].astype(np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "sample_id": self.ids[idx],
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "special_tokens_mask": self.special_tokens_mask[idx],
            "label": self.labels[idx],
            "prompt_id": self.prompts[idx],
        }


def collate_tokenized(batch):
    input_ids = torch.tensor(np.stack([item["input_ids"] for item in batch]), dtype=torch.long)
    attention_mask = torch.tensor(np.stack([item["attention_mask"] for item in batch]), dtype=torch.long)
    special_tokens_mask = torch.tensor(np.stack([item["special_tokens_mask"] for item in batch]), dtype=torch.long)
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.float32)
    prompt_ids = torch.tensor([item["prompt_id"] for item in batch], dtype=torch.long)
    sample_ids = [item["sample_id"] for item in batch]
    return input_ids, attention_mask, special_tokens_mask, labels, prompt_ids, sample_ids


class ESMFineTunePTM(nn.Module):
    def __init__(self, esm_model, head_model):
        super().__init__()
        self.esm = esm_model
        self.head = head_model

    @staticmethod
    def strip_special_tokens(hidden, attention_mask, special_tokens_mask, input_ids=None):
        valid = attention_mask.bool() & (special_tokens_mask == 0)
        lengths = valid.sum(dim=1).long().clamp_min(1)
        max_len = int(lengths.max().item())
        bsz, _tok_len, hidden_size = hidden.shape
        x_emb = hidden.new_zeros((bsz, max_len, hidden_size))
        x_mask = torch.zeros((bsz, max_len), device=hidden.device, dtype=torch.long)
        x_tokens = None
        if input_ids is not None:
            x_tokens = torch.zeros((bsz, max_len), device=input_ids.device, dtype=input_ids.dtype)
        for i in range(bsz):
            emb_i = hidden[i, valid[i], :]
            length = emb_i.size(0)
            x_emb[i, :length, :] = emb_i
            x_mask[i, :length] = 1
            if x_tokens is not None:
                x_tokens[i, :length] = input_ids[i, valid[i]]
        return x_emb, x_mask, x_tokens

    def forward(self, input_ids, attention_mask, special_tokens_mask, prompt_id):
        out = self.esm(input_ids=input_ids, attention_mask=attention_mask)
        x_emb, x_mask, x_tokens = self.strip_special_tokens(
            out.last_hidden_state,
            attention_mask,
            special_tokens_mask,
            input_ids=input_ids,
        )
        return self.head(x_emb, x_mask, prompt_id)


def metric_dict(y_true, y_prob, threshold=0.5):
    y_pred = [1 if v >= threshold else 0 for v in y_prob]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    acc = (tp + tn) / max(1, tp + tn + fp + fn)
    sen = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spe = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    mcc = matthews_corrcoef(y_true, y_pred) if len(set(y_true)) > 1 else 0.0
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = 0.0
    return {
        "ACC": float(acc),
        "SEN": float(sen),
        "SPE": float(spe),
        "MCC": float(mcc),
        "AUC": float(auc),
        "threshold": float(threshold),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


@torch.no_grad()
def collect_predictions(model, loader, device, force_prompt_id=None, progress_desc=None):
    model.eval()
    y_true, y_prob = [], []
    iterator = loader
    if progress_desc and tqdm is not None:
        iterator = tqdm(loader, desc=progress_desc, unit="batch", leave=False)
    for input_ids, attention_mask, special_tokens_mask, y, prompt_id, _sid in iterator:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        special_tokens_mask = special_tokens_mask.to(device)
        prompt_id = prompt_id.to(device)
        if force_prompt_id is not None:
            prompt_id = torch.full_like(prompt_id, int(force_prompt_id))
        outputs = model(input_ids, attention_mask, special_tokens_mask, prompt_id)
        y_prob.extend(outputs["prob"].detach().cpu().numpy().tolist())
        y_true.extend(y.numpy().astype(int).tolist())
    return y_true, y_prob


@torch.no_grad()
def eval_model(model, loader, device, force_prompt_id=None, threshold=0.5, progress_desc=None):
    y_true, y_prob = collect_predictions(
        model,
        loader,
        device,
        force_prompt_id=force_prompt_id,
        progress_desc=progress_desc,
    )
    return metric_dict(y_true, y_prob, threshold=threshold)


def best_mcc_threshold(y_true, y_prob):
    if len(set(y_true)) < 2:
        return 0.5, 0.0
    candidates = sorted(set(float(x) for x in y_prob))
    # Include the conventional threshold and edge-safe thresholds.
    candidates = sorted(set([0.5, 0.0, 1.0] + candidates))
    best_threshold = 0.5
    best_mcc = -1.0
    for threshold in candidates:
        y_pred = [1 if v >= threshold else 0 for v in y_prob]
        mcc = matthews_corrcoef(y_true, y_pred)
        if mcc > best_mcc or (mcc == best_mcc and abs(threshold - 0.5) < abs(best_threshold - 0.5)):
            best_mcc = float(mcc)
            best_threshold = float(threshold)
    return best_threshold, best_mcc


def train_one_epoch(model, loader, optimizer, scheduler, scaler, device, args, unk_id, progress_desc=None):
    model.train()
    total = 0.0
    n = 0
    use_amp = args.fp16 and device.type == "cuda"
    optimizer.zero_grad(set_to_none=True)
    iterator = loader
    if progress_desc and tqdm is not None:
        iterator = tqdm(loader, desc=progress_desc, unit="batch", leave=False)
    for step, (input_ids, attention_mask, special_tokens_mask, y, prompt_id, _sid) in enumerate(iterator, start=1):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        special_tokens_mask = special_tokens_mask.to(device)
        y = y.to(device)
        prompt_id = maybe_drop_prompt_ids(prompt_id.to(device), unk_id, args.prompt_token_dropout)

        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            outputs = model(input_ids, attention_mask, special_tokens_mask, prompt_id)
            loss, _parts = compute_loss(outputs, y, args)
            step_loss = loss / max(1, args.grad_accum_steps)

        should_step = step % max(1, args.grad_accum_steps) == 0 or step == len(loader)
        if scaler is not None:
            scaler.scale(step_loss).backward()
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
        else:
            step_loss.backward()
            if should_step:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        total += float(loss.detach()) * y.size(0)
        n += y.size(0)
    return total / max(1, n)


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_split_rows(windows_tsv, split, max_samples=None, seed=42):
    df = pd.read_csv(windows_tsv, sep="\t")
    rows = df[df["split"] == split].copy()
    if rows.empty:
        raise ValueError(f"No rows found for split={split!r} in {windows_tsv}")
    rows["label"] = rows["label"].astype(int)
    if max_samples is not None and len(rows) > max_samples:
        rows = rows.sample(n=max_samples, random_state=seed).reset_index(drop=True)
    return rows


def make_loader(rows, prompt_by_id, prompt_vocab, prompt_mode, tokenizer, batch_size, shuffle):
    ds = WindowSequenceDataset(rows, prompt_by_id, prompt_vocab, prompt_mode, tokenizer)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0, collate_fn=collate_tokenized)
    return ds, loader


def apply_esm_training_policy(esm, args):
    if args.gradient_checkpointing and hasattr(esm, "gradient_checkpointing_enable"):
        esm.gradient_checkpointing_enable()
    if args.freeze_esm_embeddings and hasattr(esm, "embeddings"):
        for param in esm.embeddings.parameters():
            param.requires_grad = False
    if args.freeze_esm_first_n_layers > 0:
        encoder = getattr(esm, "encoder", None)
        layers = getattr(encoder, "layer", None)
        if layers is None:
            print("WARNING: could not find esm.encoder.layer; skip layer freezing")
        else:
            for layer in list(layers)[: args.freeze_esm_first_n_layers]:
                for param in layer.parameters():
                    param.requires_grad = False


def load_matching_checkpoint(model, checkpoint_path, device):
    loaded = torch.load(checkpoint_path, map_location=device)
    current = model.state_dict()
    compatible = {}
    skipped = []
    for key, value in loaded.items():
        if key in current and current[key].shape == value.shape:
            compatible[key] = value
        else:
            skipped.append(key)
    missing, unexpected = model.load_state_dict(compatible, strict=False)
    print(
        f"Warm-started from {checkpoint_path}: loaded={len(compatible)} "
        f"skipped_shape_or_missing={len(skipped)} missing_after_load={len(missing)} unexpected={len(unexpected)}"
    )
    if skipped:
        print("Skipped keys preview:", ", ".join(skipped[:8]))


def write_multi_test_report(out_dir, rows):
    fields = [
        "test_set",
        "samples",
        "force_unk",
        "ACC",
        "SEN",
        "SPE",
        "MCC",
        "AUC",
        "threshold",
        "TN",
        "FP",
        "FN",
        "TP",
    ]
    write_csv(out_dir / "multi_test_results.csv", rows, fields)
    lines = [
        "# ESM Fine-Tune Multi-Test Results",
        "",
        "| Test set | N | Force UNK | Threshold | ACC | SEN | SPE | MCC | AUC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['test_set']} | {row['samples']} | {row['force_unk']} | {row['threshold']:.4f} | "
            f"{row['ACC']:.4f} | {row['SEN']:.4f} | {row['SPE']:.4f} | "
            f"{row['MCC']:.4f} | {row['AUC']:.4f} |"
        )
    (out_dir / "multi_test_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    set_seed(args.seed)
    out_root = Path(args.out_dir)
    out_dir = out_root / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.prompt_mode == "none":
        args.prompt_fusion = "none"
        args.prompt_dropout = 0.0
        args.prompt_token_dropout = 0.0

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    train_rows = load_split_rows(args.windows_tsv, args.train_split, args.max_train_samples, args.seed)
    val_rows = load_split_rows(args.windows_tsv, args.val_split, args.max_val_samples, args.seed)
    test_rows = load_split_rows(args.windows_tsv, args.test_split, args.max_test_samples, args.seed)
    prompt_by_id = load_prompt_metadata(args.metadata_tsv, args.prompt_mode, shuffle_prompts=False, seed=args.seed)
    prompt_vocab = build_prompt_vocab(train_rows["sample_id"].tolist(), prompt_by_id, args.prompt_mode)
    (out_dir / "prompt_vocab.json").write_text(json.dumps(prompt_vocab, indent=2), encoding="utf-8")

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token is not None else tokenizer.cls_token
    esm = AutoModel.from_pretrained(args.model_dir, trust_remote_code=False)
    if args.freeze_esm:
        for param in esm.parameters():
            param.requires_grad = False
    else:
        apply_esm_training_policy(esm, args)

    train_ds, train_loader = make_loader(
        train_rows, prompt_by_id, prompt_vocab, args.prompt_mode, tokenizer, args.batch_size, shuffle=True
    )
    val_ds, val_loader = make_loader(
        val_rows, prompt_by_id, prompt_vocab, args.prompt_mode, tokenizer, args.batch_size, shuffle=False
    )
    test_ds, test_loader = make_loader(
        test_rows, prompt_by_id, prompt_vocab, args.prompt_mode, tokenizer, args.batch_size, shuffle=False
    )
    extra_eval = []
    for split in args.extra_eval_splits:
        rows = load_split_rows(args.windows_tsv, split, None, args.seed)
        ds, loader = make_loader(rows, prompt_by_id, prompt_vocab, args.prompt_mode, tokenizer, args.batch_size, shuffle=False)
        extra_eval.append((split, ds, loader))

    head = AblationPTMModel(
        num_prompts=len(prompt_vocab),
        prompt_mode=args.prompt_mode,
        prompt_fusion=args.prompt_fusion,
        use_site_relative=not args.disable_site_relative,
        use_center_attention=not args.disable_center_attention,
        use_prototype=not args.disable_prototype,
        use_evidence=not args.disable_evidence,
        prompt_dropout=args.prompt_dropout,
    )
    model = ESMFineTunePTM(esm, head).to(device)

    # Initialize LazyLinear layers before counting parameters and creating the optimizer.
    init_batch = next(iter(train_loader))
    with torch.no_grad():
        init_input_ids, init_attention_mask, init_special_mask, _init_y, init_prompt_id, _init_sid = init_batch
        _ = model(
            init_input_ids[:1].to(device),
            init_attention_mask[:1].to(device),
            init_special_mask[:1].to(device),
            init_prompt_id[:1].to(device),
        )

    if args.init_checkpoint:
        load_matching_checkpoint(model, args.init_checkpoint, device)

    head_params = [p for p in model.head.parameters() if p.requires_grad]
    esm_params = [p for p in model.esm.parameters() if p.requires_grad]
    optimizer = optim.AdamW(
        [
            {"params": esm_params, "lr": args.lr_esm},
            {"params": head_params, "lr": args.lr_head},
        ],
        weight_decay=args.weight_decay,
    )
    updates_per_epoch = math.ceil(len(train_loader) / max(1, args.grad_accum_steps))
    total_updates = max(1, updates_per_epoch * args.epochs)
    warmup_updates = int(total_updates * args.warmup_ratio)
    scheduler = None
    if args.scheduler == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_updates,
            num_training_steps=total_updates,
        )
    scaler = torch.amp.GradScaler("cuda", enabled=(args.fp16 and device.type == "cuda"))

    config = {
        "run_name": args.run_name,
        "model_dir": args.model_dir,
        "init_checkpoint": args.init_checkpoint,
        "freeze_esm": args.freeze_esm,
        "gradient_checkpointing": args.gradient_checkpointing,
        "freeze_esm_embeddings": args.freeze_esm_embeddings,
        "freeze_esm_first_n_layers": args.freeze_esm_first_n_layers,
        "train_split": args.train_split,
        "val_split": args.val_split,
        "test_split": args.test_split,
        "extra_eval_splits": args.extra_eval_splits,
        "force_unk_splits": args.force_unk_splits,
        "prompt_mode": args.prompt_mode,
        "prompt_fusion": args.prompt_fusion,
        "prompt_dropout": args.prompt_dropout,
        "prompt_token_dropout": args.prompt_token_dropout,
        "use_site_relative": not args.disable_site_relative,
        "use_center_attention": not args.disable_center_attention,
        "use_prototype": not args.disable_prototype,
        "use_evidence": not args.disable_evidence,
        "lambda_sparse": args.lambda_sparse,
        "lambda_div": args.lambda_div,
        "lambda_evidence": args.lambda_evidence,
        "lambda_width": args.lambda_width,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr_head": args.lr_head,
        "lr_esm": args.lr_esm,
        "grad_accum_steps": args.grad_accum_steps,
        "scheduler": args.scheduler,
        "warmup_ratio": args.warmup_ratio,
        "early_stop_patience": args.early_stop_patience,
        "updates_per_epoch": updates_per_epoch,
        "total_updates": total_updates,
        "warmup_updates": warmup_updates,
        "seed": args.seed,
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "test_samples": len(test_ds),
        "head_trainable_params": int(sum(p.numel() for p in head_params)),
        "esm_trainable_params": int(sum(p.numel() for p in esm_params)),
        "total_trainable_params": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(json.dumps(config, indent=2))

    best_mcc = -1.0
    best_epoch = 0
    epochs_since_improvement = 0
    best_path = out_dir / "best.pt"
    epoch_rows = []
    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            scaler,
            device,
            args,
            prompt_vocab["<unk>"],
            progress_desc=f"train {epoch}/{args.epochs}",
        )
        val = eval_model(model, val_loader, device)
        epoch_rows.append({"epoch": epoch, "loss": loss, **{f"val_{k}": v for k, v in val.items()}})
        print(
            f"Epoch {epoch:3d}/{args.epochs} | Loss={loss:.4f} | "
            f"VAL_ACC={val['ACC']:.4f} | VAL_SEN={val['SEN']:.4f} | VAL_SPE={val['SPE']:.4f} | "
            f"VAL_MCC={val['MCC']:.4f} | VAL_AUC={val['AUC']:.4f}"
        )
        if val["MCC"] > best_mcc:
            best_mcc = val["MCC"]
            best_epoch = epoch
            epochs_since_improvement = 0
            torch.save(model.state_dict(), best_path)
        else:
            epochs_since_improvement += 1
            if args.early_stop_patience > 0 and epochs_since_improvement >= args.early_stop_patience:
                print(f"Early stopping at epoch {epoch}; best_epoch={best_epoch} best_val_mcc={best_mcc:.4f}")
                break

    model.load_state_dict(torch.load(best_path, map_location=device))
    val_true, val_prob = collect_predictions(model, val_loader, device)
    calibrated_threshold, calibrated_val_mcc = best_mcc_threshold(val_true, val_prob)
    print(
        f"Validation-calibrated threshold={calibrated_threshold:.4f}; "
        f"val_mcc_at_threshold={calibrated_val_mcc:.4f}"
    )
    test = eval_model(model, test_loader, device)
    test_calibrated = eval_model(model, test_loader, device, threshold=calibrated_threshold)
    unk_test = None
    unk_test_calibrated = None
    if args.prompt_mode != "none":
        unk_test = eval_model(model, test_loader, device, force_prompt_id=prompt_vocab["<unk>"])
        unk_test_calibrated = eval_model(
            model,
            test_loader,
            device,
            force_prompt_id=prompt_vocab["<unk>"],
            threshold=calibrated_threshold,
        )

    multi_rows = []
    force_unk_splits = set(args.force_unk_splits)
    for split_name, ds, loader in [(args.test_split, test_ds, test_loader), *extra_eval]:
        force_unk = args.prompt_mode != "none" and split_name in force_unk_splits
        metrics = eval_model(
            model,
            loader,
            device,
            force_prompt_id=prompt_vocab["<unk>"] if force_unk else None,
            threshold=calibrated_threshold,
        )
        multi_rows.append(
            {
                "test_set": split_name,
                "samples": len(ds),
                "force_unk": int(force_unk),
                **metrics,
            }
        )
        print(
            f"[EVAL] {split_name} force_unk={int(force_unk)} "
            f"threshold={calibrated_threshold:.4f} mcc={metrics['MCC']:.4f} auc={metrics['AUC']:.4f}"
        )
    write_multi_test_report(out_dir, multi_rows)

    result = {
        **config,
        "best_epoch": best_epoch,
        "best_val_mcc": float(best_mcc),
        "calibrated_threshold": float(calibrated_threshold),
        "calibrated_val_mcc": float(calibrated_val_mcc),
        **{f"test_{k}": v for k, v in test.items()},
        **{f"calibrated_test_{k}": v for k, v in test_calibrated.items()},
    }
    if unk_test is not None:
        result.update({f"unk_test_{k}": v for k, v in unk_test.items()})
    if unk_test_calibrated is not None:
        result.update({f"calibrated_unk_test_{k}": v for k, v in unk_test_calibrated.items()})
    result["multi_test"] = multi_rows
    (out_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_csv(
        out_dir / "epoch_metrics.csv",
        epoch_rows,
        [
            "epoch",
            "loss",
            "val_ACC",
            "val_SEN",
            "val_SPE",
            "val_MCC",
            "val_AUC",
            "val_threshold",
            "val_TN",
            "val_FP",
            "val_FN",
            "val_TP",
        ],
    )
    print(
        f"Done. best_epoch={best_epoch} best_val_mcc={best_mcc:.4f} "
        f"test_mcc={test['MCC']:.4f} test_auc={test['AUC']:.4f} "
        f"calibrated_test_mcc={test_calibrated['MCC']:.4f}"
    )
    if unk_test is not None:
        print(f"UNK test: mcc={unk_test['MCC']:.4f} auc={unk_test['AUC']:.4f}")


if __name__ == "__main__":
    main()

