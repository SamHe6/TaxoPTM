import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import confusion_matrix, matthews_corrcoef, roc_auc_score
from torch.utils.data import DataLoader, Dataset

from model import (
    CenterConditionedAttention,
    EvidenceSelector,
    MotifPrototypeBank,
    SiteRelativeEncoder,
)


def prototype_sparsity_loss(scores: torch.Tensor) -> torch.Tensor:
    entropy = -(scores * (scores.clamp_min(1e-8).log())).sum(dim=-1)
    return entropy.mean()


def prototype_diversity_loss(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits, dim=-1)
    corr = torch.matmul(probs.t(), probs) / probs.size(0)
    ident = torch.eye(corr.size(0), device=corr.device)
    return ((corr - ident) ** 2).mean()


def evidence_repulsion_loss(mu: torch.Tensor, sigma: torch.Tensor, span_weights: torch.Tensor) -> torch.Tensor:
    batch_size = mu.size(0)
    num_slots = mu.size(1)
    if num_slots <= 1:
        return torch.zeros((), device=mu.device)
    mu_diff = (mu.unsqueeze(2) - mu.unsqueeze(1)).abs()
    sigma_scale = sigma.unsqueeze(2) + sigma.unsqueeze(1)
    proximity = torch.exp(-mu_diff / sigma_scale.clamp_min(1e-4))
    overlap = torch.einsum("bkn,bjn->bkj", span_weights, span_weights)
    ident = torch.eye(num_slots, device=mu.device).unsqueeze(0)
    pair_mask = 1.0 - ident
    denom = (batch_size * pair_mask.sum()).clamp_min(1.0)
    proximity_loss = (proximity * pair_mask).sum() / denom
    overlap_loss = (overlap * pair_mask).sum() / denom
    return proximity_loss + overlap_loss


def evidence_width_loss(
    sigma: torch.Tensor,
    sigma_prior: torch.Tensor,
    tolerance: float = 0.03,
    global_min: float = 0.08,
    global_max: float = 0.45,
) -> torch.Tensor:
    lower = sigma_prior - tolerance
    upper = sigma_prior + tolerance
    prior_band = F.relu(lower - sigma) + F.relu(sigma - upper)
    global_band = F.relu(global_min - sigma) + F.relu(sigma - global_max)
    prior_center = (sigma - sigma_prior).pow(2)
    return prior_band.mean() + global_band.mean() + prior_center.mean()


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_DIR / "data" / "processed" / "open_phospho_clean_cdhit"
DEFAULT_META = PROJECT_DIR / "data" / "processed" / "species_benchmarks" / "all_samples_with_taxon_group.tsv"
DEFAULT_OUT_DIR = PROJECT_DIR / "outputs" / "our_model_head"
DEFAULT_SEED = 42


def parse_args():
    parser = argparse.ArgumentParser(description="Train the prompt-aware PTM head.")
    parser.add_argument("--train_npz", default=str(DEFAULT_DATA_DIR / "train_esm_embeddings.npz"))
    parser.add_argument("--val_npz", default=str(DEFAULT_DATA_DIR / "val_esm_embeddings.npz"))
    parser.add_argument("--test_npz", default=str(DEFAULT_DATA_DIR / "test_esm_embeddings.npz"))
    parser.add_argument("--metadata_tsv", default=str(DEFAULT_META))
    parser.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--run_name", default="our_model_head")

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
    parser.add_argument("--shuffle_prompts", action="store_true")

    parser.add_argument("--disable_site_relative", action="store_true")
    parser.add_argument("--disable_center_attention", action="store_true")
    parser.add_argument("--disable_prototype", action="store_true")
    parser.add_argument("--disable_evidence", action="store_true")

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")

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


def to_broad_group(taxon_group):
    if taxon_group in {"human", "mouse", "animal_other"}:
        return "animal"
    if taxon_group in {"yeast", "fungi_other"}:
        return "fungi"
    if taxon_group in {"plant", "protist", "virus", "bacteria_or_archaea"}:
        return taxon_group
    return "other"


def load_prompt_metadata(metadata_tsv, prompt_mode, shuffle_prompts=False, seed=42):
    if prompt_mode == "none":
        return {}

    meta = pd.read_csv(metadata_tsv, sep="\t")
    if prompt_mode == "tax_id":
        meta["prompt"] = "taxid_" + meta["tax_id"].fillna("unknown").astype(str)
    elif prompt_mode == "taxon_group":
        meta["prompt"] = meta["taxon_group"].fillna("other").astype(str)
    elif prompt_mode == "broad_group":
        meta["prompt"] = meta["taxon_group"].fillna("other").astype(str).map(to_broad_group)
    else:
        raise ValueError(f"Unknown prompt_mode: {prompt_mode}")

    sample_ids = meta["sample_id"].astype(str).tolist()
    prompts = meta["prompt"].astype(str).tolist()
    if shuffle_prompts:
        rng = random.Random(seed)
        rng.shuffle(prompts)
    return dict(zip(sample_ids, prompts))


def build_prompt_vocab(train_npz, prompt_by_id, prompt_mode):
    if prompt_mode == "none":
        return {"<unk>": 0}
    with np.load(train_npz, allow_pickle=True) as data:
        train_ids = [str(x) for x in data["ids"]]
    labels = sorted({prompt_by_id.get(sample_id, "other") for sample_id in train_ids})
    vocab = {"<unk>": 0}
    for label in labels:
        if label not in vocab:
            vocab[label] = len(vocab)
    return vocab


class AblationNPZDataset(Dataset):
    def __init__(self, npz_path, prompt_by_id, prompt_vocab, prompt_mode):
        data = np.load(npz_path, allow_pickle=True)
        self.emb = data["emb"].astype(np.float32)
        self.mask = data["mask"].astype(np.int64)
        self.labels = data["labels"].astype(np.float32)
        self.ids = np.asarray(data["ids"], dtype=object)
        prompts = []
        for sample_id in self.ids:
            if prompt_mode == "none":
                prompts.append(0)
            else:
                label = prompt_by_id.get(str(sample_id), "other")
                prompts.append(prompt_vocab.get(label, prompt_vocab["<unk>"]))
        self.prompts = np.asarray(prompts, dtype=np.int64)

    def __len__(self):
        return int(len(self.labels))

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.emb[idx]),
            torch.from_numpy(self.mask[idx]),
            torch.tensor(self.labels[idx], dtype=torch.float32),
            torch.tensor(self.prompts[idx], dtype=torch.long),
            str(self.ids[idx]),
        )


def collate_fn(batch):
    x_emb, x_mask, y, prompt_id, sid = zip(*batch)
    return (
        torch.stack(x_emb, dim=0),
        torch.stack(x_mask, dim=0),
        torch.stack(y, dim=0),
        torch.stack(prompt_id, dim=0),
        list(sid),
    )


class PlainTokenEncoder(nn.Module):
    def __init__(self, units):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LazyLinear(units),
            nn.GELU(),
            nn.LayerNorm(units),
        )

    def forward(self, x_emb, x_mask):
        valid_mask = x_mask.bool()
        lengths = x_mask.sum(dim=1).long().clamp_min(1)
        centers = (lengths // 2).long()
        bsz, seq_len, _ = x_emb.shape
        h = self.proj(x_emb) * valid_mask.unsqueeze(-1)
        bucket = torch.zeros((bsz, seq_len), device=x_emb.device, dtype=torch.long)
        idx = torch.arange(seq_len, device=x_emb.device).view(1, seq_len).expand(bsz, seq_len)
        rel_pos = idx - centers.view(-1, 1)
        return {
            "h": h,
            "valid_mask": valid_mask,
            "centers": centers,
            "rel_pos": rel_pos,
            "bucket": bucket,
        }


class AblationPTMModel(nn.Module):
    def __init__(
        self,
        num_prompts,
        prompt_mode="taxon_group",
        prompt_fusion="add_concat",
        use_site_relative=True,
        use_center_attention=True,
        use_prototype=True,
        use_evidence=True,
        units=256,
        heads=4,
        num_buckets=16,
        max_distance=32,
        num_prototypes=12,
        num_evidence=3,
        span_radius=2,
        dropout=0.1,
        prompt_dropout=0.2,
    ):
        super().__init__()
        self.prompt_enabled = prompt_mode != "none" and prompt_fusion != "none"
        self.prompt_fusion = prompt_fusion if self.prompt_enabled else "none"
        self.use_center_attention = use_center_attention
        self.use_prototype = use_prototype
        self.use_evidence = use_evidence
        self.heads = heads
        self.units = units

        if use_site_relative:
            self.encoder = SiteRelativeEncoder(units=units, num_buckets=num_buckets, max_distance=max_distance)
        else:
            self.encoder = PlainTokenEncoder(units=units)

        if use_center_attention:
            self.center_attn = CenterConditionedAttention(
                units=units,
                heads=heads,
                num_buckets=num_buckets,
                dropout=dropout,
            )
        else:
            self.center_attn = None

        if use_prototype:
            self.prototype_bank = MotifPrototypeBank(units=units, num_prototypes=num_prototypes)
        else:
            self.prototype_bank = None

        if use_evidence:
            self.evidence_selector = EvidenceSelector(
                units=units,
                num_evidence=num_evidence,
                span_radius=span_radius,
            )
        else:
            self.evidence_selector = None

        if self.prompt_enabled:
            self.prompt_emb = nn.Embedding(num_prompts, units)
            self.prompt_norm = nn.LayerNorm(units)
            self.prompt_dropout = nn.Dropout(prompt_dropout)
        else:
            self.prompt_emb = None
            self.prompt_norm = None
            self.prompt_dropout = None

        fused_units = units * 2  # center + context
        if use_prototype:
            fused_units += units
        if use_evidence:
            fused_units += units
        if self.prompt_fusion in {"add_concat", "concat_only"}:
            fused_units += units

        self.classifier = nn.Sequential(
            nn.Linear(fused_units, units),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(units, 1),
        )

    def build_prompt_vec(self, prompt_id):
        if not self.prompt_enabled:
            return None
        prompt_vec = self.prompt_norm(self.prompt_emb(prompt_id))
        return self.prompt_dropout(prompt_vec)

    def forward(self, x_emb, x_mask, prompt_id):
        enc = self.encoder(x_emb, x_mask)
        h = enc["h"]
        valid_mask = enc["valid_mask"]
        centers = enc["centers"]

        prompt_vec = self.build_prompt_vec(prompt_id)

        center_repr = h.gather(1, centers.view(-1, 1, 1).expand(-1, 1, h.size(-1))).squeeze(1)
        if self.use_center_attention:
            attn_out = self.center_attn(h, valid_mask, centers, enc["bucket"])
            context_vec = attn_out["context_vec"]
            attn_weights = attn_out["attn_weights"]
            center_repr = attn_out["center_repr"]
        else:
            context_vec = center_repr
            uniform = valid_mask.float()
            uniform = uniform / uniform.sum(dim=1, keepdim=True).clamp_min(1.0)
            attn_weights = uniform.unsqueeze(1).expand(-1, self.heads, -1)

        conditioned_context = context_vec
        if prompt_vec is not None and self.prompt_fusion in {"add_concat", "add_only"}:
            conditioned_context = context_vec + prompt_vec

        outputs = {
            "centers": centers,
            "rel_pos": enc["rel_pos"],
            "attn_weights": attn_weights,
        }
        fused = [center_repr, context_vec]

        if self.use_prototype:
            proto_out = self.prototype_bank(conditioned_context)
            outputs["prototype_logits"] = proto_out["prototype_logits"]
            outputs["prototype_scores"] = proto_out["prototype_scores"]
            fused.append(proto_out["proto_summary"])

        if self.use_evidence:
            evidence_out = self.evidence_selector(h, valid_mask, centers, conditioned_context, attn_weights)
            outputs["evidence_positions"] = evidence_out["evidence_positions"]
            outputs["evidence_weights"] = evidence_out["evidence_weights"]
            outputs["evidence_mu"] = evidence_out["evidence_mu"]
            outputs["evidence_sigma"] = evidence_out["evidence_sigma"]
            outputs["evidence_sigma_prior"] = evidence_out["evidence_sigma_prior"]
            outputs["span_weights"] = evidence_out["span_weights"]
            fused.append(evidence_out["evidence_summary"])

        if prompt_vec is not None and self.prompt_fusion in {"add_concat", "concat_only"}:
            fused.append(prompt_vec)

        logits = self.classifier(torch.cat(fused, dim=-1)).squeeze(-1)
        outputs["logits"] = logits
        outputs["prob"] = torch.sigmoid(torch.clamp(logits, -30, 30))
        return outputs


def maybe_drop_prompt_ids(prompt_id, unk_id, drop_prob):
    if drop_prob <= 0:
        return prompt_id
    keep = torch.rand_like(prompt_id.float()) >= drop_prob
    unk = torch.full_like(prompt_id, int(unk_id))
    return torch.where(keep, prompt_id, unk)


def compute_loss(outputs, y, args):
    cls_loss = F.binary_cross_entropy_with_logits(outputs["logits"], y.float())
    total = cls_loss
    parts = {"cls": float(cls_loss.detach())}

    if "prototype_scores" in outputs and args.lambda_sparse:
        sparse = prototype_sparsity_loss(outputs["prototype_scores"])
        total = total + args.lambda_sparse * sparse
        parts["sparse"] = float(sparse.detach())
    else:
        parts["sparse"] = 0.0

    if "prototype_logits" in outputs and args.lambda_div:
        div = prototype_diversity_loss(outputs["prototype_logits"])
        total = total + args.lambda_div * div
        parts["div"] = float(div.detach())
    else:
        parts["div"] = 0.0

    if "span_weights" in outputs and args.lambda_evidence:
        evidence = evidence_repulsion_loss(outputs["evidence_mu"], outputs["evidence_sigma"], outputs["span_weights"])
        total = total + args.lambda_evidence * evidence
        parts["evidence"] = float(evidence.detach())
    else:
        parts["evidence"] = 0.0

    if "evidence_sigma_prior" in outputs and args.lambda_width:
        width = evidence_width_loss(outputs["evidence_sigma"], outputs["evidence_sigma_prior"])
        total = total + args.lambda_width * width
        parts["width"] = float(width.detach())
    else:
        parts["width"] = 0.0

    parts["total"] = float(total.detach())
    return total, parts


def metric_dict(y_true, y_prob):
    y_pred = [1 if v >= 0.5 else 0 for v in y_prob]
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
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


@torch.no_grad()
def eval_model(model, loader, device, force_prompt_id=None):
    model.eval()
    y_true, y_prob = [], []
    for x_emb, x_mask, y, prompt_id, _sid in loader:
        prompt_id = prompt_id.to(device)
        if force_prompt_id is not None:
            prompt_id = torch.full_like(prompt_id, int(force_prompt_id))
        outputs = model(x_emb.to(device), x_mask.to(device), prompt_id)
        y_prob.extend(outputs["prob"].detach().cpu().numpy().tolist())
        y_true.extend(y.numpy().astype(int).tolist())
    return metric_dict(y_true, y_prob)


def train_one_epoch(model, loader, optimizer, device, args, unk_id):
    model.train()
    total = 0.0
    n = 0
    for x_emb, x_mask, y, prompt_id, _sid in loader:
        x_emb = x_emb.to(device)
        x_mask = x_mask.to(device)
        y = y.to(device)
        prompt_id = maybe_drop_prompt_ids(prompt_id.to(device), unk_id, args.prompt_token_dropout)

        optimizer.zero_grad()
        outputs = model(x_emb, x_mask, prompt_id)
        loss, _parts = compute_loss(outputs, y, args)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total += float(loss.detach()) * y.size(0)
        n += y.size(0)
    return total / max(1, n)


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    set_seed(args.seed)
    out_root = Path(args.out_dir)
    out_dir = out_root / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    if args.prompt_mode == "none":
        args.prompt_fusion = "none"
        args.prompt_token_dropout = 0.0
        args.prompt_dropout = 0.0

    prompt_by_id = load_prompt_metadata(
        args.metadata_tsv,
        args.prompt_mode,
        shuffle_prompts=args.shuffle_prompts,
        seed=args.seed,
    )
    prompt_vocab = build_prompt_vocab(args.train_npz, prompt_by_id, args.prompt_mode)
    (out_dir / "prompt_vocab.json").write_text(json.dumps(prompt_vocab, indent=2), encoding="utf-8")

    train_ds = AblationNPZDataset(args.train_npz, prompt_by_id, prompt_vocab, args.prompt_mode)
    val_ds = AblationNPZDataset(args.val_npz, prompt_by_id, prompt_vocab, args.prompt_mode)
    test_ds = AblationNPZDataset(args.test_npz, prompt_by_id, prompt_vocab, args.prompt_mode)
    print(f"Run: {args.run_name}")
    print(f"Train samples: {len(train_ds)}")
    print(f"Validation samples: {len(val_ds)}")
    print(f"Final test samples: {len(test_ds)}")
    print(f"Prompt mode: {args.prompt_mode}; fusion={args.prompt_fusion}; vocab_size={len(prompt_vocab)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)

    model = AblationPTMModel(
        num_prompts=len(prompt_vocab),
        prompt_mode=args.prompt_mode,
        prompt_fusion=args.prompt_fusion,
        use_site_relative=not args.disable_site_relative,
        use_center_attention=not args.disable_center_attention,
        use_prototype=not args.disable_prototype,
        use_evidence=not args.disable_evidence,
        prompt_dropout=args.prompt_dropout,
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    config = {
        "run_name": args.run_name,
        "prompt_mode": args.prompt_mode,
        "prompt_fusion": args.prompt_fusion,
        "prompt_dropout": args.prompt_dropout,
        "prompt_token_dropout": args.prompt_token_dropout,
        "shuffle_prompts": args.shuffle_prompts,
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
        "lr": args.lr,
        "seed": args.seed,
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    best_mcc = -1.0
    best_epoch = 0
    best_path = out_dir / "best.pt"
    epoch_rows = []
    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, device, args, prompt_vocab["<unk>"])
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
            torch.save(model.state_dict(), best_path)

    model.load_state_dict(torch.load(best_path, map_location=device))
    test = eval_model(model, test_loader, device)
    unk_test = None
    if args.prompt_mode != "none":
        unk_test = eval_model(model, test_loader, device, force_prompt_id=prompt_vocab["<unk>"])

    result = {
        **config,
        "best_epoch": best_epoch,
        "best_val_mcc": float(best_mcc),
        **{f"test_{k}": v for k, v in test.items()},
    }
    if unk_test is not None:
        result.update({f"unk_test_{k}": v for k, v in unk_test.items()})

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
            "val_TN",
            "val_FP",
            "val_FN",
            "val_TP",
        ],
    )

    print(
        f"Done. best_epoch={best_epoch} best_val_mcc={best_mcc:.4f} "
        f"test_mcc={test['MCC']:.4f} test_auc={test['AUC']:.4f}"
    )
    if unk_test is not None:
        print(f"UNK test: mcc={unk_test['MCC']:.4f} auc={unk_test['AUC']:.4f}")


if __name__ == "__main__":
    main()


