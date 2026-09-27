import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModel, AutoTokenizer

from ptm_head import AblationPTMModel
from train_our_model import ESMFineTunePTM, clean_seq


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = PROJECT_DIR / "models" / "esm650m_prompt_no_mlm_final"
DEFAULT_MODEL_DIR = "facebook/esm2_t33_650M_UR50D"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run phosphosite inference with a fine-tuned prompt-aware ESM PTM checkpoint."
    )
    parser.add_argument("--run_dir", default=str(DEFAULT_RUN_DIR), help="Directory containing best.pt/config.json/prompt_vocab.json.")
    parser.add_argument("--input_tsv", required=True, help="TSV with at least a window column.")
    parser.add_argument("--out_tsv", required=True)
    parser.add_argument("--model_dir", default=None, help="Base ESM model path or HF model id. Overrides config model_dir.")
    parser.add_argument(
        "--load_base_weights",
        action="store_true",
        help="Load pretrained base ESM weights before loading best.pt. By default only config/tokenizer are loaded.",
    )
    parser.add_argument("--window_col", default="window")
    parser.add_argument("--sample_id_col", default="sample_id")
    parser.add_argument(
        "--prompt_col",
        default=None,
        help="Prompt column. Defaults to the configured prompt_mode, e.g. taxon_group.",
    )
    parser.add_argument("--force_unk", action="store_true", help="Force all samples to use the <unk> prompt.")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=None, help="Defaults to calibrated_threshold from result.json if present.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


class InferenceDataset(Dataset):
    def __init__(self, rows, tokenizer, prompt_vocab, prompt_mode, prompt_col, args):
        self.rows = rows.reset_index(drop=True)
        self.ids = (
            self.rows[args.sample_id_col].astype(str).tolist()
            if args.sample_id_col in self.rows.columns
            else [f"sample_{i + 1:06d}" for i in range(len(self.rows))]
        )
        sequences = [clean_seq(seq) for seq in self.rows[args.window_col].astype(str).tolist()]
        prompts = []
        for _, row in self.rows.iterrows():
            if prompt_mode == "none" or args.force_unk:
                prompts.append(prompt_vocab["<unk>"])
                continue
            label = str(row.get(prompt_col, "other"))
            prompts.append(prompt_vocab.get(label, prompt_vocab["<unk>"]))
        self.prompts = prompts
        enc = tokenizer(
            sequences,
            return_tensors="np",
            padding=True,
            truncation=False,
            add_special_tokens=True,
            return_special_tokens_mask=True,
        )
        self.input_ids = enc["input_ids"]
        self.attention_mask = enc["attention_mask"]
        self.special_tokens_mask = enc["special_tokens_mask"]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        return {
            "sample_id": self.ids[idx],
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "special_tokens_mask": self.special_tokens_mask[idx],
            "prompt_id": self.prompts[idx],
        }


def collate(batch):
    return {
        "sample_id": [item["sample_id"] for item in batch],
        "input_ids": torch.tensor(np.stack([item["input_ids"] for item in batch]), dtype=torch.long),
        "attention_mask": torch.tensor(np.stack([item["attention_mask"] for item in batch]), dtype=torch.long),
        "special_tokens_mask": torch.tensor(np.stack([item["special_tokens_mask"] for item in batch]), dtype=torch.long),
        "prompt_id": torch.tensor([item["prompt_id"] for item in batch], dtype=torch.long),
    }


def resolve_model_dir(args, config, run_dir):
    candidates = []
    if args.model_dir:
        candidates.append(args.model_dir)
    configured = str(config.get("model_dir", ""))
    if configured:
        candidates.append(configured)
    for candidate in candidates:
        path = Path(candidate)
        if path.is_absolute() and path.exists():
            return str(path)
        for base in (Path.cwd(), run_dir, run_dir.parent, PROJECT_DIR):
            resolved = base / path
            if resolved.exists():
                return str(resolved)
    configured = str(config.get("model_dir", ""))
    return DEFAULT_MODEL_DIR


def load_model(run_dir, model_dir, config, prompt_vocab, device):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token is not None else tokenizer.cls_token
    if config.get("_load_base_weights", False):
        esm = AutoModel.from_pretrained(model_dir, trust_remote_code=False)
    else:
        esm_config = AutoConfig.from_pretrained(model_dir, trust_remote_code=False)
        esm = AutoModel.from_config(esm_config, trust_remote_code=False)
    head = AblationPTMModel(
        num_prompts=len(prompt_vocab),
        prompt_mode=config["prompt_mode"],
        prompt_fusion=config["prompt_fusion"],
        use_site_relative=config["use_site_relative"],
        use_center_attention=config["use_center_attention"],
        use_prototype=config["use_prototype"],
        use_evidence=config["use_evidence"],
        prompt_dropout=0.0,
    )
    model = ESMFineTunePTM(esm, head).to(device)
    return tokenizer, model


@torch.no_grad()
def initialize_lazy_layers(model, dataset, device):
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate)
    batch = next(iter(loader))
    _ = model(
        batch["input_ids"].to(device),
        batch["attention_mask"].to(device),
        batch["special_tokens_mask"].to(device),
        batch["prompt_id"].to(device),
    )


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    rows = []
    for batch in loader:
        outputs = model(
            batch["input_ids"].to(device),
            batch["attention_mask"].to(device),
            batch["special_tokens_mask"].to(device),
            batch["prompt_id"].to(device),
        )
        probs = outputs["prob"].detach().cpu().tolist()
        for sample_id, prompt_id, prob in zip(batch["sample_id"], batch["prompt_id"].tolist(), probs):
            rows.append({"sample_id": sample_id, "prompt_id": prompt_id, "probability": float(prob)})
    return rows


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    config = read_json(run_dir / "config.json")
    prompt_vocab = read_json(run_dir / "prompt_vocab.json")
    result_path = run_dir / "result.json"
    result = read_json(result_path) if result_path.exists() else {}
    threshold = args.threshold if args.threshold is not None else float(result.get("calibrated_threshold", 0.5))

    input_df = pd.read_csv(args.input_tsv, sep="\t")
    if args.window_col not in input_df.columns:
        raise ValueError(f"Missing required window column: {args.window_col}")
    prompt_mode = config["prompt_mode"]
    prompt_col = args.prompt_col or prompt_mode
    if prompt_mode != "none" and not args.force_unk and prompt_col not in input_df.columns:
        print(f"WARNING: prompt column '{prompt_col}' not found; using <unk> for all samples.")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    model_dir = resolve_model_dir(args, config, run_dir)
    config["_load_base_weights"] = bool(args.load_base_weights)
    tokenizer, model = load_model(run_dir, model_dir, config, prompt_vocab, device)
    dataset = InferenceDataset(input_df, tokenizer, prompt_vocab, prompt_mode, prompt_col, args)
    initialize_lazy_layers(model, dataset, device)
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    predictions = predict(model, loader, device)

    pred_df = pd.DataFrame(predictions)
    pred_df["prediction"] = (pred_df["probability"] >= threshold).astype(int)
    pred_df["threshold"] = threshold
    out_df = input_df.copy()
    if args.sample_id_col not in out_df.columns:
        out_df.insert(0, args.sample_id_col, pred_df["sample_id"])
    out_df["phosphosite_probability"] = pred_df["probability"]
    out_df["prediction"] = pred_df["prediction"]
    out_df["threshold"] = pred_df["threshold"]
    out_path = Path(args.out_tsv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, sep="\t", index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"Wrote predictions: {out_path}")


if __name__ == "__main__":
    main()

