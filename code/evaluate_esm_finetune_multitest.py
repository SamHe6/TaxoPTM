import argparse
import csv
import json
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

from ptm_head import AblationPTMModel, load_prompt_metadata
from train_our_model import (
    ESMFineTunePTM,
    best_mcc_threshold,
    collect_predictions,
    eval_model,
    load_split_rows,
    make_loader,
)


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_DIR / "data" / "processed" / "open_phospho_species_balanced_eval"
DEFAULT_WINDOWS = DEFAULT_DATA_DIR / "phospho_windows_41aa.tsv"
DEFAULT_META = DEFAULT_DATA_DIR / "all_samples_with_taxon_group.tsv"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def resolve_model_dir(config, run_dir):
    configured = str(config.get("model_dir", ""))
    path = Path(configured)
    if path.is_absolute() and path.exists():
        return str(path)
    for base in (Path.cwd(), run_dir, run_dir.parent, PROJECT_DIR):
        resolved = base / path
        if resolved.exists():
            return str(resolved)
    return configured


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned ESM PTM checkpoint on multiple split names.")
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--windows_tsv", default=str(DEFAULT_WINDOWS))
    parser.add_argument("--metadata_tsv", default=str(DEFAULT_META))
    parser.add_argument("--splits", nargs="+", default=["test_balanced", "test_imbalanced", "ood"])
    parser.add_argument("--force_unk_splits", nargs="*", default=["ood"])
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--threshold", choices=["fixed_0p5", "val_mcc"], default="val_mcc")
    parser.add_argument(
        "--load_base_weights",
        action="store_true",
        help="Load pretrained ESM weights before best.pt. Default uses config-only initialization for faster local evaluation.",
    )
    parser.add_argument("--out_csv", default=None)
    parser.add_argument("--out_md", default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--no_progress", action="store_true", help="Disable tqdm progress bars.")
    return parser.parse_args()


def write_outputs(csv_path, md_path, rows):
    fields = [
        "test_set",
        "samples",
        "force_unk",
        "threshold",
        "ACC",
        "SEN",
        "SPE",
        "MCC",
        "AUC",
        "TN",
        "FP",
        "FN",
        "TP",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Fine-Tuned ESM Multi-Test Evaluation",
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
    Path(md_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    config = read_json(run_dir / "config.json")
    prompt_vocab = read_json(run_dir / "prompt_vocab.json")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    prompt_mode = config["prompt_mode"]
    prompt_by_id = load_prompt_metadata(args.metadata_tsv, prompt_mode, shuffle_prompts=False, seed=config.get("seed", 42))

    model_dir = resolve_model_dir(config, run_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token is not None else tokenizer.cls_token
    if args.load_base_weights:
        esm = AutoModel.from_pretrained(model_dir, trust_remote_code=False)
    else:
        esm_config = AutoConfig.from_pretrained(model_dir, trust_remote_code=False)
        esm = AutoModel.from_config(esm_config, trust_remote_code=False)
    head = AblationPTMModel(
        num_prompts=len(prompt_vocab),
        prompt_mode=prompt_mode,
        prompt_fusion=config["prompt_fusion"],
        use_site_relative=config["use_site_relative"],
        use_center_attention=config["use_center_attention"],
        use_prototype=config["use_prototype"],
        use_evidence=config["use_evidence"],
        prompt_dropout=0.0,
    )
    model = ESMFineTunePTM(esm, head).to(device)

    val_rows = load_split_rows(args.windows_tsv, config.get("val_split", "val"), None, config.get("seed", 42))
    _val_ds, val_loader = make_loader(val_rows, prompt_by_id, prompt_vocab, prompt_mode, tokenizer, args.batch_size, False)

    # Initialize LazyLinear layers before loading weights.
    init_batch = next(iter(val_loader))
    with torch.no_grad():
        input_ids, attention_mask, special_mask, _y, prompt_id, _sid = init_batch
        _ = model(
            input_ids[:1].to(device),
            attention_mask[:1].to(device),
            special_mask[:1].to(device),
            prompt_id[:1].to(device),
        )
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device))

    threshold = 0.5
    if args.threshold == "val_mcc":
        val_true, val_prob = collect_predictions(
            model,
            val_loader,
            device,
            progress_desc=None if args.no_progress else "val threshold",
        )
        threshold, val_mcc = best_mcc_threshold(val_true, val_prob)
        print(f"[VAL] threshold={threshold:.4f} mcc={val_mcc:.4f}")

    rows = []
    force_unk_splits = set(args.force_unk_splits or [])
    for split in args.splits:
        test_rows = load_split_rows(args.windows_tsv, split, None, config.get("seed", 42))
        ds, loader = make_loader(test_rows, prompt_by_id, prompt_vocab, prompt_mode, tokenizer, args.batch_size, False)
        force_unk = prompt_mode != "none" and split in force_unk_splits
        metrics = eval_model(
            model,
            loader,
            device,
            force_prompt_id=prompt_vocab["<unk>"] if force_unk else None,
            threshold=threshold,
            progress_desc=None if args.no_progress else split,
        )
        rows.append({"test_set": split, "samples": len(ds), "force_unk": int(force_unk), **metrics})
        print(f"[RESULT] {split} force_unk={int(force_unk)} threshold={threshold:.4f} MCC={metrics['MCC']:.4f} AUC={metrics['AUC']:.4f}")

    out_csv = Path(args.out_csv) if args.out_csv else run_dir / f"multi_test_results_{args.threshold}.csv"
    out_md = Path(args.out_md) if args.out_md else run_dir / f"multi_test_results_{args.threshold}.md"
    write_outputs(out_csv, out_md, rows)


if __name__ == "__main__":
    main()

