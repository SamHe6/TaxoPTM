import argparse
import csv
import json
import math
import random
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_DIR / "data" / "processed" / "phospho_windows_41aa.tsv"
DEFAULT_OUT_DIR = PROJECT_DIR / "data" / "processed" / "open_phospho_clean"

ALLOWED_AA = set("ACDEFGHIKLMNPQRSTVWY")
PHOSPHO_RESIDUES = {"S", "T", "Y"}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Clean the open phosphorylation dataset with high-confidence filters "
            "and a CD-HIT-like window identity filter."
        )
    )
    parser.add_argument("--input", default=DEFAULT_INPUT, type=Path)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, type=Path)
    parser.add_argument("--window-size", type=int, default=41)
    parser.add_argument("--identity-threshold", type=float, default=0.8)
    parser.add_argument("--min-positive-pmids", type=int, default=2)
    parser.add_argument("--negative-ratio", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--cd-hit-path",
        default=None,
        type=Path,
        help="Optional path to cd-hit executable. If omitted, use the built-in CD-HIT-like filter.",
    )
    parser.add_argument("--cd-hit-word-length", type=int, default=5)
    parser.add_argument("--cd-hit-threads", type=int, default=0)
    parser.add_argument(
        "--keep-multisource-without-pmid",
        action="store_true",
        help="Keep positive rows supported by multiple sources even if PMID count is below threshold.",
    )
    return parser.parse_args()


def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)


def pmid_count(row):
    return len([x for x in row.get("pmids", "").split(";") if x.strip()])


def source_count(row):
    return len([x for x in row.get("sources", "").split(";") if x.strip()])


def is_canonical_window(window):
    return bool(window) and set(window).issubset(ALLOWED_AA)


def high_confidence_positive(row, min_pmids, keep_multisource_without_pmid):
    if row.get("label") != "1":
        return False, "not_positive"
    if row.get("quality_tier") != "high":
        return False, "positive_not_high_quality"
    if pmid_count(row) >= min_pmids:
        return True, ""
    if keep_multisource_without_pmid and source_count(row) >= 2:
        return True, ""
    return False, "positive_low_pmid_support"


def valid_negative(row):
    if row.get("label") != "0":
        return False, "not_negative"
    if row.get("quality_tier") != "sampled_negative":
        return False, "negative_not_sampled_negative"
    return True, ""


def base_filter_reason(row, window_size, min_pmids, keep_multisource_without_pmid):
    window = row.get("window", "")
    if len(window) != window_size:
        return "bad_window_length"
    if not is_canonical_window(window):
        return "noncanonical_window"
    residue = row.get("residue", "")
    if residue not in PHOSPHO_RESIDUES:
        return "center_not_sty"
    try:
        center_index = int(row.get("center_index", "-1"))
    except ValueError:
        return "bad_center_index"
    if center_index < 0 or center_index >= len(window):
        return "bad_center_index"
    if window[center_index] != residue:
        return "center_residue_mismatch"

    if row.get("label") == "1":
        ok, reason = high_confidence_positive(row, min_pmids, keep_multisource_without_pmid)
        return "" if ok else reason
    if row.get("label") == "0":
        ok, reason = valid_negative(row)
        return "" if ok else reason
    return "bad_label"


def row_priority(row):
    label = row.get("label")
    if label == "1":
        return (
            2,
            pmid_count(row),
            source_count(row),
            row.get("quality_tier", ""),
            row.get("site_key", ""),
        )
    return (
        1,
        0,
        0,
        row.get("quality_tier", ""),
        row.get("site_key", ""),
    )


def hamming_mismatches(left, right, max_mismatches):
    mismatches = 0
    for a, b in zip(left, right):
        if a != b:
            mismatches += 1
            if mismatches > max_mismatches:
                return mismatches
    return mismatches


def make_slices(length, parts):
    slices = []
    for i in range(parts):
        start = round(i * length / parts)
        end = round((i + 1) * length / parts)
        slices.append((start, end))
    return slices


def greedy_identity_filter(rows, window_size, identity_threshold):
    max_mismatches = math.floor((1.0 - identity_threshold) * window_size + 1e-9)
    slices = make_slices(window_size, max_mismatches + 1)
    buckets = defaultdict(list)
    kept = []
    removed = []

    sorted_rows = sorted(rows, key=row_priority, reverse=True)
    for row in sorted_rows:
        window = row["window"]
        candidate_indices = set()
        for slice_id, (start, end) in enumerate(slices):
            candidate_indices.update(buckets[(slice_id, window[start:end])])

        representative = None
        representative_identity = 0.0
        for kept_idx in candidate_indices:
            kept_row = kept[kept_idx]
            mismatches = hamming_mismatches(window, kept_row["window"], max_mismatches)
            if mismatches <= max_mismatches:
                representative = kept_row
                representative_identity = 1.0 - (mismatches / window_size)
                break

        if representative is not None:
            removed.append(
                {
                    **row,
                    "remove_reason": "redundant_window_identity",
                    "representative_sample_id": representative.get("sample_id", ""),
                    "representative_site_key": representative.get("site_key", ""),
                    "representative_label": representative.get("label", ""),
                    "identity": f"{representative_identity:.4f}",
                }
            )
            continue

        kept_idx = len(kept)
        kept.append(row)
        for slice_id, (start, end) in enumerate(slices):
            buckets[(slice_id, window[start:end])].append(kept_idx)

    return kept, removed, max_mismatches


def write_cdhit_fasta(rows, path):
    sorted_rows = sorted(enumerate(rows), key=lambda x: row_priority(x[1]), reverse=True)
    id_to_index = {}
    ordered_rows = []
    with open(path, "w", encoding="ascii", newline="\n") as handle:
        for out_idx, (_original_idx, row) in enumerate(sorted_rows):
            seq_id = f"row{out_idx:08d}"
            id_to_index[seq_id] = out_idx
            ordered_rows.append(row)
            handle.write(f">{seq_id}\n")
            handle.write(row["window"] + "\n")
    return ordered_rows, id_to_index


def parse_cdhit_clusters(clstr_path):
    member_to_rep = {}
    member_identity = {}
    current_members = []
    current_rep = None
    id_re = re.compile(r">([^\.>\s]+)\.\.\.")
    identity_re = re.compile(r"at\s+([0-9.]+)%")

    def flush():
        if current_rep is None:
            return
        for member, identity in current_members:
            member_to_rep[member] = current_rep
            member_identity[member] = identity

    with open(clstr_path, "r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">Cluster"):
                flush()
                current_members = []
                current_rep = None
                continue
            id_match = id_re.search(line)
            if not id_match:
                continue
            member = id_match.group(1)
            identity_match = identity_re.search(line)
            identity = ""
            if identity_match:
                identity = f"{float(identity_match.group(1)) / 100.0:.4f}"
            if line.endswith("*"):
                current_rep = member
                identity = "1.0000"
            current_members.append((member, identity))
    flush()
    return member_to_rep, member_identity


def cdhit_identity_filter(rows, out_dir, identity_threshold, cd_hit_path, word_length, threads):
    cdhit_dir = out_dir / "cdhit"
    cdhit_dir.mkdir(parents=True, exist_ok=True)
    input_fasta = cdhit_dir / "candidates.fa"
    output_fasta = cdhit_dir / "representatives.fa"
    log_path = cdhit_dir / "cd-hit.log"

    ordered_rows, id_to_index = write_cdhit_fasta(rows, input_fasta)
    command = [
        str(cd_hit_path),
        "-i",
        str(input_fasta),
        "-o",
        str(output_fasta),
        "-c",
        str(identity_threshold),
        "-n",
        str(word_length),
        "-M",
        "0",
        "-T",
        str(threads),
        "-d",
        "0",
        "-l",
        "10",
    ]
    proc = subprocess.run(command, text=True, capture_output=True, check=False)
    log_path.write_text(
        "COMMAND: " + " ".join(command) + "\n\nSTDOUT:\n" + proc.stdout + "\nSTDERR:\n" + proc.stderr,
        encoding="utf-8",
    )
    if proc.returncode != 0:
        raise RuntimeError(f"cd-hit failed with exit code {proc.returncode}; see {log_path}")

    kept_ids = []
    with open(output_fasta, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith(">"):
                kept_ids.append(line[1:].strip().split()[0])
    kept_id_set = set(kept_ids)
    member_to_rep, member_identity = parse_cdhit_clusters(Path(str(output_fasta) + ".clstr"))

    kept = [ordered_rows[id_to_index[seq_id]] for seq_id in kept_ids]
    removed = []
    for seq_id, out_idx in id_to_index.items():
        if seq_id in kept_id_set:
            continue
        row = ordered_rows[out_idx]
        rep_id = member_to_rep.get(seq_id, "")
        rep_row = ordered_rows[id_to_index[rep_id]] if rep_id in id_to_index else {}
        removed.append(
            {
                **row,
                "remove_reason": "cdhit_redundant_window_identity",
                "representative_sample_id": rep_row.get("sample_id", ""),
                "representative_site_key": rep_row.get("site_key", ""),
                "representative_label": rep_row.get("label", ""),
                "identity": member_identity.get(seq_id, ""),
            }
        )
    return kept, removed


def read_windows(path):
    with open(path, "r", encoding="utf-8", errors="replace", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(rows, path, fieldnames):
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def balance_by_split(rows, negative_ratio, seed):
    rng = random.Random(seed)
    by_split_label = defaultdict(list)
    for row in rows:
        by_split_label[(row["split"], row["label"])].append(row)

    kept = []
    removed = []
    for split in sorted({row["split"] for row in rows}):
        positives = by_split_label[(split, "1")]
        negatives = by_split_label[(split, "0")]
        rng.shuffle(negatives)
        negative_target = min(len(negatives), int(round(len(positives) * negative_ratio)))
        kept.extend(positives)
        kept.extend(negatives[:negative_target])
        for row in negatives[negative_target:]:
            removed.append({**row, "remove_reason": "negative_balance_downsample"})
    return kept, removed


def assign_sample_ids(rows):
    counters = defaultdict(int)
    for row in sorted(rows, key=lambda r: (r["split"], r["label"], r["uniprot_acc"], int(r["position"]))):
        prefix = "PosClean" if row["label"] == "1" else "NegClean"
        key = (row["split"], prefix)
        counters[key] += 1
        row["sample_id"] = f"{prefix}_{row['split']}_{counters[key]:06d}"
    rows.sort(key=lambda r: (r["split"], r["label"], r["sample_id"]))


def write_fasta_like(rows, out_dir):
    paths = {
        "train": out_dir / "Training_open_phospho.txt",
        "val": out_dir / "Validation_open_phospho.txt",
        "test": out_dir / "Independent_open_phospho.txt",
    }
    handles = {split: open(path, "w", encoding="utf-8", newline="\n") for split, path in paths.items()}
    try:
        for row in rows:
            header = (
                f">{row['sample_id']}|{row['label']}|{row['split']}|{row['uniprot_acc']}|"
                f"{row['residue']}{row['position']}|{row['sources']}"
            )
            handles[row["split"]].write(header + "\n")
            handles[row["split"]].write(row["window"] + "\n")
    finally:
        for handle in handles.values():
            handle.close()
    return paths


def write_clean_master(rows, path):
    fieldnames = [
        "site_key",
        "uniprot_acc",
        "protein_id",
        "gene",
        "organism",
        "tax_id",
        "sequence_length",
        "residue",
        "position",
        "mod_type",
        "sources",
        "evidence_count",
        "pmids",
        "quality_tier",
        "sequence",
        "is_positive",
        "exclusion_reason",
    ]
    master_rows = []
    seen = set()
    for row in rows:
        if row["label"] != "1" or row["site_key"] in seen:
            continue
        seen.add(row["site_key"])
        master_rows.append(
            {
                "site_key": row["site_key"],
                "uniprot_acc": row["uniprot_acc"],
                "protein_id": row.get("protein_id", ""),
                "gene": row.get("gene", ""),
                "organism": row.get("organism", ""),
                "tax_id": row.get("tax_id", ""),
                "sequence_length": "",
                "residue": row["residue"],
                "position": row["position"],
                "mod_type": "Phosphorylation",
                "sources": row.get("sources", ""),
                "evidence_count": str(pmid_count(row)),
                "pmids": row.get("pmids", ""),
                "quality_tier": row.get("quality_tier", ""),
                "sequence": "",
                "is_positive": "1",
                "exclusion_reason": "",
            }
        )
    write_tsv(master_rows, path, fieldnames)


def summarize(rows):
    split_counts = defaultdict(Counter)
    source_counts = Counter()
    pmid_bins = Counter()
    for row in rows:
        split_counts[row["split"]][row["label"]] += 1
        if row["label"] == "1":
            for source in row.get("sources", "").split(";"):
                if source:
                    source_counts[source] += 1
            n = pmid_count(row)
            if n >= 10:
                pmid_bins[">=10"] += 1
            elif n >= 5:
                pmid_bins["5-9"] += 1
            elif n >= 2:
                pmid_bins["2-4"] += 1
            elif n == 1:
                pmid_bins["1"] += 1
            else:
                pmid_bins["0"] += 1
    return {
        "split_counts": {split: dict(counts) for split, counts in split_counts.items()},
        "positive_sources": dict(source_counts),
        "positive_pmid_bins": dict(pmid_bins),
        "rows": len(rows),
        "unique_proteins": len({row["uniprot_acc"] for row in rows}),
    }


def main():
    args = parse_args()
    if args.window_size % 2 != 1:
        raise ValueError("--window-size must be odd")
    if not (0 < args.identity_threshold <= 1):
        raise ValueError("--identity-threshold must be in (0, 1]")

    out_dir = args.out_dir.resolve()
    ensure_dir(out_dir)
    rows = read_windows(args.input)

    filtered = []
    removed_filter = []
    for row in rows:
        reason = base_filter_reason(
            row,
            args.window_size,
            args.min_positive_pmids,
            args.keep_multisource_without_pmid,
        )
        if reason:
            removed_filter.append({**row, "remove_reason": reason})
        else:
            filtered.append(row)

    max_mismatches = math.floor((1.0 - args.identity_threshold) * args.window_size + 1e-9)
    redundancy_method = "cdhit_like_hamming"
    if args.cd_hit_path is not None:
        nonredundant, removed_redundant = cdhit_identity_filter(
            filtered,
            out_dir,
            args.identity_threshold,
            args.cd_hit_path,
            args.cd_hit_word_length,
            args.cd_hit_threads,
        )
        redundancy_method = "cd-hit"
    else:
        nonredundant, removed_redundant, max_mismatches = greedy_identity_filter(
            filtered, args.window_size, args.identity_threshold
        )
    balanced, removed_balance = balance_by_split(nonredundant, args.negative_ratio, args.seed)
    assign_sample_ids(balanced)

    window_fieldnames = list(rows[0].keys()) if rows else []
    if "remove_reason" not in window_fieldnames:
        removed_fieldnames = window_fieldnames + [
            "remove_reason",
            "representative_sample_id",
            "representative_site_key",
            "representative_label",
            "identity",
        ]
    else:
        removed_fieldnames = window_fieldnames

    write_tsv(balanced, out_dir / "phospho_windows_41aa.tsv", window_fieldnames)
    write_clean_master(balanced, out_dir / "phospho_sites_master.tsv")
    write_fasta_like(balanced, out_dir)
    write_tsv(
        removed_filter + removed_redundant + removed_balance,
        out_dir / "cleaning_removed_rows.tsv",
        removed_fieldnames,
    )

    report = {
        "input": str(args.input.resolve()),
        "out_dir": str(out_dir),
        "window_size": args.window_size,
        "identity_threshold": args.identity_threshold,
        "max_mismatches_allowed": max_mismatches,
        "redundancy_method": redundancy_method,
        "cd_hit_path": str(args.cd_hit_path) if args.cd_hit_path else "",
        "min_positive_pmids": args.min_positive_pmids,
        "negative_ratio": args.negative_ratio,
        "seed": args.seed,
        "input_rows": len(rows),
        "after_high_confidence_filter": len(filtered),
        "after_identity_filter": len(nonredundant),
        "final_rows": len(balanced),
        "removed_by_filter": dict(Counter(row["remove_reason"] for row in removed_filter)),
        "removed_redundant": len(removed_redundant),
        "removed_for_balance": len(removed_balance),
        "final_summary": summarize(balanced),
    }
    with open(out_dir / "cleaning_summary.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    build_summary = {
        "raw_dir": "",
        "out_dir": str(out_dir),
        "sequence_records": report["final_summary"]["unique_proteins"],
        "merged_positive_sites": report["final_summary"]["split_counts"].get("train", {}).get("1", 0)
        + report["final_summary"]["split_counts"].get("val", {}).get("1", 0)
        + report["final_summary"]["split_counts"].get("test", {}).get("1", 0),
        "positive_windows": report["final_summary"]["split_counts"].get("train", {}).get("1", 0)
        + report["final_summary"]["split_counts"].get("val", {}).get("1", 0)
        + report["final_summary"]["split_counts"].get("test", {}).get("1", 0),
        "negative_windows_sampled": report["final_summary"]["split_counts"].get("train", {}).get("0", 0)
        + report["final_summary"]["split_counts"].get("val", {}).get("0", 0)
        + report["final_summary"]["split_counts"].get("test", {}).get("0", 0),
        "negative_candidates_seen": "",
        "window_rows_after_overlap_removal": report["final_rows"],
        "overlap_removed_by_split": {},
        "min_quality_tier": "high",
        "max_positive_windows": "",
        "seed": args.seed,
        "cleaning_identity_threshold": args.identity_threshold,
        "cleaning_min_positive_pmids": args.min_positive_pmids,
    }
    with open(out_dir / "open_phospho_build_summary.json", "w", encoding="utf-8") as handle:
        json.dump(build_summary, handle, indent=2, sort_keys=True)

    md = [
        "# Clean High-Confidence Open Phospho Dataset",
        "",
        "## Cleaning Rules",
        "",
        f"- Positive rows require `quality_tier=high` and at least `{args.min_positive_pmids}` PMID evidence IDs.",
        "- Negative rows must be sampled negatives from the previous exclusion-aware build.",
        f"- Canonical amino-acid windows only; window length must be `{args.window_size}`.",
        f"- CD-HIT-like greedy window identity threshold: `{args.identity_threshold}`.",
        f"- Maximum mismatches allowed at this threshold: `{max_mismatches}`.",
        f"- Redundancy method used: `{redundancy_method}`.",
        "",
        "## Counts",
        "",
        f"- Input rows: {len(rows)}",
        f"- After high-confidence/base filters: {len(filtered)}",
        f"- After identity filtering: {len(nonredundant)}",
        f"- Final balanced rows: {len(balanced)}",
        f"- Redundant rows removed: {len(removed_redundant)}",
        f"- Negative balance rows removed: {len(removed_balance)}",
        "",
        "## Final Split Counts",
        "",
        "| Split | Negatives | Positives |",
        "|---|---:|---:|",
    ]
    summary = summarize(balanced)
    for split in ("train", "val", "test"):
        counts = summary["split_counts"].get(split, {})
        md.append(f"| {split} | {counts.get('0', 0)} | {counts.get('1', 0)} |")
    md.extend(
        [
            "",
            "## Outputs",
            "",
            f"- `{out_dir / 'phospho_windows_41aa.tsv'}`",
            f"- `{out_dir / 'phospho_sites_master.tsv'}`",
            f"- `{out_dir / 'Training_open_phospho.txt'}`",
            f"- `{out_dir / 'Validation_open_phospho.txt'}`",
            f"- `{out_dir / 'Independent_open_phospho.txt'}`",
            f"- `{out_dir / 'cleaning_removed_rows.tsv'}`",
        ]
    )
    (out_dir / "cleaning_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
