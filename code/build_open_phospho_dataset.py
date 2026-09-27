import argparse
import csv
import json
import random
import re
from collections import defaultdict
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = PROJECT_DIR / "data" / "raw" / "open_phospho"
DEFAULT_OUT_DIR = PROJECT_DIR / "data" / "processed"

PHOSPHO_RESIDUES = {"S", "T", "Y"}
NOTE_TO_RESIDUE = {
    "phosphoserine": "S",
    "phosphothreonine": "T",
    "phosphotyrosine": "Y",
}
MOD_RES_RE = re.compile(
    r"MOD_RES\s+(\d+);\s*/note=\"([^\"]+)\"(?:;\s*/evidence=\"([^\"]+)\")?"
)
PUBMED_RE = re.compile(r"PubMed:(\d+)|(?<!\d)(\d{5,})(?!\d)")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a 41-aa open phosphorylation dataset from collected open sources."
    )
    parser.add_argument("--raw-dir", default=DEFAULT_RAW_DIR, type=Path)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, type=Path)
    parser.add_argument("--window-size", type=int, default=41)
    parser.add_argument("--negative-ratio", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--test-frac", type=float, default=0.1)
    parser.add_argument(
        "--max-positive-windows",
        type=int,
        default=None,
        help="Optional cap for model-export size. Master table is still built from all parsed sites.",
    )
    parser.add_argument(
        "--min-quality-tier",
        choices=["medium", "high"],
        default="medium",
        help="Minimum positive quality tier used in model windows.",
    )
    return parser.parse_args()


def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)


def clean_sequence(seq):
    return re.sub(r"[^A-Z]", "", (seq or "").upper())


def quality_rank(tier):
    return {"medium": 1, "high": 2}.get(tier, 0)


def parse_pmids(text):
    if not text:
        return set()
    pmids = set()
    for match in PUBMED_RE.finditer(text):
        value = match.group(1) or match.group(2)
        if value:
            pmids.add(value)
    return pmids


def uniprot_quality(evidence):
    if not evidence:
        return "medium"
    if "ECO:0000269" in evidence or "ECO:0007744" in evidence:
        return "high"
    return "medium"


def note_to_residue(note):
    lowered = (note or "").lower()
    for key, residue in NOTE_TO_RESIDUE.items():
        if key in lowered:
            return residue
    return ""


def read_uniprot_records(path):
    records = {}
    if not path.exists():
        return records
    with open(path, "r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            acc = row.get("Entry", "").strip()
            seq = clean_sequence(row.get("Sequence", ""))
            if not acc or not seq:
                continue
            current = records.get(acc)
            record = {
                "uniprot_acc": acc,
                "protein_id": row.get("Entry Name", "").strip(),
                "gene": row.get("Gene Names (primary)", "").strip(),
                "organism": row.get("Organism", "").strip(),
                "tax_id": row.get("Organism (ID)", "").strip(),
                "sequence_length": int(row.get("Length", "0") or 0),
                "sequence": seq,
                "modified_residue": row.get("Modified residue", ""),
            }
            if current is None or (not current.get("modified_residue") and record["modified_residue"]):
                records[acc] = record
    return records


def iter_uniprot_mod_sites(record):
    text = record.get("modified_residue", "")
    for match in MOD_RES_RE.finditer(text):
        position = int(match.group(1))
        note = match.group(2) or ""
        evidence = match.group(3) or ""
        residue = note_to_residue(note)
        if residue not in PHOSPHO_RESIDUES:
            continue
        yield {
            "uniprot_acc": record["uniprot_acc"],
            "position": position,
            "residue": residue,
            "mod_type": note.split(";")[0],
            "source": "UniProtKB",
            "pmids": parse_pmids(evidence),
            "quality_tier": uniprot_quality(evidence),
        }


def empty_site(record, residue, position):
    return {
        "site_key": f"{record['uniprot_acc']}|{residue}|{position}",
        "uniprot_acc": record["uniprot_acc"],
        "protein_id": record.get("protein_id", ""),
        "gene": record.get("gene", ""),
        "organism": record.get("organism", ""),
        "tax_id": record.get("tax_id", ""),
        "sequence_length": len(record.get("sequence", "")),
        "residue": residue,
        "position": position,
        "mod_type": "Phosphorylation",
        "sources": set(),
        "pmids": set(),
        "quality_values": [],
        "sequence": record.get("sequence", ""),
        "is_positive": "1",
        "exclusion_reason": "",
        "source_records": 0,
    }


def add_site(sites, record, residue, position, source, pmids, quality_tier, mod_type="Phosphorylation"):
    seq = record.get("sequence", "")
    if position < 1 or position > len(seq):
        return False, "position_out_of_range"
    if seq[position - 1] != residue:
        return False, "residue_mismatch"
    key = f"{record['uniprot_acc']}|{residue}|{position}"
    if key not in sites:
        sites[key] = empty_site(record, residue, position)
    site = sites[key]
    site["sources"].add(source)
    site["pmids"].update(pmids)
    site["quality_values"].append(quality_tier)
    site["source_records"] += 1
    if mod_type:
        site["mod_type"] = mod_type
    return True, ""


def merge_sequence_records(raw_dir):
    parsed = raw_dir / "parsed"
    sequence_records = {}
    for name in ("uniprot_phospho_records.tsv", "uniprot_dbptm_accessions.tsv"):
        records = read_uniprot_records(parsed / name)
        for acc, record in records.items():
            current = sequence_records.get(acc)
            if current is None:
                sequence_records[acc] = record
            elif not current.get("modified_residue") and record.get("modified_residue"):
                sequence_records[acc] = record
    return sequence_records


def build_sites(raw_dir, sequence_records):
    parsed = raw_dir / "parsed"
    sites = {}
    counters = defaultdict(int)
    invalid_rows = []

    for record in sequence_records.values():
        for site in iter_uniprot_mod_sites(record):
            ok, reason = add_site(
                sites,
                record,
                site["residue"],
                site["position"],
                site["source"],
                site["pmids"],
                site["quality_tier"],
                site["mod_type"],
            )
            counters["uniprot_sites_seen"] += 1
            if not ok:
                counters[f"uniprot_{reason}"] += 1
                invalid_rows.append({**site, "reason": reason})

    dbptm_path = parsed / "dbptm_phosphorylation_sites.tsv"
    if dbptm_path.exists():
        with open(dbptm_path, "r", encoding="utf-8", errors="replace", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                counters["dbptm_sites_seen"] += 1
                acc = row.get("uniprot_acc", "").strip()
                record = sequence_records.get(acc)
                if record is None:
                    counters["dbptm_missing_sequence"] += 1
                    continue
                try:
                    position = int(row.get("position", "0"))
                except ValueError:
                    counters["dbptm_bad_position"] += 1
                    continue
                seq = record.get("sequence", "")
                residue = seq[position - 1] if 1 <= position <= len(seq) else ""
                if residue not in PHOSPHO_RESIDUES:
                    counters["dbptm_non_sty_or_out_of_range"] += 1
                    continue
                center = row.get("center_residue", "").strip().upper()
                if center and center != "-" and center != residue:
                    counters["dbptm_window_center_mismatch"] += 1
                ok, reason = add_site(
                    sites,
                    record,
                    residue,
                    position,
                    "dbPTM",
                    parse_pmids(row.get("pmids", "")),
                    "high",
                    "Phosphorylation",
                )
                if not ok:
                    counters[f"dbptm_{reason}"] += 1
                    invalid_rows.append(
                        {
                            "uniprot_acc": acc,
                            "position": position,
                            "residue": residue,
                            "source": "dbPTM",
                            "reason": reason,
                        }
                    )

    return sites, counters, invalid_rows


def finalized_site_row(site):
    quality_tier = "high" if any(x == "high" for x in site["quality_values"]) else "medium"
    pmids = sorted(site["pmids"], key=lambda x: (len(x), x))
    sources = sorted(site["sources"])
    evidence_count = len(pmids) if pmids else site["source_records"]
    return {
        "site_key": site["site_key"],
        "uniprot_acc": site["uniprot_acc"],
        "protein_id": site["protein_id"],
        "gene": site["gene"],
        "organism": site["organism"],
        "tax_id": site["tax_id"],
        "sequence_length": str(site["sequence_length"]),
        "residue": site["residue"],
        "position": str(site["position"]),
        "mod_type": site["mod_type"],
        "sources": ";".join(sources),
        "evidence_count": str(evidence_count),
        "pmids": ";".join(pmids),
        "quality_tier": quality_tier,
        "sequence": site["sequence"],
        "is_positive": site["is_positive"],
        "exclusion_reason": site["exclusion_reason"],
    }


def write_master_table(sites, out_path):
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
    with open(out_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for site_key in sorted(sites):
            writer.writerow(finalized_site_row(sites[site_key]))


def make_window(sequence, position, flank):
    start = position - flank - 1
    end = position + flank
    if start < 0 or end > len(sequence):
        return ""
    return sequence[start:end]


def build_positive_windows(sites, window_size, min_quality_tier, max_positive_windows, rng):
    flank = window_size // 2
    rows = []
    for site in sites.values():
        final = finalized_site_row(site)
        if quality_rank(final["quality_tier"]) < quality_rank(min_quality_tier):
            continue
        position = int(final["position"])
        window = make_window(final["sequence"], position, flank)
        if len(window) != window_size:
            continue
        if window[flank] != final["residue"]:
            continue
        rows.append(
            {
                "sample_id": "",
                "split": "",
                "label": "1",
                "site_key": final["site_key"],
                "uniprot_acc": final["uniprot_acc"],
                "protein_id": final["protein_id"],
                "gene": final["gene"],
                "organism": final["organism"],
                "tax_id": final["tax_id"],
                "residue": final["residue"],
                "position": final["position"],
                "center_index": str(flank),
                "window": window,
                "sources": final["sources"],
                "quality_tier": final["quality_tier"],
                "pmids": final["pmids"],
            }
        )
    if max_positive_windows is not None and len(rows) > max_positive_windows:
        rows = rng.sample(rows, max_positive_windows)
    return rows


def iter_negative_candidates(sequence_records, exclusion_positions, window_size):
    flank = window_size // 2
    for acc, record in sequence_records.items():
        seq = record.get("sequence", "")
        if not seq:
            continue
        excluded = exclusion_positions.get(acc, set())
        for idx, residue in enumerate(seq, start=1):
            if residue not in PHOSPHO_RESIDUES or idx in excluded:
                continue
            window = make_window(seq, idx, flank)
            if len(window) != window_size:
                continue
            yield record, residue, idx, window


def sample_negative_windows(sequence_records, sites, window_size, negative_count, rng):
    exclusion_positions = defaultdict(set)
    for site in sites.values():
        exclusion_positions[site["uniprot_acc"]].add(site["position"])

    reservoir = []
    seen = 0
    for record, residue, position, window in iter_negative_candidates(
        sequence_records, exclusion_positions, window_size
    ):
        seen += 1
        candidate = {
            "sample_id": "",
            "split": "",
            "label": "0",
            "site_key": f"{record['uniprot_acc']}|{residue}|{position}",
            "uniprot_acc": record["uniprot_acc"],
            "protein_id": record.get("protein_id", ""),
            "gene": record.get("gene", ""),
            "organism": record.get("organism", ""),
            "tax_id": record.get("tax_id", ""),
            "residue": residue,
            "position": str(position),
            "center_index": str(window_size // 2),
            "window": window,
            "sources": "sampled_negative",
            "quality_tier": "sampled_negative",
            "pmids": "",
        }
        if len(reservoir) < negative_count:
            reservoir.append(candidate)
        else:
            j = rng.randrange(seen)
            if j < negative_count:
                reservoir[j] = candidate
    return reservoir, seen


def assign_splits(rows, train_frac, val_frac, test_frac, rng):
    total = train_frac + val_frac + test_frac
    if abs(total - 1.0) > 1e-6:
        raise ValueError("train/val/test fractions must sum to 1.0")
    accessions = sorted({row["uniprot_acc"] for row in rows})
    rng.shuffle(accessions)
    n = len(accessions)
    train_cut = int(n * train_frac)
    val_cut = train_cut + int(n * val_frac)
    split_by_acc = {}
    for idx, acc in enumerate(accessions):
        if idx < train_cut:
            split_by_acc[acc] = "train"
        elif idx < val_cut:
            split_by_acc[acc] = "val"
        else:
            split_by_acc[acc] = "test"
    for row in rows:
        row["split"] = split_by_acc[row["uniprot_acc"]]


def remove_cross_split_window_overlaps(rows):
    split_priority = {"train": 3, "val": 2, "test": 1}
    by_window = defaultdict(list)
    for idx, row in enumerate(rows):
        by_window[row["window"]].append(idx)

    drop = set()
    removed = defaultdict(int)
    for indices in by_window.values():
        splits = {rows[i]["split"] for i in indices}
        if len(splits) <= 1:
            continue
        keep_split = max(splits, key=lambda split: split_priority.get(split, 0))
        for i in indices:
            if rows[i]["split"] != keep_split:
                drop.add(i)
                removed[rows[i]["split"]] += 1
    kept = [row for idx, row in enumerate(rows) if idx not in drop]
    return kept, dict(removed)


def assign_sample_ids(rows):
    counters = defaultdict(int)
    for row in rows:
        prefix = "Pos" if row["label"] == "1" else "Neg"
        split = row["split"]
        counters[(split, prefix)] += 1
        row["sample_id"] = f"{prefix}_{split}_{counters[(split, prefix)]:06d}"


def write_windows(rows, out_path):
    fieldnames = [
        "sample_id",
        "split",
        "label",
        "site_key",
        "uniprot_acc",
        "protein_id",
        "gene",
        "organism",
        "tax_id",
        "residue",
        "position",
        "center_index",
        "window",
        "sources",
        "quality_tier",
        "pmids",
    ]
    with open(out_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_fasta_txt(rows, out_dir):
    mapping = {
        "train": out_dir / "Training_open_phospho.txt",
        "val": out_dir / "Validation_open_phospho.txt",
        "test": out_dir / "Independent_open_phospho.txt",
    }
    handles = {split: open(path, "w", encoding="utf-8", newline="\n") for split, path in mapping.items()}
    try:
        for row in rows:
            split = row["split"]
            header = (
                f">{row['sample_id']}|{row['label']}|{split}|{row['uniprot_acc']}|"
                f"{row['residue']}{row['position']}|{row['sources']}"
            )
            handles[split].write(header + "\n")
            handles[split].write(row["window"] + "\n")
    finally:
        for handle in handles.values():
            handle.close()
    return {split: str(path) for split, path in mapping.items()}


def write_invalid_rows(invalid_rows, out_path):
    if not invalid_rows:
        out_path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in invalid_rows for key in row})
    with open(out_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in invalid_rows:
            writer.writerow(row)


def main():
    args = parse_args()
    if args.window_size % 2 != 1:
        raise ValueError("--window-size must be odd so the modified residue has a fixed center index")

    rng = random.Random(args.seed)
    raw_dir = args.raw_dir.resolve()
    out_dir = args.out_dir.resolve()
    ensure_dir(out_dir)

    sequence_records = merge_sequence_records(raw_dir)
    sites, counters, invalid_rows = build_sites(raw_dir, sequence_records)

    master_path = out_dir / "phospho_sites_master.tsv"
    write_master_table(sites, master_path)
    write_invalid_rows(invalid_rows, out_dir / "phospho_invalid_sites.tsv")

    positive_rows = build_positive_windows(
        sites,
        args.window_size,
        args.min_quality_tier,
        args.max_positive_windows,
        rng,
    )
    negative_target = int(round(len(positive_rows) * args.negative_ratio))
    negative_rows, negative_candidates_seen = sample_negative_windows(
        sequence_records, sites, args.window_size, negative_target, rng
    )

    rows = positive_rows + negative_rows
    assign_splits(rows, args.train_frac, args.val_frac, args.test_frac, rng)
    rows, overlap_removed = remove_cross_split_window_overlaps(rows)
    rows.sort(key=lambda r: (r["split"], r["label"], r["uniprot_acc"], int(r["position"])))
    assign_sample_ids(rows)

    windows_path = out_dir / f"phospho_windows_{args.window_size}aa.tsv"
    write_windows(rows, windows_path)
    txt_paths = write_fasta_txt(rows, out_dir)

    split_counts = defaultdict(lambda: defaultdict(int))
    for row in rows:
        split_counts[row["split"]][row["label"]] += 1

    summary = {
        "raw_dir": str(raw_dir),
        "out_dir": str(out_dir),
        "sequence_records": len(sequence_records),
        "merged_positive_sites": len(sites),
        "positive_windows": len(positive_rows),
        "negative_windows_requested": negative_target,
        "negative_windows_sampled": len(negative_rows),
        "negative_candidates_seen": negative_candidates_seen,
        "window_rows_after_overlap_removal": len(rows),
        "overlap_removed_by_split": overlap_removed,
        "split_counts": {split: dict(counts) for split, counts in split_counts.items()},
        "source_counters": dict(counters),
        "master_path": str(master_path),
        "windows_path": str(windows_path),
        "txt_paths": txt_paths,
        "window_size": args.window_size,
        "seed": args.seed,
        "min_quality_tier": args.min_quality_tier,
        "max_positive_windows": args.max_positive_windows,
    }
    summary_path = out_dir / "open_phospho_build_summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
