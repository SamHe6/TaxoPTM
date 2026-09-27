import argparse
import csv
import io
import json
import re
import sys
import tarfile
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = PROJECT_DIR / "data" / "raw" / "open_phospho"

DBPTM_PHOSPHO_URL = "https://biomics.lab.nycu.edu.tw/dbPTM/download/experiment/Phosphorylation.gz"
UNIPROT_SEARCH_URL = "https://rest.uniprot.org/uniprotkb/search"
UNIPROT_ACCESSIONS_URL = "https://rest.uniprot.org/uniprotkb/accessions"
UNIPROT_QUERY = "(ft_mod_res:phosphoserine) OR (ft_mod_res:phosphothreonine) OR (ft_mod_res:phosphotyrosine)"
UNIPROT_FIELDS = "accession,id,gene_primary,organism_name,organism_id,length,sequence,ft_mod_res"

USER_AGENT = "PTM-project-open-phospho-collector/1.0"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect open phosphorylation-site data from dbPTM and UniProt."
    )
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, type=Path)
    parser.add_argument(
        "--sources",
        default="all",
        help="Comma-separated sources: all, uniprot, dbptm, audit",
    )
    parser.add_argument("--include-audit-only-sources", action="store_true")
    parser.add_argument("--force", action="store_true", help="Redownload/reparse existing files")
    parser.add_argument("--uniprot-page-size", type=int, default=500)
    parser.add_argument(
        "--max-uniprot-records",
        type=int,
        default=None,
        help="Optional cap for smoke tests. Default collects all matching UniProt records.",
    )
    parser.add_argument(
        "--max-dbptm-accessions",
        type=int,
        default=None,
        help="Optional cap for smoke tests when fetching UniProt metadata for dbPTM accessions.",
    )
    parser.add_argument("--accession-batch-size", type=int, default=200)
    parser.add_argument("--request-delay", type=float, default=0.05)
    return parser.parse_args()


def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)


def source_set(raw_sources):
    values = {x.strip().lower() for x in raw_sources.split(",") if x.strip()}
    if "all" in values:
        return {"uniprot", "dbptm", "audit"}
    return values


def request_url(url, method="GET"):
    return Request(url, method=method, headers={"User-Agent": USER_AGENT})


def http_get_text(url, timeout=60, retries=6, backoff=5.0):
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            with urlopen(request_url(url), timeout=timeout) as response:
                return response.read().decode("utf-8", errors="replace"), response.headers
        except (HTTPError, URLError, TimeoutError) as exc:
            last_error = exc
            if isinstance(exc, HTTPError) and 400 <= exc.code < 500 and exc.code not in {408, 429}:
                raise
            if attempt < retries:
                time.sleep(backoff * attempt)
    raise last_error


def download_file(url, out_path, force=False):
    if out_path.exists() and out_path.stat().st_size > 0 and not force:
        return {"status": "skipped_existing", "path": str(out_path), "bytes": out_path.stat().st_size}
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    last_error = None
    for attempt in range(1, 6):
        try:
            with urlopen(request_url(url), timeout=120) as response, open(tmp_path, "wb") as handle:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
            last_error = None
            break
        except (HTTPError, URLError, TimeoutError) as exc:
            last_error = exc
            if attempt < 5:
                time.sleep(5.0 * attempt)
    if last_error is not None:
        raise last_error
    tmp_path.replace(out_path)
    return {"status": "downloaded", "path": str(out_path), "bytes": out_path.stat().st_size}


def probe_url(url):
    for method in ("HEAD", "GET"):
        try:
            req = request_url(url, method=method)
            with urlopen(req, timeout=20) as response:
                return {
                    "http_status": str(response.status),
                    "content_type": response.headers.get("Content-Type", ""),
                    "content_length": response.headers.get("Content-Length", ""),
                    "error": "",
                }
        except HTTPError as exc:
            if method == "HEAD" and exc.code in {405, 403, 400}:
                continue
            return {
                "http_status": str(exc.code),
                "content_type": "",
                "content_length": "",
                "error": str(exc),
            }
        except (URLError, TimeoutError) as exc:
            return {
                "http_status": "",
                "content_type": "",
                "content_length": "",
                "error": str(exc),
            }
    return {"http_status": "", "content_type": "", "content_length": "", "error": "unreachable"}


def write_source_audit(out_dir, include_audit_only=False):
    audit_path = out_dir / "source_audit.tsv"
    sources = [
        {
            "source": "UniProtKB",
            "url": "https://rest.uniprot.org/uniprotkb/search?query=ft_mod_res:phosphoserine&format=tsv&size=1",
            "status": "included",
            "reason": "Official REST API provides canonical sequences and modified-residue annotations.",
        },
        {
            "source": "dbPTM",
            "url": DBPTM_PHOSPHO_URL,
            "status": "included",
            "reason": "Official dbPTM experimental phosphorylation download is available without login.",
        },
        {
            "source": "PhosphoSitePlus",
            "url": "https://www.phosphosite.org/staticLicensing.action",
            "status": "audit_only",
            "reason": "PSP pages state that data use is subject to a formal license agreement.",
        },
        {
            "source": "qPTM",
            "url": "http://qptm.omicsbio.info",
            "status": "audit_only",
            "reason": "No stable, unauthenticated bulk site-level download was confirmed for this workflow.",
        },
        {
            "source": "PTMcode",
            "url": "https://ptmcode.embl.de/",
            "status": "audit_only",
            "reason": "Functional PTM-association resource; no current stable bulk phosphosite table was confirmed.",
        },
        {
            "source": "PTMeXchange",
            "url": "https://www.proteomexchange.org/ptmexchange",
            "status": "audit_only",
            "reason": "Study/build-level PTM exchange resource; not normalized as a direct UniProt site table here.",
        },
        {
            "source": "PeptideAtlas Human Phospho",
            "url": "https://peptideatlas.org/builds/human/phospho/",
            "status": "audit_only",
            "reason": "Open phosphoproteome builds exist, but build-specific flat files require separate normalization.",
        },
    ]

    if not include_audit_only:
        rows = sources
    else:
        rows = sources

    with open(audit_path, "w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "source",
            "url",
            "status",
            "http_status",
            "content_type",
            "content_length",
            "reason",
            "error",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            probe = probe_url(row["url"])
            writer.writerow({**row, **probe})
    return audit_path


def parse_dbptm_tar_gz(gz_path, out_path, force=False):
    if out_path.exists() and out_path.stat().st_size > 0 and not force:
        return {"status": "skipped_existing", "rows": count_data_rows(out_path)}

    rows = 0
    bad_rows = 0
    with tarfile.open(gz_path, mode="r:gz") as archive, open(
        out_path, "w", encoding="utf-8", newline=""
    ) as out_handle:
        fieldnames = [
            "protein_id",
            "uniprot_acc",
            "position",
            "mod_type",
            "pmids",
            "window21",
            "center_residue",
            "source",
        ]
        writer = csv.DictWriter(out_handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for member in archive.getmembers():
            if not member.isfile():
                continue
            extracted = archive.extractfile(member)
            if extracted is None:
                continue
            text_handle = io.TextIOWrapper(extracted, encoding="utf-8", errors="replace")
            for line in text_handle:
                line = line.rstrip("\n\r")
                if not line or "\t" not in line:
                    continue
                parts = line.split("\t")
                if len(parts) < 6:
                    bad_rows += 1
                    continue
                protein_id, acc, pos, mod_type, pmids, window = parts[:6]
                if not pos.isdigit():
                    bad_rows += 1
                    continue
                center = ""
                clean_window = window.strip().upper()
                if len(clean_window) >= 11:
                    center = clean_window[10]
                writer.writerow(
                    {
                        "protein_id": protein_id,
                        "uniprot_acc": acc,
                        "position": pos,
                        "mod_type": mod_type,
                        "pmids": pmids,
                        "window21": clean_window,
                        "center_residue": center,
                        "source": "dbPTM",
                    }
                )
                rows += 1
    return {"status": "parsed", "rows": rows, "bad_rows": bad_rows}


def count_data_rows(path):
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        return max(0, sum(1 for _ in handle) - 1)


def download_uniprot_phospho(out_path, page_size=500, max_records=None, delay=0.05, force=False):
    if out_path.exists() and out_path.stat().st_size > 0 and not force:
        return {"status": "skipped_existing", "rows": count_data_rows(out_path)}

    params = urlencode(
        {
            "query": UNIPROT_QUERY,
            "fields": UNIPROT_FIELDS,
            "format": "tsv",
            "size": str(page_size),
        }
    )
    url = f"{UNIPROT_SEARCH_URL}?{params}"
    rows_written = 0
    pages = 0
    header_written = False
    next_re = re.compile(r"<([^>]+)>;\s*rel=\"next\"")

    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8", newline="") as out_handle:
        while url:
            text, headers = http_get_text(url, timeout=120)
            lines = text.splitlines()
            if not lines:
                break
            if header_written:
                data_lines = lines[1:]
            else:
                data_lines = lines
                header_written = True

            if max_records is not None:
                remaining = max_records - rows_written
                if remaining <= 0:
                    break
                if header_written and data_lines:
                    header_offset = 1 if data_lines[0].startswith("Entry\t") else 0
                    keep = remaining + header_offset
                    data_lines = data_lines[:keep]

            for line in data_lines:
                if line.startswith("Entry\t") and rows_written > 0:
                    continue
                out_handle.write(line + "\n")
                if not line.startswith("Entry\t"):
                    rows_written += 1

            pages += 1
            if max_records is not None and rows_written >= max_records:
                break

            link = headers.get("Link", "")
            match = next_re.search(link)
            url = match.group(1) if match else None
            if delay:
                time.sleep(delay)

    tmp_path.replace(out_path)
    return {"status": "downloaded", "rows": rows_written, "pages": pages}


def read_dbptm_accessions(parsed_path, max_accessions=None):
    accessions = []
    seen = set()
    with open(parsed_path, "r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            acc = row.get("uniprot_acc", "").strip()
            if not acc or acc in seen:
                continue
            seen.add(acc)
            accessions.append(acc)
            if max_accessions is not None and len(accessions) >= max_accessions:
                break
    return accessions


def fetch_accession_batch(accessions, timeout=120):
    params = urlencode(
        {
            "accessions": ",".join(accessions),
            "fields": UNIPROT_FIELDS,
            "format": "tsv",
        }
    )
    url = f"{UNIPROT_ACCESSIONS_URL}?{params}"
    text, _headers = http_get_text(url, timeout=timeout)
    return text


def append_uniprot_accession_records(accessions, out_path, batch_size=200, delay=0.05, force=False):
    if out_path.exists() and out_path.stat().st_size > 0 and not force:
        return {"status": "skipped_existing", "rows": count_data_rows(out_path)}

    rows_written = 0
    batches = 0
    header_written = False
    failures = []
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8", newline="") as out_handle:
        for start in range(0, len(accessions), batch_size):
            batch = accessions[start : start + batch_size]
            try:
                text = fetch_accession_batch(batch)
            except Exception as exc:  # noqa: BLE001 - keep collecting other batches.
                failures.append({"batch_start": start, "batch_size": len(batch), "error": str(exc)})
                continue
            lines = text.splitlines()
            if not lines:
                continue
            if header_written:
                data_lines = lines[1:]
            else:
                data_lines = lines
                header_written = True
            for line in data_lines:
                if line.startswith("Entry\t") and rows_written > 0:
                    continue
                out_handle.write(line + "\n")
                if not line.startswith("Entry\t"):
                    rows_written += 1
            batches += 1
            if delay:
                time.sleep(delay)
    tmp_path.replace(out_path)
    return {
        "status": "downloaded",
        "rows": rows_written,
        "batches": batches,
        "failures": failures[:20],
        "failure_count": len(failures),
    }


def main():
    args = parse_args()
    out_dir = args.out_dir.resolve()
    raw_dir = out_dir / "downloads"
    parsed_dir = out_dir / "parsed"
    ensure_dir(raw_dir)
    ensure_dir(parsed_dir)

    selected = source_set(args.sources)
    manifest = {"out_dir": str(out_dir), "sources": sorted(selected), "steps": {}}

    if "audit" in selected:
        audit_path = write_source_audit(out_dir, args.include_audit_only_sources)
        manifest["steps"]["source_audit"] = {"path": str(audit_path)}

    dbptm_parsed_path = parsed_dir / "dbptm_phosphorylation_sites.tsv"
    if "dbptm" in selected:
        dbptm_gz = raw_dir / "dbptm_Phosphorylation.gz"
        manifest["steps"]["dbptm_download"] = download_file(
            DBPTM_PHOSPHO_URL, dbptm_gz, force=args.force
        )
        manifest["steps"]["dbptm_parse"] = parse_dbptm_tar_gz(
            dbptm_gz, dbptm_parsed_path, force=args.force
        )
        accessions = read_dbptm_accessions(
            dbptm_parsed_path, max_accessions=args.max_dbptm_accessions
        )
        dbptm_uniprot_path = parsed_dir / "uniprot_dbptm_accessions.tsv"
        manifest["steps"]["dbptm_uniprot_metadata"] = append_uniprot_accession_records(
            accessions,
            dbptm_uniprot_path,
            batch_size=args.accession_batch_size,
            delay=args.request_delay,
            force=args.force,
        )
        manifest["steps"]["dbptm_uniprot_metadata"]["requested_accessions"] = len(accessions)

    if "uniprot" in selected:
        uniprot_path = parsed_dir / "uniprot_phospho_records.tsv"
        manifest["steps"]["uniprot_phospho_records"] = download_uniprot_phospho(
            uniprot_path,
            page_size=args.uniprot_page_size,
            max_records=args.max_uniprot_records,
            delay=args.request_delay,
            force=args.force,
        )

    manifest_path = out_dir / "collection_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("Interrupted")
