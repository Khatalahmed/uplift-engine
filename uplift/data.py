"""
data.py

WHAT : Loads the Criteo Uplift dataset (v2.1, ~14M rows from a randomized
       advertising trial) into Parquet and queries it with DuckDB SQL.
WHY  : 14M rows is too big to treat casually in pandas on a laptop.
       DuckDB scans the compressed CSV once, writes columnar Parquet, and
       answers the experiment summary in SQL without loading everything.
FLOW : download() -> to_parquet() -> arm_summary() / sample()
COLUMNS: f0..f11 (anonymized pre-treatment features), treatment (1 = ad
       exposure eligible, ~85%), exposure, visit, conversion.
NOTE : this is an ADVERTISING trial, not retention. It proves the methods
       work on real randomized data at scale; the retention story lives in
       the synthetic campaign (simulate.py) where the truth is known.
SOURCE: Criteo's own Hugging Face org (the old go.criteo.net link is
       dead). Licence CC BY-NC-SA 4.0 - never commit the data. Cite:
       Diemert et al., "A Large Scale Benchmark for Uplift Modeling",
       AdKDD @ KDD 2018.

USAGE
    python -m uplift.data download      # large file - see README first
    python -m uplift.data parquet
    python -m uplift.data summary
"""
import argparse
import hashlib
import urllib.request
from pathlib import Path

import duckdb
import pandas as pd

DATA_DIR = Path("data")
CSV_GZ = DATA_DIR / "criteo-uplift-v2.1.csv.gz"
PARQUET = DATA_DIR / "criteo-uplift-v2.1.parquet"
URL = ("https://huggingface.co/datasets/criteo/criteo-uplift/resolve/main/"
       "criteo-research-uplift-v2.1.csv.gz")
SHA256 = "2716e1bf0fd157a93b5bf86924d9088419dfbac2022c6cd90030220634f616dc"  # 311,422,618 bytes

FEATURES = [f"f{i}" for i in range(12)]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download(url: str = URL, dest: Path = CSV_GZ) -> Path:
    """Fetch once, then refuse to continue if the file is not the published one."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        print(f"downloading {url} -> {dest}")
        tmp = dest.with_suffix(".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(dest)
    digest = _sha256(dest)
    if digest != SHA256:
        raise RuntimeError(f"checksum mismatch for {dest}: {digest} - delete it and re-download")
    print(f"verified {dest} ({dest.stat().st_size:,} bytes)")
    return dest


def to_parquet(src: Path = CSV_GZ, dest: Path = PARQUET) -> Path:
    """CSV -> Parquet, adding a frozen `fold` (0-9) so every later split is
    reproducible: hashed row number, written once, never recomputed."""
    duckdb.sql(f"""
        COPY (
            SELECT *, CAST(hash(row_number() OVER ()) % 10 AS TINYINT) AS fold
            FROM read_csv_auto('{src.as_posix()}')
        ) TO '{dest.as_posix()}' (FORMAT PARQUET)
    """)
    return dest


def arm_summary(path: Path = PARQUET) -> pd.DataFrame:
    """Per-arm counts and outcome rates - the first table of any readout."""
    return duckdb.sql(f"""
        SELECT treatment,
               COUNT(*)                 AS n,
               SUM(visit)               AS visits,
               AVG(visit)               AS visit_rate,
               SUM(conversion)          AS conversions,
               AVG(conversion)          AS conversion_rate,
               AVG(exposure)            AS exposure_rate
        FROM '{path.as_posix()}'
        GROUP BY treatment ORDER BY treatment
    """).df()


def feature_balance(path: Path = PARQUET) -> pd.DataFrame:
    """Standardized mean difference per feature, computed in SQL over all rows."""
    parts = ", ".join(
        f"AVG({f}) FILTER (WHERE treatment = 1) AS {f}_m1, AVG({f}) FILTER (WHERE treatment = 0) AS {f}_m0, "
        f"VAR_SAMP({f}) FILTER (WHERE treatment = 1) AS {f}_v1, VAR_SAMP({f}) FILTER (WHERE treatment = 0) AS {f}_v0"
        for f in FEATURES)
    r = duckdb.sql(f"SELECT {parts} FROM '{path.as_posix()}'").df().iloc[0]
    rows = []
    for f in FEATURES:
        sd = ((r[f"{f}_v1"] + r[f"{f}_v0"]) / 2) ** 0.5
        smd = (r[f"{f}_m1"] - r[f"{f}_m0"]) / sd if sd > 0 else 0.0
        rows.append({"feature": f, "smd": float(smd), "imbalanced": bool(abs(smd) > 0.1)})
    return pd.DataFrame(rows)


def load_folds(folds: list[int], path: Path = PARQUET) -> pd.DataFrame:
    """Rows whose frozen fold is in `folds`; features as float32 to halve memory."""
    cols = ", ".join(f"CAST({f} AS FLOAT) AS {f}" for f in FEATURES)
    return duckdb.sql(f"SELECT {cols}, treatment, visit, conversion FROM '{path.as_posix()}' "
                      f"WHERE fold IN ({', '.join(map(str, folds))})").df()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["download", "parquet", "summary"])
    a = ap.parse_args()
    if a.step == "download":
        download()
    elif a.step == "parquet":
        print(to_parquet())
    else:
        print(arm_summary().to_string(index=False))


if __name__ == "__main__":
    main()
