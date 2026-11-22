"""Check a local prediction CSV against Kaggle's official test and sample files."""

import argparse
import csv
import hashlib
from io import TextIOWrapper
from pathlib import Path
from zipfile import ZipFile


def verify(path):
    archive = Path(__file__).parent / "data" / "kagglewars.zip"
    with ZipFile(archive) as files:
        with TextIOWrapper(files.open("sample_submission (4).csv"), encoding="utf-8-sig", newline="") as file:
            header, *sample = csv.reader(file)
        with TextIOWrapper(files.open("test (1).csv"), encoding="utf-8-sig", newline="") as file:
            test = csv.reader(file)
            column = next(test).index("id")
            test_ids = [row[column] for row in test]
    with path.open(encoding="utf-8-sig", newline="") as file:
        columns, *rows = csv.reader(file)
    if columns != header or columns != ["id", "target"]:
        raise ValueError("Expected the official id,target header")
    if len(rows) != len(sample) or any(len(row) != 2 for row in rows):
        raise ValueError("Unexpected number of prediction rows or columns")
    ids = [row[0] for row in rows]
    if ids != [row[0] for row in sample] or ids != test_ids or len(set(ids)) != len(ids):
        raise ValueError("IDs differ from the official test and sample files")
    if {row[1] for row in rows} != set(map(str, range(11))):
        raise ValueError("Expected predictions covering all integer labels 0–10")
    print(f"{path.name}: {len(rows):,} verified rows, SHA-256 {hashlib.sha256(path.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    verify(parser.parse_args().csv)
