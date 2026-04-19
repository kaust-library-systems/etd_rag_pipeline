# utils.py
# Shared utilities for the ETD_RAG pipeline code.
# Marcelo


import csv
from pathlib import Path
from os import PathLike
from urllib.parse import urlparse

def get_handler(handle_url: str) -> str:
    """Return just the handler from the handle url.
    Input:  http://hdl.handle.net/10754/224071
    Return: 10754/224071
    """
    return urlparse(handle_url).path[1:]

def get_file_metadata(
    filename: str,
    metadata_file: str | PathLike[str] = "metadata.csv",
    ) -> dict | None:
    """Look up metadata for a file in the CSV."""
    metadata_path = Path(metadata_file)

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

    lookup_stem = Path(filename).stem

    with open(metadata_path, encoding="utf-8", newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if Path(row["File"]).stem == lookup_stem:
                return dict(row)

    return None


