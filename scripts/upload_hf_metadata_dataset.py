#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FOLDER = ROOT / "data/hf_dataset"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload prepared metadata to Hugging Face Datasets.")
    parser.add_argument(
        "--repo-id",
        required=True,
        help="Hugging Face dataset repo id, e.g. username/embodiedai-kb-metadata",
    )
    parser.add_argument("--folder", type=Path, default=DEFAULT_FOLDER)
    parser.add_argument("--private", action="store_true", help="Create the dataset as private.")
    parser.add_argument(
        "--commit-message",
        default="Upload embodied AI literature metadata",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.folder.exists():
        raise SystemExit(
            f"Dataset folder not found: {args.folder}. "
            "Run scripts/prepare_hf_metadata_dataset.py first."
        )
    api = HfApi()
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="dataset",
        private=args.private,
        exist_ok=True,
    )
    api.upload_folder(
        folder_path=args.folder,
        repo_id=args.repo_id,
        repo_type="dataset",
        commit_message=args.commit_message,
    )
    print(f"Uploaded dataset to https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
