"""Command line entry point for the WaveSense data audit."""
from __future__ import annotations

import argparse
from pathlib import Path

from wavesense.audit import audit_folder
from wavesense.download import ARTICLES, fetch_sample


def main() -> None:
    parser = argparse.ArgumentParser(prog="wavesense", description="Audit Open Guided Waves CFRP recordings")
    commands = parser.add_subparsers(dest="command", required=True)

    sample = commands.add_parser("sample", help="download a small schema-audit sample")
    sample.add_argument("--condition", choices=sorted(ARTICLES), default="healthy")
    sample.add_argument("--count", type=int, default=1, help="maximum HDF5 members to retrieve")
    sample.add_argument("--max-member-mb", type=int, default=128)
    sample.add_argument("--data-dir", type=Path, default=Path("data/raw"))

    audit = commands.add_parser("audit", help="inspect local HDF5 schema and write an inventory")
    audit.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    audit.add_argument("--output-dir", type=Path, default=Path("outputs/audit"))

    subset = commands.add_parser('subset', help='download bounded 40 kHz prototype data across all five conditions')
    subset.add_argument('--count', type=int, default=12, help='acquisitions per condition, 4–40')
    subset.add_argument('--data-dir', type=Path, default=Path('data/raw'))
    prepare = commands.add_parser('train', help='build manifest, train baseline and run fixed holdout experiments')
    prepare.add_argument('--data-dir', type=Path, default=Path('data/raw'))
    prepare.add_argument('--output-dir', type=Path, default=Path('outputs/model'))
    commands.add_parser('prepare-demo',help='freeze the presentation model, fetch 40 unused test recordings and evaluate without changing the original baseline')

    args = parser.parse_args()
    if args.command == "sample":
        result = fetch_sample(args.data_dir, args.condition, args.count, args.max_member_mb)
        print(f"Retrieved {len(result['members'])} of {result['archive_hdf5_members']} HDF5 members.")
        print(f"Source article: {result['doi']}; purpose: {result['purpose']}.")
        print(f"Network bytes transferred: {result['network_bytes']:,}")
    elif args.command == "audit":
        result = audit_folder(args.data_dir, args.output_dir)
        print(f"Inspected {len(result['files'])} HDF5 files; errors: {len(result['errors'])}.")
        print(f"Inventory: {args.output_dir.resolve()}")
    elif args.command == 'subset':
        from wavesense.subset import fetch_subset
        fetch_subset(args.data_dir, args.count)
    elif args.command == 'train':
        from wavesense.ingest import build_manifest
        from wavesense.model import train
        manifest = build_manifest(args.data_dir, args.output_dir)
        result = train(manifest, args.output_dir)
        print(f"Model: {result['version']}; manifest errors: {len(manifest['errors'])}")
        print(result['evaluations'][0])
    elif args.command == 'prepare-demo':
        from wavesense.temperature import develop, evaluate_test
        from wavesense.demo_data import fetch_test
        develop()
        fetch_test()
        evaluate_test()


if __name__ == "__main__":
    main()
