#!/usr/bin/env python3
"""Generate the JSON schema for the top-level Occupant profile model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
DEFAULT_OUTPUT = SRC_ROOT / "obgeneration" / "data" / "ob_profile.schema.json"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from obgeneration.model.occupant_profile import Occupant


def generate_schema() -> dict:
    return Occupant.model_json_schema()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the JSON schema for obgeneration.model.occupant_profile.Occupant."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Path to write the schema JSON. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Print the schema to stdout instead of writing a file.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    schema = generate_schema()
    rendered = json.dumps(schema, indent=2) + "\n"

    if args.stdout:
        sys.stdout.write(rendered)
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered)
    print(f"Wrote schema to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
