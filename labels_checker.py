#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple


REQUIRED_KEYS = [
    "region",
    "parent_tile",
    "tile_size",
    "crop",
    "image_path",
    "coord_convention",
    "num_nodes",
    "num_edges",
    "nodes",
    "edges",
]


def load_json(path: Path) -> Tuple[bool, Any, str]:
    """
    Load a JSON file.

    Returns:
        (success, data_or_None, error_message)
    """
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return True, data, ""
    except Exception as e:
        return False, None, str(e)


def validate_label_structure(label: Dict[str, Any], path: Path) -> Tuple[List[str], List[str]]:
    """
    Validate that a label dict matches the expected schema and is internally consistent.

    Returns:
        (errors, warnings)
    """
    errors: List[str] = []
    warnings: List[str] = []

    # Required keys
    for key in REQUIRED_KEYS:
        if key not in label:
            errors.append(f"Missing required key '{key}'")
    if errors:
        return errors, warnings

    # Basic type checks
    if not isinstance(label["nodes"], list):
        errors.append("Field 'nodes' is not a list")
        return errors, warnings

    if not isinstance(label["edges"], list):
        errors.append("Field 'edges' is not a list")
        return errors, warnings

    num_nodes = label["num_nodes"]
    num_edges = label["num_edges"]

    if not isinstance(num_nodes, int):
        errors.append(f"'num_nodes' is not an int (got {type(num_nodes)})")
    if not isinstance(num_edges, int):
        errors.append(f"'num_edges' is not an int (got {type(num_edges)})")

    if errors:
        return errors, warnings

    # num_nodes / num_edges consistency
    if len(label["nodes"]) != num_nodes:
        errors.append(
            f"'num_nodes'={num_nodes} but len(nodes)={len(label['nodes'])}"
        )

    if len(label["edges"]) != num_edges:
        errors.append(
            f"'num_edges'={num_edges} but len(edges)={len(label['edges'])}"
        )

    # Node idx checks
    idx_values: List[int] = []
    for i, node in enumerate(label["nodes"]):
        if "idx" not in node:
            errors.append(f"Node at position {i} missing 'idx'")
            continue
        idx = node["idx"]
        if not isinstance(idx, int):
            errors.append(f"Node at position {i} has non-int 'idx' ({idx})")
            continue
        idx_values.append(idx)

    if not errors:
        unique_idxs = set(idx_values)
        expected_idxs = set(range(num_nodes))
        if unique_idxs != expected_idxs:
            errors.append(
                f"Node 'idx' values are not exactly 0..{num_nodes-1} "
                f"(got {sorted(unique_idxs)})"
            )

    # Edge index range checks
    for i, edge in enumerate(label["edges"]):
        for key in ("src_idx", "dst_idx"):
            if key not in edge:
                errors.append(f"Edge at position {i} missing '{key}'")
                continue
            idx_val = edge[key]
            if not isinstance(idx_val, int):
                errors.append(
                    f"Edge at position {i} has non-int {key} ({idx_val})"
                )
                continue
            if not (0 <= idx_val < num_nodes):
                errors.append(
                    f"Edge at position {i} has {key}={idx_val} out of range "
                    f"[0, {num_nodes-1}]"
                )

    # Image path existence (warning only)
    image_path = Path(label["image_path"])
    if not image_path.exists():
        warnings.append(f"image_path does not exist on disk: {image_path}")

    return errors, warnings


def find_json_files(root: Path) -> List[Path]:
    """
    Recursively find all JSON files under root.
    """
    return sorted(root.rglob("*.json"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate RoadGraphPlus label JSON files."
    )
    parser.add_argument(
        "--root",
        type=str,
        default="data/labels",
        help="Root directory containing label JSONs (default: data/labels)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="label_validation_report.txt",
        help="Path to write a text report (default: label_validation_report.txt)",
    )
    args = parser.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        raise SystemExit(f"Root path does not exist or is not a directory: {root}")

    json_files = find_json_files(root)
    if not json_files:
        print(f"No .json files found under {root}")
        return

    print(f"Found {len(json_files)} .json files under {root}")

    total = len(json_files)
    syntax_fail = 0
    struct_fail = 0
    with_issues: List[Tuple[Path, List[str], List[str]]] = []

    for i, path in enumerate(json_files, start=1):
        ok, data, err = load_json(path)
        if not ok:
            syntax_fail += 1
            with_issues.append((path, [f"JSON decode error: {err}"], []))
        else:
            errors, warnings = validate_label_structure(data, path)
            if errors or warnings:
                if errors:
                    struct_fail += 1
                with_issues.append((path, errors, warnings))

        if i % 100 == 0 or i == total:
            print(f"Checked {i}/{total} files...", flush=True)

    print("\nValidation summary:")
    print(f"  Total label files:           {total}")
    print(f"  JSON syntax failures:        {syntax_fail}")
    print(f"  Structural validation errors:{struct_fail}")
    print(f"  Files with any issues:       {len(with_issues)}")

    # Write detailed report
    out_path = Path(args.output)
    with out_path.open("w", encoding="utf-8") as f:
        for path, errors, warnings in with_issues:
            f.write(f"=== {path} ===\n")
            if errors:
                f.write("  ERRORS:\n")
                for e in errors:
                    f.write(f"    - {e}\n")
            if warnings:
                f.write("  WARNINGS:\n")
                for w in warnings:
                    f.write(f"    - {w}\n")
            f.write("\n")

    print(f"\nDetailed report written to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
