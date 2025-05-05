#!/usr/bin/env python3
"""
Convert area CSV (Design, Frontal Area (m²)) to DrivAerNet
projected_areas.txt YAML mapping.

Usage
-----
python convert_area_csv_to_yaml.py areas.csv
python convert_area_csv_to_yaml.py areas.csv -o projected_areas.txt
python convert_area_csv_to_yaml.py areas.csv --no_prefix
"""
from pathlib import Path
import pandas as pd
import yaml
import argparse


def csv_to_yaml(csv_path: Path,
                out_path: Path,
                add_prefix: bool = True,
                prefix: str = "combined_",
                suffix: str = ".stl") -> None:
    """Read CSV and write YAML-style txt required by DrivAerNet."""
    df = pd.read_csv(csv_path)

    # 构造字典：key -> area
    mapping = {}
    for _, row in df.iterrows():
        key = str(row["Design"])
        if add_prefix:
            key = f"{prefix}{key}{suffix}"
        mapping[key] = float(row["Frontal Area (m²)"])

    # 写入 YAML（flow_style=False => 每行一条记录）
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(mapping, f,
                       default_flow_style=False,
                       sort_keys=False,
                       allow_unicode=True)

    print(f"✓ Saved projected areas to: {out_path.resolve()}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert DrivAerNet area CSV to projected_areas.txt")
    parser.add_argument("csv_path", type=Path, help="CSV file with 'Design,Frontal Area (m²)'")
    parser.add_argument("-o", "--out_path", type=Path, default=Path("../projected_areas.txt"),
                        help="output txt (default: projected_areas.txt)")
    parser.add_argument("--no_prefix", action="store_true", help="do NOT wrap design with 'combined_' & '.stl'")
    args = parser.parse_args()

    csv_to_yaml(args.csv_path,
                args.out_path,
                add_prefix=not args.no_prefix)


if __name__ == "__main__":
    main()
