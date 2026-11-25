"""Explore and audit WEFRA LIFE CSV deliveries (read-only).

Run from the project root:
    python src/explore_data.py

Finds delivery_1, delivery_2, delivery_3, ... automatically. The script
reports issues but NEVER edits input CSVs or writes to the database.
"""

from pathlib import Path
import re
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

PRIMARY_KEYS = {
    "products.csv": "product_id",
    "campaigns.csv": "campaign_id",
    "posts.csv": "post_id",
    "comments.csv": "comment_id",
}

RELATIONSHIPS = [
    ("campaigns.csv", "product_id", "products.csv"),
    ("posts.csv", "campaign_id", "campaigns.csv"),
    ("comments.csv", "post_id", "posts.csv"),
]


def delivery_sort_key(folder: Path):
    """Sort delivery_2 before delivery_10."""
    return int(folder.name.split("_")[-1])


def read_csv(path: Path) -> pd.DataFrame:
    # Keep IDs, dates and comment text as strings; preserve UTF-8 characters.
    return pd.read_csv(path, encoding="utf-8-sig", dtype="string")


def report_file(filename: str, df: pd.DataFrame) -> None:
    print(f"\n--- {filename} ---")
    print(f"Rows: {len(df)} | Columns: {len(df.columns)}")
    print("Column names:", df.columns.tolist())
    print("\nFirst 3 records:")
    print(df.head(3).to_string(index=False))
    print("\nMissing values per column:")
    print(df.isna().sum().to_string())
    print("\nExact duplicate rows:", int(df.duplicated().sum()))

    key = PRIMARY_KEYS[filename]
    if key not in df.columns:
        print(f"WARNING: Missing primary-key column {key!r}")
        return

    missing_key = df[key].isna() | df[key].str.strip().eq("").fillna(False)
    print("Missing/blank primary keys:", int(missing_key.sum()))

    repeated = df[df.duplicated(subset=[key], keep=False) & ~missing_key]
    print("Rows sharing a primary key:", len(repeated))
    print("Unique repeated primary keys:", repeated[key].nunique())
    if not repeated.empty:
        print("Examples of repeated IDs:")
        print(repeated.sort_values(key).head(6).to_string(index=False))

        # Distinguish exact repeats from different versions of one ID.
        distinct_versions = df.loc[~missing_key].drop_duplicates()
        conflicting_keys = distinct_versions.loc[
            distinct_versions.duplicated(subset=[key], keep=False), key
        ].nunique()
        print("IDs with differing row values:", conflicting_keys)

    if filename == "comments.csv" and "comment_text" in df:
        blank = df["comment_text"].isna() | df["comment_text"].str.strip().eq("")
        blank = blank.fillna(True)
        print("Blank/missing comment text:", int(blank.sum()))
        print("Examples:")
        visible = [col for col in ("comment_id", "post_id", "comment_text") if col in df]
        print(df.loc[blank, visible].head(5).to_string(index=False))


def main() -> None:
    if not DATA_DIR.is_dir():
        raise SystemExit(f"Data directory not found: {DATA_DIR}")

    folders = sorted(
        (p for p in DATA_DIR.iterdir() if p.is_dir() and re.fullmatch(r"delivery_\d+", p.name)),
        key=delivery_sort_key,
    )
    if not folders:
        raise SystemExit(f"No delivery_N folders found under {DATA_DIR}")

    # Parent IDs present in earlier deliveries (delivery 2 may omit products.csv).
    known_ids = {filename: set() for filename in PRIMARY_KEYS}

    for folder in folders:
        print("\n" + "=" * 65)
        print(f"DATA EXPLORATION + QUALITY CHECK: {folder.name}")
        print("=" * 65)
        datasets = {}

        for filename in PRIMARY_KEYS:
            path = folder / filename
            if not path.is_file():
                print(f"\n--- {filename}: Not supplied in this delivery ---")
                continue
            try:
                df = read_csv(path)
            except (OSError, UnicodeError, pd.errors.ParserError) as exc:
                print(f"\nERROR reading {filename}: {exc}")
                continue
            datasets[filename] = df
            report_file(filename, df)

        print("\n--- Relationship checks ---")
        for child_file, fk, parent_file in RELATIONSHIPS:
            child = datasets.get(child_file)
            if child is None:
                continue
            if fk not in child.columns:
                print(f"{child_file}: Missing foreign-key column {fk!r}")
                continue

            valid_parents = set(known_ids[parent_file])
            parent = datasets.get(parent_file)
            parent_key = PRIMARY_KEYS[parent_file]
            if parent is not None and parent_key in parent.columns:
                valid_parents.update(parent[parent_key].dropna().str.strip())

            fk_values = child[fk].str.strip()
            missing_fk = fk_values.isna() | fk_values.eq("").fillna(False)
            invalid = ~missing_fk & ~fk_values.isin(valid_parents)
            print(
                f"{child_file}.{fk}: "
                f"missing/blank={int(missing_fk.sum())}, "
                f"unresolved references={int(invalid.sum())}, "
                f"unique unresolved IDs={fk_values[invalid].nunique()}"
            )
            if invalid.any():
                print("Examples:", fk_values[invalid].dropna().unique()[:10].tolist())

        for filename, df in datasets.items():
            key = PRIMARY_KEYS[filename]
            if key in df.columns:
                known_ids[filename].update(df[key].dropna().str.strip())

    print("\nInspection complete. No CSV files or database records were changed.")


if __name__ == "__main__":
    main()
