import pandas as pd
import glob
import os
import re

BASE = r".\results\Alzheimer"
OUT = os.path.join(BASE, "Alzheimer_genomewide_PRS.tsv")

# Find chromosome PRS files
files = sorted(
    glob.glob(os.path.join(BASE, "chr*", "chr*_raw_prs.tsv")),
    key=lambda x: int(re.search(r"chr(\d+)", x).group(1))
)

print("=" * 70)
print("COMBINING ALZHEIMER'S CHROMOSOME-WISE PRS")
print("=" * 70)

print(f"\nFiles found: {len(files)}")

if len(files) != 22:
    raise RuntimeError(f"Expected 22 chromosome files, found {len(files)}")

dfs = []

for f in files:
    chrom = re.search(r"chr(\d+)", f).group(1)

    df = pd.read_csv(f, sep="\t")

    print(f"Chr{chrom}: {len(df)} individuals | columns: {list(df.columns)}")

    if len(df) != 2504:
        raise RuntimeError(
            f"Chr{chrom} has {len(df)} individuals instead of 2504"
        )

    if "#IID" not in df.columns:
        raise RuntimeError(f"#IID column missing in Chr{chrom}")

    # Identify the PRS column
    prs_columns = [c for c in df.columns if c != "#IID"]

    if len(prs_columns) != 1:
        raise RuntimeError(
            f"Chr{chrom} should have exactly one PRS column, "
            f"found: {prs_columns}"
        )

    prs_col = prs_columns[0]

    # Rename to chromosome-specific name
    df = df[["#IID", prs_col]].copy()
    df.rename(columns={prs_col: f"CHR{chrom}_PRS"}, inplace=True)

    # Check duplicate individuals
    if df["#IID"].duplicated().any():
        raise RuntimeError(f"Duplicate #IID values found in Chr{chrom}")

    df.set_index("#IID", inplace=True)

    dfs.append(df)

# ----------------------------------------------------------------------
# Verify that all chromosomes contain the same individuals
# ----------------------------------------------------------------------

reference_ids = dfs[0].index

for i, df in enumerate(dfs[1:], start=2):
    if not df.index.equals(reference_ids):
        raise RuntimeError(
            f"Individual IDs/order do not match at chromosome {i}"
        )

print("\nAll chromosomes contain the same 2,504 individuals.")

# ----------------------------------------------------------------------
# Combine chromosome PRS
# ----------------------------------------------------------------------

combined = pd.concat(dfs, axis=1)

# Verify 22 chromosome columns
chr_columns = [f"CHR{i}_PRS" for i in range(1, 23)]

missing_columns = [
    c for c in chr_columns if c not in combined.columns
]

if missing_columns:
    raise RuntimeError(
        f"Missing chromosome PRS columns: {missing_columns}"
    )

# Calculate genome-wide PRS
combined["ALZHEIMERS_PRS"] = combined[chr_columns].sum(axis=1)

# ----------------------------------------------------------------------
# Final QC
# ----------------------------------------------------------------------

print("\n" + "=" * 70)
print("FINAL GENOME-WIDE PRS QC")
print("=" * 70)

print(f"\nIndividuals: {len(combined)}")
print(f"Chromosome PRS columns: {len(chr_columns)}")
print(f"Missing genome-wide PRS: {combined['ALZHEIMERS_PRS'].isna().sum()}")

print("\nGenome-wide Alzheimer's PRS distribution:")
print(combined["ALZHEIMERS_PRS"].describe())

# ----------------------------------------------------------------------
# Save
# ----------------------------------------------------------------------

combined.reset_index().to_csv(
    OUT,
    sep="\t",
    index=False
)

print("\nSaved:")
print(OUT)

print("\nFinal columns:")
print(list(combined.reset_index().columns))

print("\n" + "=" * 70)
print("GENOME-WIDE ALZHEIMER'S PRS CREATED SUCCESSFULLY")
print("=" * 70)
