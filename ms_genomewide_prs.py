import os
import pandas as pd

# ============================================================
# MULTIPLE SCLEROSIS — GENOME-WIDE PRS
# Handles both:
#   1. #IID + SCORE1_SUM
#   2. IID + PRS
# ============================================================

BASE = r"C:\PRS_Study"

RESULTS_DIR = os.path.join(
    BASE,
    "results",
    "Multiple_Sclerosis"
)

OUTPUT_FILE = os.path.join(
    RESULTS_DIR,
    "MS_genomewide_PRS.tsv"
)

chromosome_data = []

# ============================================================
# LOAD CHROMOSOME PRS FILES
# ============================================================

for chrom in range(1, 23):

    file_path = os.path.join(
        RESULTS_DIR,
        f"Multiple_Sclerosis_chr{chrom}",
        f"MS_chr{chrom}_PRS.tsv"
    )

    if not os.path.exists(file_path):

        raise FileNotFoundError(
            f"Missing chromosome {chrom} PRS file:\n"
            f"{file_path}"
        )

    df = pd.read_csv(
        file_path,
        sep="\t"
    )

    print(
        f"\nChr{chrom} columns: {list(df.columns)}"
    )

    # --------------------------------------------------------
    # FORMAT 1: Raw PLINK score output
    # #IID + SCORE1_SUM
    # --------------------------------------------------------

    if "SCORE1_SUM" in df.columns:

        if "#IID" in df.columns:
            iid_column = "#IID"
        elif "IID" in df.columns:
            iid_column = "IID"
        else:
            raise ValueError(
                f"No IID column found in chromosome {chrom}."
            )

        chr_df = df[
            [iid_column, "SCORE1_SUM"]
        ].copy()

        chr_df.columns = [
            "IID",
            f"PRS_chr{chrom}"
        ]

        print(
            f"Chr{chrom}: using SCORE1_SUM"
        )

    # --------------------------------------------------------
    # FORMAT 2: Already simplified PRS file
    # IID + PRS
    # --------------------------------------------------------

    elif (
        "IID" in df.columns
        and "PRS" in df.columns
    ):

        chr_df = df[
            ["IID", "PRS"]
        ].copy()

        chr_df.columns = [
            "IID",
            f"PRS_chr{chrom}"
        ]

        print(
            f"Chr{chrom}: using existing PRS column"
        )

    else:

        raise ValueError(
            f"Unrecognized PRS format for chromosome {chrom}."
            f"\nColumns found: {list(df.columns)}"
        )

    # --------------------------------------------------------
    # Numeric conversion
    # --------------------------------------------------------

    chr_df[
        f"PRS_chr{chrom}"
    ] = pd.to_numeric(
        chr_df[f"PRS_chr{chrom}"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Sample count
    # --------------------------------------------------------

    if len(chr_df) != 2504:

        raise ValueError(
            f"Chromosome {chrom} has "
            f"{len(chr_df)} samples instead of 2504."
        )

    # --------------------------------------------------------
    # Duplicate IID check
    # --------------------------------------------------------

    if chr_df["IID"].duplicated().any():

        raise ValueError(
            f"Duplicate IID found in chromosome {chrom}."
        )

    # --------------------------------------------------------
    # Missing score check
    # --------------------------------------------------------

    missing = chr_df[
        f"PRS_chr{chrom}"
    ].isna().sum()

    if missing != 0:

        raise ValueError(
            f"Chromosome {chrom} has "
            f"{missing} missing PRS values."
        )

    chromosome_data.append(
        chr_df
    )

    print(
        f"Chr{chrom} loaded successfully: "
        f"{len(chr_df)} samples"
    )

# ============================================================
# MERGE ALL CHROMOSOMES
# ============================================================

print("\n")
print("=" * 70)
print("MERGING CHROMOSOME-LEVEL PRS")
print("=" * 70)

genomewide = chromosome_data[0]

for chr_df in chromosome_data[1:]:

    genomewide = genomewide.merge(
        chr_df,
        on="IID",
        how="inner"
    )

    print(
        f"Merged: {len(genomewide)} samples"
    )

# ============================================================
# VERIFY SAMPLE COUNT
# ============================================================

if len(genomewide) != 2504:

    raise ValueError(
        f"Final merged sample count is "
        f"{len(genomewide)}, expected 2504."
    )

# ============================================================
# CHROMOSOME PRS COLUMNS
# ============================================================

prs_columns = [
    f"PRS_chr{chrom}"
    for chrom in range(1, 23)
]

# ============================================================
# VERIFY NO MISSING VALUES
# ============================================================

total_missing = (
    genomewide[prs_columns]
    .isna()
    .sum()
    .sum()
)

if total_missing != 0:

    raise ValueError(
        f"Total missing chromosome PRS values: "
        f"{total_missing}"
    )

# ============================================================
# CALCULATE GENOME-WIDE PRS
# ============================================================

genomewide["PRS"] = (
    genomewide[prs_columns]
    .sum(axis=1)
)

# ============================================================
# FINAL TABLE
# ============================================================

final = genomewide[
    ["IID", "PRS"]
].copy()

# ============================================================
# SAVE
# ============================================================

final.to_csv(
    OUTPUT_FILE,
    sep="\t",
    index=False
)

# ============================================================
# SUMMARY STATISTICS
# ============================================================

print("\n")
print("=" * 70)
print("MULTIPLE SCLEROSIS GENOME-WIDE PRS")
print("=" * 70)

print(
    f"Samples: {len(final)}"
)

print(
    f"Mean: {final['PRS'].mean():.8f}"
)

print(
    f"SD: {final['PRS'].std():.8f}"
)

print(
    f"Median: {final['PRS'].median():.8f}"
)

print(
    f"Min: {final['PRS'].min():.8f}"
)

print(
    f"Q1: {final['PRS'].quantile(0.25):.8f}"
)

print(
    f"Q3: {final['PRS'].quantile(0.75):.8f}"
)

print(
    f"Max: {final['PRS'].max():.8f}"
)

print(
    f"Missing: {final['PRS'].isna().sum()}"
)

print("\nOutput:")
print(OUTPUT_FILE)

print("\nStatus: SUCCESS")
print("=" * 70)
