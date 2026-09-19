import os
import pandas as pd
import numpy as np

ROOT = r"C:\PRS_Study"
SCHIZO = os.path.join(ROOT, "results", "Schizophrenia")

OUTPUT = os.path.join(
    SCHIZO,
    "Schizophrenia_genomewide_PRS.tsv"
)


# ============================================================
# LOAD ONE CHROMOSOME
# ============================================================

def load_chromosome(chrom):

    folder = os.path.join(
        SCHIZO,
        f"Schizophrenia_chr{chrom}"
    )

    score_file = os.path.join(
        folder,
        f"Schizophrenia_chr{chrom}_PRS.sscore"
    )

    if not os.path.exists(score_file):
        raise FileNotFoundError(
            f"Missing chromosome {chrom} score file:\n{score_file}"
        )

    df = pd.read_csv(
        score_file,
        sep=r"\s+"
    )

    print(
        f"Chr{chrom}: {len(df)} samples | "
        f"columns: {list(df.columns)}"
    )

    # --------------------------------------------------------
    # Identify sample ID column
    # --------------------------------------------------------

    if "IID" in df.columns:
        iid_col = "IID"

    elif "#IID" in df.columns:
        iid_col = "#IID"

    else:
        raise ValueError(
            f"No IID column found for chromosome {chrom}.\n"
            f"Columns: {list(df.columns)}"
        )

    # --------------------------------------------------------
    # Identify PRS column
    # --------------------------------------------------------

    if "SCORE1_SUM" in df.columns:
        score_col = "SCORE1_SUM"

    else:

        score_candidates = [
            c for c in df.columns
            if "SCORE" in c.upper()
            and "SUM" in c.upper()
        ]

        if len(score_candidates) == 1:
            score_col = score_candidates[0]

        else:
            raise ValueError(
                f"Could not identify PRS score column "
                f"for chromosome {chrom}.\n"
                f"Columns: {list(df.columns)}"
            )

    print(
        f"         ID column: {iid_col}"
    )

    print(
        f"         PRS column: {score_col}"
    )

    # --------------------------------------------------------
    # Keep required columns
    # --------------------------------------------------------

    temp = df[
        [iid_col, score_col]
    ].copy()

    temp = temp.rename(
        columns={
            iid_col: "IID",
            score_col: f"CHR{chrom}_PRS"
        }
    )

    # --------------------------------------------------------
    # Convert PRS to numeric
    # --------------------------------------------------------

    temp[f"CHR{chrom}_PRS"] = pd.to_numeric(
        temp[f"CHR{chrom}_PRS"],
        errors="coerce"
    )

    if temp[f"CHR{chrom}_PRS"].isna().any():

        raise ValueError(
            f"Missing/non-numeric PRS values "
            f"in chromosome {chrom}"
        )

    # --------------------------------------------------------
    # Check duplicate individuals
    # --------------------------------------------------------

    duplicates = temp["IID"].duplicated().sum()

    if duplicates != 0:

        raise ValueError(
            f"Chr{chrom} contains {duplicates} "
            f"duplicate IID values."
        )

    return temp


# ============================================================
# LOAD CHR1–CHR22
# ============================================================

print("=" * 70)
print("SCHIZOPHRENIA GENOME-WIDE PRS AGGREGATION")
print("=" * 70)

print("\nLoading chromosome score files...\n")

chromosome_scores = []

for chrom in range(1, 23):

    df = load_chromosome(chrom)

    if len(df) != 2504:

        raise ValueError(
            f"Chr{chrom}: expected 2504 samples, "
            f"found {len(df)}"
        )

    chromosome_scores.append(df)


# ============================================================
# MERGE CHROMOSOMES
# ============================================================

print("\n" + "=" * 70)
print("MERGING CHROMOSOMES")
print("=" * 70)

genomewide = chromosome_scores[0]

for chrom in range(1, 22):

    before = len(genomewide)

    genomewide = genomewide.merge(
        chromosome_scores[chrom],
        on="IID",
        how="inner",
        validate="one_to_one"
    )

    after = len(genomewide)

    print(
        f"Chr{chrom + 1}: {before} -> {after} individuals"
    )


# ============================================================
# FINAL SAMPLE CHECK
# ============================================================

print("\n" + "=" * 70)
print("FINAL SAMPLE QC")
print("=" * 70)

print(
    "Individuals after merging:",
    len(genomewide)
)

if len(genomewide) != 2504:

    raise ValueError(
        f"Expected 2504 individuals but found "
        f"{len(genomewide)}"
    )


# ============================================================
# SUM CHROMOSOME PRS
# ============================================================

chromosome_columns = [
    f"CHR{chrom}_PRS"
    for chrom in range(1, 23)
]

genomewide["PRS"] = genomewide[
    chromosome_columns
].sum(axis=1)


# ============================================================
# FINAL QC
# ============================================================

missing = genomewide["PRS"].isna().sum()

print(
    "Missing genome-wide PRS:",
    missing
)

if missing != 0:

    raise ValueError(
        "Missing genome-wide PRS values detected."
    )


# ============================================================
# SUMMARY
# ============================================================

prs = genomewide["PRS"]

print("\n" + "=" * 70)
print("SCHIZOPHRENIA GENOME-WIDE PRS SUMMARY")
print("=" * 70)

print(f"Mean:   {prs.mean():.6f}")
print(f"SD:     {prs.std():.6f}")
print(f"Median: {prs.median():.6f}")
print(f"Min:    {prs.min():.6f}")
print(f"Q1:     {prs.quantile(0.25):.6f}")
print(f"Q3:     {prs.quantile(0.75):.6f}")
print(f"Max:    {prs.max():.6f}")


# ============================================================
# SAVE
# ============================================================

genomewide[
    ["IID", "PRS"]
].to_csv(
    OUTPUT,
    sep="\t",
    index=False
)

print("\nGenome-wide PRS saved to:")
print(OUTPUT)


# ============================================================
# FIRST 10
# ============================================================

print("\nFirst 10 individuals:")

print(
    genomewide[
        ["IID", "PRS"]
    ].head(10).to_string(index=False)
)


print("\n" + "=" * 70)
print("GENOME-WIDE AGGREGATION COMPLETE")
print("=" * 70)
