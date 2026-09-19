import os
import pandas as pd
import numpy as np

# ============================================================
# MDD Genome-wide PRS Aggregation
# Chr1 + Chr2 ... Chr22
# ============================================================

PROJECT = r"C:\PRS_Study"

RESULTS_DIR = os.path.join(
    PROJECT,
    "results",
    "MDD"
)

# ============================================================
# Output files
# ============================================================

GENOMEWIDE_FILE = os.path.join(
    RESULTS_DIR,
    "MDD_genomewide_PRS.tsv"
)

SUMMARY_FILE = os.path.join(
    RESULTS_DIR,
    "MDD_genomewide_summary.tsv"
)


# ============================================================
# Storage
# ============================================================

chromosome_scores = []


# ============================================================
# Load chromosome-level PRS
# ============================================================

print("\n" + "=" * 80)
print("MDD GENOME-WIDE PRS AGGREGATION")
print("=" * 80)

for chrom in range(1, 23):

    chr_dir = os.path.join(
        RESULTS_DIR,
        f"MDD_chr{chrom}"
    )

    prs_file = os.path.join(
        chr_dir,
        f"MDD_chr{chrom}_PRS.tsv"
    )

    print(
        f"\nChecking Chr{chrom}..."
    )

    if not os.path.exists(prs_file):

        raise FileNotFoundError(
            f"\nMissing chromosome PRS file:\n"
            f"{prs_file}"
        )

    df = pd.read_csv(
        prs_file,
        sep="\t"
    )

    # --------------------------------------------------------
    # Check required columns
    # --------------------------------------------------------

    if "IID" not in df.columns:

        raise ValueError(
            f"Chr{chrom}: IID column not found.\n"
            f"Columns: {df.columns.tolist()}"
        )

    if "PRS" not in df.columns:

        raise ValueError(
            f"Chr{chrom}: PRS column not found.\n"
            f"Columns: {df.columns.tolist()}"
        )

    # --------------------------------------------------------
    # Numeric PRS
    # --------------------------------------------------------

    df["PRS"] = pd.to_numeric(
        df["PRS"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # QC
    # --------------------------------------------------------

    n_samples = len(df)

    missing = df["PRS"].isna().sum()

    unique_ids = df["IID"].nunique()

    print(
        f"  Samples:       {n_samples:,}"
    )

    print(
        f"  Unique IIDs:   {unique_ids:,}"
    )

    print(
        f"  Missing PRS:   {missing:,}"
    )

    if n_samples != 2504:

        raise ValueError(
            f"Chr{chrom}: expected 2504 samples, "
            f"found {n_samples}."
        )

    if unique_ids != 2504:

        raise ValueError(
            f"Chr{chrom}: expected 2504 unique IIDs, "
            f"found {unique_ids}."
        )

    if missing != 0:

        raise ValueError(
            f"Chr{chrom}: {missing} missing PRS values."
        )

    # --------------------------------------------------------
    # Rename chromosome score
    # --------------------------------------------------------

    df = df[
        [
            "IID",
            "PRS"
        ]
    ].copy()

    df.rename(
        columns={
            "PRS": f"CHR{chrom}_PRS"
        },
        inplace=True
    )

    chromosome_scores.append(
        df
    )


# ============================================================
# Merge all chromosomes
# ============================================================

print("\n" + "=" * 80)
print("MERGING CHROMOSOME-LEVEL PRS")
print("=" * 80)

genomewide = chromosome_scores[0].copy()

for chrom in range(2, 23):

    genomewide = genomewide.merge(
        chromosome_scores[chrom - 1],
        on="IID",
        how="inner",
        validate="one_to_one"
    )

    print(
        f"After Chr{chrom}: "
        f"{len(genomewide):,} individuals"
    )


# ============================================================
# Verify all 22 chromosome columns
# ============================================================

expected_columns = [
    f"CHR{chrom}_PRS"
    for chrom in range(1, 23)
]

missing_columns = [
    col
    for col in expected_columns
    if col not in genomewide.columns
]

if missing_columns:

    raise ValueError(
        "Missing chromosome columns:\n"
        + "\n".join(missing_columns)
    )


# ============================================================
# Verify sample count
# ============================================================

if len(genomewide) != 2504:

    raise ValueError(
        f"Expected 2504 individuals after merging, "
        f"found {len(genomewide)}."
    )

if genomewide["IID"].nunique() != 2504:

    raise ValueError(
        "Genome-wide dataset does not contain "
        "2504 unique individuals."
    )


# ============================================================
# Check missing chromosome scores
# ============================================================

total_missing = (
    genomewide[
        expected_columns
    ]
    .isna()
    .sum()
    .sum()
)

print(
    f"\nTotal missing chromosome scores: "
    f"{total_missing:,}"
)

if total_missing != 0:

    raise ValueError(
        "Missing chromosome-level PRS values detected."
    )


# ============================================================
# Calculate genome-wide PRS
# ============================================================

print(
    "\nCalculating genome-wide PRS..."
)

genomewide["PRS"] = (
    genomewide[
        expected_columns
    ]
    .sum(axis=1)
)


# ============================================================
# Final output
# ============================================================

final_output = genomewide[
    [
        "IID",
        "PRS"
    ]
].copy()

final_output.to_csv(
    GENOMEWIDE_FILE,
    sep="\t",
    index=False
)


# ============================================================
# Genome-wide statistics
# ============================================================

prs = final_output["PRS"]

N = len(prs)

mean_prs = prs.mean()
sd_prs = prs.std()
median_prs = prs.median()

min_prs = prs.min()
max_prs = prs.max()

q1 = prs.quantile(0.25)
q3 = prs.quantile(0.75)

p20 = prs.quantile(0.20)
p80 = prs.quantile(0.80)

missing_genomewide = prs.isna().sum()


# ============================================================
# Summary table
# ============================================================

summary = pd.DataFrame(
    [{
        "Disease": "Major depressive disorder",
        "PGS_ID": "PGS000907",
        "N": N,
        "Mean": mean_prs,
        "SD": sd_prs,
        "Median": median_prs,
        "Min": min_prs,
        "Q1": q1,
        "Q3": q3,
        "Max": max_prs,
        "P20": p20,
        "P80": p80,
        "Missing": int(
            missing_genomewide
        )
    }]
)

summary.to_csv(
    SUMMARY_FILE,
    sep="\t",
    index=False
)


# ============================================================
# Final report
# ============================================================

print("\n" + "=" * 80)
print("MDD GENOME-WIDE PRS SUMMARY")
print("=" * 80)

print(
    f"N:             {N:,}"
)

print(
    f"Mean:          {mean_prs:.8f}"
)

print(
    f"SD:            {sd_prs:.8f}"
)

print(
    f"Median:        {median_prs:.8f}"
)

print(
    f"Min:           {min_prs:.8f}"
)

print(
    f"Q1:            {q1:.8f}"
)

print(
    f"Q3:            {q3:.8f}"
)

print(
    f"Max:           {max_prs:.8f}"
)

print(
    f"20th percentile: {p20:.8f}"
)

print(
    f"80th percentile: {p80:.8f}"
)

print(
    f"Missing:        {missing_genomewide:,}"
)

print("\nOutput:")
print(GENOMEWIDE_FILE)

print("\nSummary:")
print(SUMMARY_FILE)

print("\n" + "=" * 80)
print("MDD GENOME-WIDE AGGREGATION COMPLETE")
print("=" * 80)
