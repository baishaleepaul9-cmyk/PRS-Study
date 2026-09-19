import os
import pandas as pd
import numpy as np

# ============================================================
# PARKINSON'S DISEASE — GENOME-WIDE PRS AGGREGATION
# ============================================================

PROJECT = r"C:\PRS_Study"

RESULTS_DIR = os.path.join(
    PROJECT,
    "results",
    "Parkinsons"
)

OUTPUT_FILE = os.path.join(
    RESULTS_DIR,
    "Parkinsons_genomewide_PRS.tsv"
)

SUMMARY_FILE = os.path.join(
    RESULTS_DIR,
    "Parkinsons_genomewide_summary.tsv"
)

print("=" * 80)
print("PARKINSON'S DISEASE — GENOME-WIDE PRS AGGREGATION")
print("=" * 80)

chromosome_data = []

# ============================================================
# Read chromosome-level PRS
# ============================================================

for chrom in range(1, 23):

    prs_file = os.path.join(
        RESULTS_DIR,
        f"Parkinsons_chr{chrom}",
        f"Parkinsons_chr{chrom}_PRS.tsv"
    )

    if not os.path.exists(prs_file):

        raise FileNotFoundError(
            f"Missing chromosome PRS file:\n{prs_file}"
        )

    print(
        f"\nLoading Chr{chrom}..."
    )

    df = pd.read_csv(
        prs_file,
        sep="\t"
    )

    # --------------------------------------------------------
    # Handle IID column
    # --------------------------------------------------------

    if "IID" not in df.columns:

        if "#IID" in df.columns:
            df = df.rename(
                columns={"#IID": "IID"}
            )

        else:
            raise ValueError(
                f"IID column not found in Chr{chrom}"
            )

    # --------------------------------------------------------
    # Handle PRS column
    # --------------------------------------------------------

    if "PRS" not in df.columns:

        if "SCORE1_SUM" in df.columns:

            df["PRS"] = pd.to_numeric(
                df["SCORE1_SUM"],
                errors="coerce"
            )

        elif "SCORE1_AVG" in df.columns:

            df["PRS"] = pd.to_numeric(
                df["SCORE1_AVG"],
                errors="coerce"
            )

        else:

            raise ValueError(
                f"No PRS/SCORE1_SUM/SCORE1_AVG "
                f"column found in Chr{chrom}"
            )

    df["PRS"] = pd.to_numeric(
        df["PRS"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Basic checks
    # --------------------------------------------------------

    print(
        f"  Samples: {len(df):,}"
    )

    print(
        f"  Unique IIDs: {df['IID'].nunique():,}"
    )

    print(
        f"  Missing PRS: {df['PRS'].isna().sum()}"
    )

    if len(df) != 2504:

        raise ValueError(
            f"Chr{chrom} does not contain 2504 samples."
        )

    if df["IID"].nunique() != 2504:

        raise ValueError(
            f"Chr{chrom} contains duplicate/missing IIDs."
        )

    if df["PRS"].isna().any():

        raise ValueError(
            f"Chr{chrom} contains missing PRS values."
        )

    # --------------------------------------------------------
    # Rename PRS column
    # --------------------------------------------------------

    df = df[
        [
            "IID",
            "PRS"
        ]
    ].copy()

    df = df.rename(
        columns={
            "PRS": f"PRS_chr{chrom}"
        }
    )

    chromosome_data.append(
        df
    )


# ============================================================
# Check all sample IDs match
# ============================================================

print("\n" + "-" * 80)
print("CHECKING SAMPLE ID CONSISTENCY")
print("-" * 80)

reference_ids = set(
    chromosome_data[0]["IID"]
)

for chrom_index, df in enumerate(
    chromosome_data,
    start=1
):

    ids = set(
        df["IID"]
    )

    if ids != reference_ids:

        missing = reference_ids - ids
        extra = ids - reference_ids

        raise ValueError(
            f"Sample mismatch in Chr{chrom_index}.\n"
            f"Missing: {len(missing)}\n"
            f"Extra: {len(extra)}"
        )

    print(
        f"Chr{chrom_index}: "
        f"2504 matching individuals"
    )


# ============================================================
# Merge chromosome PRS
# ============================================================

print("\n" + "-" * 80)
print("MERGING CHROMOSOME-LEVEL PRS")
print("-" * 80)

genomewide = chromosome_data[0].copy()

for df in chromosome_data[1:]:

    genomewide = genomewide.merge(
        df,
        on="IID",
        how="inner",
        validate="one_to_one"
    )

print(
    f"Genome-wide individuals: "
    f"{len(genomewide):,}"
)


# ============================================================
# Calculate genome-wide PRS
# ============================================================

prs_columns = [
    f"PRS_chr{chrom}"
    for chrom in range(1, 23)
]

genomewide["PRS"] = genomewide[
    prs_columns
].sum(
    axis=1
)


# ============================================================
# Final QC
# ============================================================

missing_prs = int(
    genomewide["PRS"].isna().sum()
)

unique_iids = int(
    genomewide["IID"].nunique()
)

print("\n" + "-" * 80)
print("FINAL GENOME-WIDE QC")
print("-" * 80)

print(
    f"Individuals:       {len(genomewide):,}"
)

print(
    f"Unique IIDs:       {unique_iids:,}"
)

print(
    f"Missing PRS:       {missing_prs}"
)

if len(genomewide) != 2504:

    raise ValueError(
        "Genome-wide dataset does not contain 2504 individuals."
    )

if unique_iids != 2504:

    raise ValueError(
        "Genome-wide dataset contains duplicate IIDs."
    )

if missing_prs != 0:

    raise ValueError(
        "Genome-wide PRS contains missing values."
    )


# ============================================================
# Statistics
# ============================================================

prs = genomewide["PRS"]

mean_prs = prs.mean()
sd_prs = prs.std()
median_prs = prs.median()
min_prs = prs.min()
max_prs = prs.max()

q1 = prs.quantile(0.25)
q3 = prs.quantile(0.75)

p20 = prs.quantile(0.20)
p80 = prs.quantile(0.80)

print("\n" + "-" * 80)
print("GENOME-WIDE PARKINSON'S PRS")
print("-" * 80)

print(
    f"N:          {len(prs):,}"
)

print(
    f"Mean:       {mean_prs:.8f}"
)

print(
    f"SD:         {sd_prs:.8f}"
)

print(
    f"Median:     {median_prs:.8f}"
)

print(
    f"Min:        {min_prs:.8f}"
)

print(
    f"Q1:         {q1:.8f}"
)

print(
    f"Q3:         {q3:.8f}"
)

print(
    f"Max:        {max_prs:.8f}"
)

print(
    f"P20:        {p20:.8f}"
)

print(
    f"P80:        {p80:.8f}"
)


# ============================================================
# Save genome-wide PRS
# ============================================================

genomewide_output = genomewide[
    ["IID", "PRS"]
].copy()

genomewide_output.to_csv(
    OUTPUT_FILE,
    sep="\t",
    index=False
)


# ============================================================
# Save summary
# ============================================================

summary = pd.DataFrame(
    [
        {
            "Disease": "Parkinson's disease",
            "PGS_ID": "PGS000903",
            "N": len(prs),
            "Mean_PRS": mean_prs,
            "SD_PRS": sd_prs,
            "Median_PRS": median_prs,
            "Min_PRS": min_prs,
            "Q1_PRS": q1,
            "Q3_PRS": q3,
            "Max_PRS": max_prs,
            "P20": p20,
            "P80": p80,
            "Missing_PRS": missing_prs
        }
    ]
)

summary.to_csv(
    SUMMARY_FILE,
    sep="\t",
    index=False
)


# ============================================================
# Final
# ============================================================

print("\n" + "=" * 80)
print("PARKINSON'S GENOME-WIDE PRS COMPLETE")
print("=" * 80)

print(
    f"\nGenome-wide PRS saved to:\n{OUTPUT_FILE}"
)

print(
    f"\nSummary saved to:\n{SUMMARY_FILE}"
)

print(
    "\nSTATUS: SUCCESS"
)
