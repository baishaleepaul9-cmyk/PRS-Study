import os
import pandas as pd
import numpy as np

# ============================================================
# MDD Relative PRS Risk Groups
# Empirical 20th / 80th percentile classification
# ============================================================

PROJECT = r"C:\PRS_Study"

RESULTS_DIR = os.path.join(
    PROJECT,
    "results",
    "MDD"
)

INPUT_FILE = os.path.join(
    RESULTS_DIR,
    "MDD_genomewide_PRS.tsv"
)

OUTPUT_FILE = os.path.join(
    RESULTS_DIR,
    "MDD_genomewide_PRS_with_risk_groups.tsv"
)

SUMMARY_FILE = os.path.join(
    RESULTS_DIR,
    "MDD_risk_group_summary.tsv"
)

# ============================================================
# Load genome-wide PRS
# ============================================================

print("\n" + "=" * 80)
print("MDD RELATIVE PRS RISK GROUPING")
print("=" * 80)

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        f"Missing input file:\n{INPUT_FILE}"
    )

df = pd.read_csv(
    INPUT_FILE,
    sep="\t"
)

# ------------------------------------------------------------
# Validate columns
# ------------------------------------------------------------

if "IID" not in df.columns:
    raise ValueError(
        f"IID column not found.\n"
        f"Columns: {df.columns.tolist()}"
    )

if "PRS" not in df.columns:
    raise ValueError(
        f"PRS column not found.\n"
        f"Columns: {df.columns.tolist()}"
    )

# ------------------------------------------------------------
# Numeric PRS
# ------------------------------------------------------------

df["PRS"] = pd.to_numeric(
    df["PRS"],
    errors="coerce"
)

# ============================================================
# Basic QC
# ============================================================

n = len(df)

unique_iids = df["IID"].nunique()

missing_prs = df["PRS"].isna().sum()

print(
    f"\nIndividuals:       {n:,}"
)

print(
    f"Unique IIDs:       {unique_iids:,}"
)

print(
    f"Missing PRS:       {missing_prs:,}"
)

if n != 2504:
    raise ValueError(
        f"Expected 2504 individuals, found {n}."
    )

if unique_iids != 2504:
    raise ValueError(
        f"Expected 2504 unique IIDs, found {unique_iids}."
    )

if missing_prs != 0:
    raise ValueError(
        f"Found {missing_prs} missing PRS values."
    )

# ============================================================
# Calculate empirical thresholds
# ============================================================

p20 = df["PRS"].quantile(0.20)
p80 = df["PRS"].quantile(0.80)

print(
    f"\n20th percentile (P20): "
    f"{p20:.10f}"
)

print(
    f"80th percentile (P80): "
    f"{p80:.10f}"
)

# ============================================================
# Assign relative PRS groups
# ============================================================

def assign_group(prs):

    if prs <= p20:
        return "Low"

    elif prs >= p80:
        return "High"

    else:
        return "Intermediate"


df["Risk_Group"] = df["PRS"].apply(
    assign_group
)

# ============================================================
# Group counts
# ============================================================

group_order = [
    "Low",
    "Intermediate",
    "High"
]

group_counts = (
    df["Risk_Group"]
    .value_counts()
    .reindex(
        group_order,
        fill_value=0
    )
)

print("\nRelative PRS group counts:")

for group in group_order:

    count = int(
        group_counts[group]
    )

    percentage = (
        count / n
    ) * 100

    print(
        f"  {group:<15} "
        f"{count:>5} "
        f"({percentage:.6f}%)"
    )

# ============================================================
# Verify expected total
# ============================================================

if group_counts.sum() != 2504:

    raise ValueError(
        "Risk-group counts do not sum to 2504."
    )

# ============================================================
# Save individual-level output
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    sep="\t",
    index=False
)

# ============================================================
# Create summary table
# ============================================================

summary_rows = []

for group in group_order:

    group_data = df[
        df["Risk_Group"] == group
    ]["PRS"]

    count = len(group_data)

    percentage = (
        count / n
    ) * 100

    summary_rows.append(
        {
            "Disease":
                "Major depressive disorder",

            "PGS_ID":
                "PGS000907",

            "Risk_Group":
                group,

            "N":
                count,

            "Percentage":
                percentage,

            "Threshold_P20":
                p20,

            "Threshold_P80":
                p80,

            "Mean_PRS":
                group_data.mean(),

            "SD_PRS":
                group_data.std(),

            "Median_PRS":
                group_data.median(),

            "Min_PRS":
                group_data.min(),

            "Max_PRS":
                group_data.max()
        }
    )

summary = pd.DataFrame(
    summary_rows
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
print("MDD RELATIVE PRS GROUP SUMMARY")
print("=" * 80)

print(
    f"P20 threshold: {p20:.10f}"
)

print(
    f"P80 threshold: {p80:.10f}"
)

for group in group_order:

    row = summary[
        summary["Risk_Group"] == group
    ].iloc[0]

    print(
        f"\n{group}:"
    )

    print(
        f"  N:       {int(row['N']):,}"
    )

    print(
        f"  Percent: {row['Percentage']:.6f}%"
    )

    print(
        f"  Mean:    {row['Mean_PRS']:.8f}"
    )

    print(
        f"  SD:      {row['SD_PRS']:.8f}"
    )

print("\n" + "=" * 80)
print("OUTPUT FILES")
print("=" * 80)

print(
    f"\nIndividual-level file:\n{OUTPUT_FILE}"
)

print(
    f"\nSummary file:\n{SUMMARY_FILE}"
)

print("\n" + "=" * 80)
print("MDD RELATIVE PRS GROUPING COMPLETE")
print("=" * 80)
