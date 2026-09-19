import os
import pandas as pd
import numpy as np

# ============================================================
# MULTIPLE SCLEROSIS — RELATIVE PRS GROUPING
# PGS002726
#
# Low          = bottom 20%
# Intermediate = middle 60%
# High         = top 20%
#
# These are relative population groups, NOT clinical
# disease-risk thresholds.
# ============================================================

BASE = r"C:\PRS_Study"

RESULTS_DIR = os.path.join(
    BASE,
    "results",
    "Multiple_Sclerosis"
)

INPUT_FILE = os.path.join(
    RESULTS_DIR,
    "MS_genomewide_PRS.tsv"
)

OUTPUT_FILE = os.path.join(
    RESULTS_DIR,
    "MS_genomewide_PRS_with_risk_groups.tsv"
)

SUMMARY_FILE = os.path.join(
    RESULTS_DIR,
    "MS_risk_group_summary.tsv"
)

# ============================================================
# Load genome-wide PRS
# ============================================================

if not os.path.exists(INPUT_FILE):

    raise FileNotFoundError(
        f"Genome-wide PRS file not found:\n{INPUT_FILE}"
    )

df = pd.read_csv(
    INPUT_FILE,
    sep="\t"
)

# ============================================================
# Check required columns
# ============================================================

required_columns = [
    "IID",
    "PRS"
]

for col in required_columns:

    if col not in df.columns:

        raise ValueError(
            f"Required column '{col}' not found."
            f"\nColumns found: {list(df.columns)}"
        )

# ============================================================
# Clean PRS
# ============================================================

df["PRS"] = pd.to_numeric(
    df["PRS"],
    errors="coerce"
)

missing = df["PRS"].isna().sum()

if missing != 0:

    raise ValueError(
        f"{missing} missing PRS values found."
    )

if len(df) != 2504:

    raise ValueError(
        f"Expected 2504 samples, found {len(df)}."
    )

# ============================================================
# Calculate empirical thresholds
# ============================================================

p20 = df["PRS"].quantile(0.20)
p80 = df["PRS"].quantile(0.80)

print("=" * 70)
print("MULTIPLE SCLEROSIS RELATIVE PRS GROUPING")
print("=" * 70)

print(
    f"Number of individuals: {len(df)}"
)

print(
    f"20th percentile: {p20:.10f}"
)

print(
    f"80th percentile: {p80:.10f}"
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
# Count groups
# ============================================================

group_order = [
    "Low",
    "Intermediate",
    "High"
]

group_counts = (
    df["Risk_Group"]
    .value_counts()
    .reindex(group_order)
    .fillna(0)
    .astype(int)
)

# ============================================================
# Percentages
# ============================================================

group_percentages = (
    group_counts / len(df) * 100
)

# ============================================================
# Print results
# ============================================================

print("\nRelative PRS groups:")

for group in group_order:

    print(
        f"{group}: "
        f"{group_counts[group]} "
        f"({group_percentages[group]:.6f}%)"
    )

# ============================================================
# Create summary table
# ============================================================

summary = pd.DataFrame(
    {
        "Disease": "Multiple Sclerosis",
        "PGS_ID": "PGS002726",
        "Risk_Group": group_order,
        "N": [
            group_counts[group]
            for group in group_order
        ],
        "Percentage": [
            group_percentages[group]
            for group in group_order
        ],
        "P20_Threshold": p20,
        "P80_Threshold": p80
    }
)

# ============================================================
# Save individual-level groups
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    sep="\t",
    index=False
)

# ============================================================
# Save summary
# ============================================================

summary.to_csv(
    SUMMARY_FILE,
    sep="\t",
    index=False
)

# ============================================================
# Final verification
# ============================================================

if df["Risk_Group"].isna().any():

    raise ValueError(
        "Some individuals were not assigned a risk group."
    )

if group_counts.sum() != 2504:

    raise ValueError(
        "Risk-group counts do not sum to 2504."
    )

print("\n")
print("=" * 70)
print("OUTPUT FILES")
print("=" * 70)

print(
    f"Individual-level:\n{OUTPUT_FILE}"
)

print(
    f"\nSummary:\n{SUMMARY_FILE}"
)

print("\nStatus: SUCCESS")
print("=" * 70)
