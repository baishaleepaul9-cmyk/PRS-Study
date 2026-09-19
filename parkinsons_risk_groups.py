import os
import pandas as pd

# ============================================================
# PARKINSON'S DISEASE — RELATIVE PRS GROUPS
# ============================================================

PROJECT = r"C:\PRS_Study"

RESULTS_DIR = os.path.join(
    PROJECT,
    "results",
    "Parkinsons"
)

INPUT_FILE = os.path.join(
    RESULTS_DIR,
    "Parkinsons_genomewide_PRS.tsv"
)

OUTPUT_FILE = os.path.join(
    RESULTS_DIR,
    "Parkinsons_genomewide_PRS_with_risk_groups.tsv"
)

SUMMARY_FILE = os.path.join(
    RESULTS_DIR,
    "Parkinsons_risk_group_summary.tsv"
)

print("=" * 80)
print("PARKINSON'S DISEASE — RELATIVE PRS GROUPING")
print("=" * 80)

# ============================================================
# Load genome-wide PRS
# ============================================================

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        f"Input file not found:\n{INPUT_FILE}"
    )

df = pd.read_csv(
    INPUT_FILE,
    sep="\t"
)

print(
    f"\nIndividuals loaded: {len(df):,}"
)

# ============================================================
# Basic checks
# ============================================================

if "IID" not in df.columns:
    raise ValueError("IID column not found.")

if "PRS" not in df.columns:
    raise ValueError("PRS column not found.")

df["PRS"] = pd.to_numeric(
    df["PRS"],
    errors="coerce"
)

if len(df) != 2504:
    raise ValueError(
        f"Expected 2504 individuals, found {len(df)}."
    )

if df["IID"].nunique() != 2504:
    raise ValueError(
        "Duplicate or missing individual IDs detected."
    )

if df["PRS"].isna().any():
    raise ValueError(
        "Missing PRS values detected."
    )

# ============================================================
# Calculate empirical thresholds
# ============================================================

p20 = df["PRS"].quantile(0.20)
p80 = df["PRS"].quantile(0.80)

print("\n" + "-" * 80)
print("EMPIRICAL PRS THRESHOLDS")
print("-" * 80)

print(
    f"20th percentile (P20): {p20:.10f}"
)

print(
    f"80th percentile (P80): {p80:.10f}"
)

# ============================================================
# Assign relative groups
# ============================================================

def assign_group(prs):

    if prs <= p20:
        return "Low"

    elif prs >= p80:
        return "High"

    else:
        return "Intermediate"


df["PRS_Group"] = df["PRS"].apply(
    assign_group
)

# ============================================================
# Group statistics
# ============================================================

group_order = [
    "Low",
    "Intermediate",
    "High"
]

summary_rows = []

for group in group_order:

    subset = df[
        df["PRS_Group"] == group
    ]

    n = len(subset)

    summary_rows.append(
        {
            "PRS_Group": group,
            "N": n,
            "Percentage": (
                n / len(df) * 100
            ),
            "Mean_PRS": subset["PRS"].mean(),
            "SD_PRS": subset["PRS"].std(),
            "Min_PRS": subset["PRS"].min(),
            "Max_PRS": subset["PRS"].max()
        }
    )

summary = pd.DataFrame(
    summary_rows
)

# ============================================================
# Print results
# ============================================================

print("\n" + "-" * 80)
print("RELATIVE PRS GROUPS")
print("-" * 80)

for _, row in summary.iterrows():

    print(
        f"{row['PRS_Group']:15s}"
        f" N = {int(row['N']):4d}"
        f" ({row['Percentage']:.6f}%)"
    )

print("\nGroup statistics:")
print(
    summary.to_string(
        index=False
    )
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
# Save summary
# ============================================================

summary.insert(
    0,
    "Disease",
    "Parkinson's disease"
)

summary.insert(
    1,
    "PGS_ID",
    "PGS000903"
)

summary.insert(
    2,
    "P20",
    p20
)

summary.insert(
    3,
    "P80",
    p80
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
print("PARKINSON'S RELATIVE PRS GROUPING COMPLETE")
print("=" * 80)

print(
    f"\nIndividual-level file:\n{OUTPUT_FILE}"
)

print(
    f"\nSummary file:\n{SUMMARY_FILE}"
)

print("\nSTATUS: SUCCESS")
