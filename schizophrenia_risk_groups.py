import pandas as pd
from pathlib import Path

# ============================================================
# SCHIZOPHRENIA — RELATIVE PRS RISK GROUP ASSIGNMENT
# ============================================================

BASE = Path(r"C:\PRS_Study")

INPUT = (
    BASE
    / "results"
    / "Schizophrenia"
    / "Schizophrenia_genomewide_PRS.tsv"
)

OUTPUT = (
    BASE
    / "results"
    / "Schizophrenia"
    / "Schizophrenia_genomewide_PRS_with_risk_groups.tsv"
)

SUMMARY = (
    BASE
    / "results"
    / "Schizophrenia"
    / "Schizophrenia_risk_group_summary.tsv"
)


# ============================================================
# 1. START
# ============================================================

print("=" * 70)
print("SCHIZOPHRENIA RELATIVE PRS RISK GROUP ANALYSIS")
print("=" * 70)


# ============================================================
# 2. CHECK INPUT FILE
# ============================================================

if not INPUT.exists():
    raise FileNotFoundError(
        f"\nInput file not found:\n{INPUT}"
    )

print(f"\nInput file:")
print(INPUT)


# ============================================================
# 3. LOAD GENOME-WIDE PRS
# ============================================================

df = pd.read_csv(INPUT, sep="\t")

print("\nLoaded data successfully.")
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")


# ============================================================
# 4. IDENTIFY PRS COLUMN
# ============================================================

possible_prs_columns = [
    "PRS",
    "SCORE1_SUM",
    "PRS_SUM"
]

prs_col = None

for col in possible_prs_columns:
    if col in df.columns:
        prs_col = col
        break

if prs_col is None:
    raise ValueError(
        "\nCould not identify the PRS column.\n"
        f"Available columns: {list(df.columns)}"
    )

print(f"\nPRS column identified: {prs_col}")


# ============================================================
# 5. CONVERT PRS TO NUMERIC
# ============================================================

df[prs_col] = pd.to_numeric(
    df[prs_col],
    errors="coerce"
)

missing = df[prs_col].isna().sum()

print(f"\nMissing PRS values: {missing}")

if missing > 0:
    print("Removing samples with missing PRS values...")
    df = df.dropna(subset=[prs_col]).copy()

print(f"Valid samples: {len(df)}")


# ============================================================
# 6. CALCULATE EMPIRICAL 20th AND 80th PERCENTILES
# ============================================================

p20 = df[prs_col].quantile(0.20)
p80 = df[prs_col].quantile(0.80)

print("\n" + "-" * 70)
print("EMPIRICAL PRS THRESHOLDS")
print("-" * 70)

print(f"20th percentile (P20): {p20:.10f}")
print(f"80th percentile (P80): {p80:.10f}")


# ============================================================
# 7. ASSIGN RELATIVE PRS GROUPS
# ============================================================
#
# Low          = bottom 20%
# Intermediate = middle 60%
# High         = top 20%
#
# These are relative population groups and NOT clinical
# disease-risk thresholds.
# ============================================================

def assign_group(x):

    if x <= p20:
        return "Low"

    elif x >= p80:
        return "High"

    else:
        return "Intermediate"


df["PRS_Risk_Group"] = df[prs_col].apply(assign_group)


# ============================================================
# 8. CALCULATE GROUP COUNTS
# ============================================================

group_order = [
    "Low",
    "Intermediate",
    "High"
]

counts = (
    df["PRS_Risk_Group"]
    .value_counts()
    .reindex(
        group_order,
        fill_value=0
    )
)

percentages = (
    counts / len(df) * 100
)


# ============================================================
# 9. PRINT GROUP DISTRIBUTION
# ============================================================

print("\n" + "-" * 70)
print("RELATIVE PRS GROUP DISTRIBUTION")
print("-" * 70)

for group in group_order:

    print(
        f"{group:<15} : "
        f"{counts[group]:>4} samples "
        f"({percentages[group]:.4f}%)"
    )


# ============================================================
# 10. CALCULATE PRS PERCENTILE POSITION
# ============================================================

df["PRS_Percentile"] = (
    df[prs_col]
    .rank(
        method="average",
        pct=True
    )
    * 100
)


# ============================================================
# 11. SAVE INDIVIDUAL-LEVEL RESULTS
# ============================================================

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    OUTPUT,
    sep="\t",
    index=False
)

print("\nIndividual-level results saved to:")

print(OUTPUT)


# ============================================================
# 12. CREATE GROUP SUMMARY
# ============================================================

summary_rows = []

for group in group_order:

    subset = df.loc[
        df["PRS_Risk_Group"] == group,
        prs_col
    ]

    summary_rows.append({

        "Disease": "Schizophrenia",

        "PRS_ID": "PGS002785",

        "Risk_Group": group,

        "N": len(subset),

        "Percentage": (
            len(subset) / len(df) * 100
        ),

        "Mean_PRS": subset.mean(),

        "SD_PRS": subset.std(),

        "Median_PRS": subset.median(),

        "Min_PRS": subset.min(),

        "Max_PRS": subset.max()
    })


# ============================================================
# 13. ADD OVERALL SUMMARY
# ============================================================

overall = pd.DataFrame([{

    "Disease": "Schizophrenia",

    "PRS_ID": "PGS002785",

    "Risk_Group": "Overall",

    "N": len(df),

    "Percentage": 100.0,

    "Mean_PRS": df[prs_col].mean(),

    "SD_PRS": df[prs_col].std(),

    "Median_PRS": df[prs_col].median(),

    "Min_PRS": df[prs_col].min(),

    "Max_PRS": df[prs_col].max()

}])


summary = pd.DataFrame(summary_rows)

summary = pd.concat(
    [
        summary,
        overall
    ],
    ignore_index=True
)


# ============================================================
# 14. SAVE SUMMARY TABLE
# ============================================================

summary.to_csv(
    SUMMARY,
    sep="\t",
    index=False
)

print("\nSummary table saved to:")

print(SUMMARY)


# ============================================================
# 15. FINAL VERIFICATION
# ============================================================

print("\n" + "=" * 70)
print("FINAL VERIFICATION")
print("=" * 70)

print(f"Total samples : {len(df)}")

print(
    f"Mean PRS     : "
    f"{df[prs_col].mean():.6f}"
)

print(
    f"SD PRS       : "
    f"{df[prs_col].std():.6f}"
)

print(
    f"Median PRS   : "
    f"{df[prs_col].median():.6f}"
)

print(
    f"Minimum PRS  : "
    f"{df[prs_col].min():.6f}"
)

print(
    f"Maximum PRS  : "
    f"{df[prs_col].max():.6f}"
)

print(
    f"P20          : "
    f"{p20:.6f}"
)

print(
    f"P80          : "
    f"{p80:.6f}"
)


# ============================================================
# 16. FINAL GROUP COUNTS
# ============================================================

print("\nRisk group counts:")

for group in group_order:

    print(
        f"{group:<15} : "
        f"{counts[group]}"
    )


# ============================================================
# 17. COMPLETION MESSAGE
# ============================================================

print("\n" + "=" * 70)
print("SCHIZOPHRENIA RISK GROUP ANALYSIS COMPLETE")
print("=" * 70)
