import pandas as pd
from pathlib import Path
import scikit_posthocs as sp


# ============================================================
# SCHIZOPHRENIA — DUNN POST-HOC TEST WITH HOLM CORRECTION
# ============================================================

BASE = Path(r"C:\PRS_Study")

# ------------------------------------------------------------
# Input
# ------------------------------------------------------------

INPUT = (
    BASE
    / "results"
    / "Schizophrenia"
    / "population_analysis"
    / "Schizophrenia_PRS_with_population_labels.tsv"
)

# ------------------------------------------------------------
# Output directory
# ------------------------------------------------------------

OUTPUT_DIR = (
    BASE
    / "results"
    / "Schizophrenia"
    / "population_analysis"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# Output files
DUNN_MATRIX = (
    OUTPUT_DIR
    / "Schizophrenia_population_Dunn_Holm_pvalues.tsv"
)

PAIRWISE = (
    OUTPUT_DIR
    / "Schizophrenia_population_pairwise_Dunn_Holm.tsv"
)

SIGNIFICANT = (
    OUTPUT_DIR
    / "Schizophrenia_population_significant_pairs.tsv"
)


# ============================================================
# 1. START
# ============================================================

print("=" * 70)
print("SCHIZOPHRENIA DUNN POST-HOC ANALYSIS")
print("=" * 70)


# ============================================================
# 2. CHECK INPUT
# ============================================================

if not INPUT.exists():
    raise FileNotFoundError(
        f"\nInput file not found:\n{INPUT}"
    )

print("\nInput file:")
print(INPUT)


# ============================================================
# 3. LOAD DATA
# ============================================================

df = pd.read_csv(
    INPUT,
    sep="\t"
)

print("\nData loaded successfully.")
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")


# ============================================================
# 4. CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "PRS",
    "super_pop"
]

for col in required_columns:

    if col not in df.columns:

        raise ValueError(
            f"Required column '{col}' not found."
        )


# ============================================================
# 5. REMOVE MISSING VALUES
# ============================================================

df = df[
    [
        "PRS",
        "super_pop"
    ]
].dropna().copy()

print(
    f"\nSamples used for Dunn test: {len(df)}"
)


# ============================================================
# 6. DEFINE POPULATION ORDER
# ============================================================

population_order = [
    "AFR",
    "AMR",
    "EAS",
    "EUR",
    "SAS"
]

# Keep only expected populations
df = df[
    df["super_pop"].isin(
        population_order
    )
].copy()


# ============================================================
# 7. PRINT SAMPLE COUNTS
# ============================================================

print("\n" + "-" * 70)
print("SAMPLE COUNTS")
print("-" * 70)

counts = (
    df["super_pop"]
    .value_counts()
    .reindex(
        population_order,
        fill_value=0
    )
)

for pop in population_order:

    print(
        f"{pop:<5} : {counts[pop]}"
    )


# ============================================================
# 8. DUNN POST-HOC TEST
# ============================================================
#
# Dunn's test performs pairwise comparisons between
# superpopulations.
#
# Holm correction controls the family-wise error rate
# across the multiple pairwise comparisons.
# ============================================================

print("\n" + "-" * 70)
print("RUNNING DUNN TEST")
print("-" * 70)

dunn_matrix = sp.posthoc_dunn(
    df,
    val_col="PRS",
    group_col="super_pop",
    p_adjust="holm"
)

# Reorder rows and columns
dunn_matrix = dunn_matrix.reindex(
    index=population_order,
    columns=population_order
)

print("\nDunn-Holm adjusted p-value matrix:")
print(dunn_matrix)


# ============================================================
# 9. SAVE P-VALUE MATRIX
# ============================================================

dunn_matrix.to_csv(
    DUNN_MATRIX,
    sep="\t"
)

print(
    "\nSaved Dunn-Holm p-value matrix:"
)

print(DUNN_MATRIX)


# ============================================================
# 10. CREATE PAIRWISE TABLE
# ============================================================

pairwise_rows = []

for i in range(
    len(population_order)
):

    for j in range(
        i + 1,
        len(population_order)
    ):

        pop1 = population_order[i]
        pop2 = population_order[j]

        p = dunn_matrix.loc[
            pop1,
            pop2
        ]

        pairwise_rows.append({

            "Disease": "Schizophrenia",

            "PRS_ID": "PGS002785",

            "Population_1": pop1,

            "Population_2": pop2,

            "Holm_Adjusted_P": p,

            "Significant_0.05": (
                p < 0.05
            )
        })


pairwise = pd.DataFrame(
    pairwise_rows
)


# ============================================================
# 11. SAVE PAIRWISE RESULTS
# ============================================================

pairwise.to_csv(
    PAIRWISE,
    sep="\t",
    index=False
)

print(
    "\nSaved pairwise Dunn-Holm results:"
)

print(PAIRWISE)


# ============================================================
# 12. EXTRACT SIGNIFICANT PAIRS
# ============================================================

significant = pairwise[
    pairwise["Significant_0.05"] == True
].copy()


# Sort by adjusted p-value
significant = significant.sort_values(
    "Holm_Adjusted_P"
)


# ============================================================
# 13. SAVE SIGNIFICANT PAIRS
# ============================================================

significant.to_csv(
    SIGNIFICANT,
    sep="\t",
    index=False
)

print(
    "\nSaved significant population pairs:"
)

print(SIGNIFICANT)


# ============================================================
# 14. PRINT ALL PAIRWISE RESULTS
# ============================================================

print("\n" + "=" * 70)
print("PAIRWISE DUNN-HOLM RESULTS")
print("=" * 70)

for _, row in pairwise.iterrows():

    p = row["Holm_Adjusted_P"]

    if p < 0.001:

        p_display = "< 0.001"

    else:

        p_display = f"{p:.6g}"

    significance = (
        "SIGNIFICANT"
        if row["Significant_0.05"]
        else "NOT SIGNIFICANT"
    )

    print(
        f"{row['Population_1']} vs "
        f"{row['Population_2']} : "
        f"p = {p_display} "
        f"--> {significance}"
    )


# ============================================================
# 15. SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("POST-HOC SUMMARY")
print("=" * 70)

total_pairs = len(pairwise)

significant_pairs = len(
    significant
)

non_significant_pairs = (
    total_pairs - significant_pairs
)

print(
    f"Total pairwise comparisons : "
    f"{total_pairs}"
)

print(
    f"Significant pairs          : "
    f"{significant_pairs}"
)

print(
    f"Non-significant pairs      : "
    f"{non_significant_pairs}"
)


# ============================================================
# 16. FINAL VERIFICATION
# ============================================================

print("\n" + "-" * 70)
print("SIGNIFICANT PAIRS")
print("-" * 70)

if len(significant) == 0:

    print(
        "No significant pairwise differences "
        "were detected."
    )

else:

    for _, row in significant.iterrows():

        p = row["Holm_Adjusted_P"]

        if p < 0.001:

            p_display = "< 0.001"

        else:

            p_display = f"{p:.6g}"

        print(
            f"{row['Population_1']} vs "
            f"{row['Population_2']} : "
            f"Holm-adjusted p = {p_display}"
        )


# ============================================================
# 17. COMPLETION
# ============================================================

print("\n" + "=" * 70)
print("SCHIZOPHRENIA DUNN-HOLM ANALYSIS COMPLETE")
print("=" * 70)

print("\nOutput directory:")
print(OUTPUT_DIR)
