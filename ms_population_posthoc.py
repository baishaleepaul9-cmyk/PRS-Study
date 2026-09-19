import os
import pandas as pd
import numpy as np
import scikit_posthocs as sp

# ============================================================
# MULTIPLE SCLEROSIS — DUNN POST-HOC TEST
# Holm correction for multiple comparisons
# ============================================================

BASE = r"C:\PRS_Study"

RESULTS_DIR = os.path.join(
    BASE,
    "results",
    "Multiple_Sclerosis"
)

INPUT_DIR = os.path.join(
    RESULTS_DIR,
    "population_analysis"
)

INPUT_FILE = os.path.join(
    INPUT_DIR,
    "MS_PRS_with_population_labels.tsv"
)

# ============================================================
# Output files
# ============================================================

PVALUES_OUTPUT = os.path.join(
    INPUT_DIR,
    "MS_population_Dunn_Holm_pvalues.tsv"
)

PAIRWISE_OUTPUT = os.path.join(
    INPUT_DIR,
    "MS_population_pairwise_Dunn_Holm.tsv"
)

SIGNIFICANT_OUTPUT = os.path.join(
    INPUT_DIR,
    "MS_population_significant_pairs.tsv"
)

# ============================================================
# Load data
# ============================================================

if not os.path.exists(INPUT_FILE):

    raise FileNotFoundError(
        f"Input file not found:\n{INPUT_FILE}"
    )

df = pd.read_csv(
    INPUT_FILE,
    sep="\t"
)

required_columns = [
    "IID",
    "PRS",
    "super_pop"
]

for col in required_columns:

    if col not in df.columns:

        raise ValueError(
            f"Required column '{col}' not found."
            f"\nColumns found: {list(df.columns)}"
        )

# ============================================================
# Population order
# ============================================================

population_order = [
    "AFR",
    "AMR",
    "EAS",
    "EUR",
    "SAS"
]

# ============================================================
# Check population counts
# ============================================================

print("=" * 70)
print("MULTIPLE SCLEROSIS — DUNN POST-HOC ANALYSIS")
print("=" * 70)

print("\nPopulation counts:")

print(
    df["super_pop"]
    .value_counts()
    .reindex(population_order)
)

# ============================================================
# Dunn's test with Holm correction
# ============================================================

print("\nRunning Dunn's test with Holm correction...")

dunn = sp.posthoc_dunn(
    df,
    val_col="PRS",
    group_col="super_pop",
    p_adjust="holm"
)

# Reorder rows and columns
dunn = dunn.reindex(
    index=population_order,
    columns=population_order
)

# ============================================================
# Save p-value matrix
# ============================================================

dunn.to_csv(
    PVALUES_OUTPUT,
    sep="\t"
)

# ============================================================
# Convert matrix to pairwise table
# ============================================================

pairwise_rows = []

for i, pop1 in enumerate(population_order):

    for j, pop2 in enumerate(population_order):

        if j <= i:
            continue

        p_value = dunn.loc[
            pop1,
            pop2
        ]

        pairwise_rows.append(
            {
                "Population_1": pop1,
                "Population_2": pop2,
                "Holm_adjusted_p": p_value
            }
        )

pairwise = pd.DataFrame(
    pairwise_rows
)

# ============================================================
# Add significance
# ============================================================

pairwise["Significant"] = (
    pairwise["Holm_adjusted_p"] < 0.05
)

pairwise["Significance"] = np.where(
    pairwise["Holm_adjusted_p"] < 0.001,
    "p < 0.001",
    np.where(
        pairwise["Holm_adjusted_p"] < 0.05,
        "p < 0.05",
        "NS"
    )
)

# ============================================================
# Save complete pairwise table
# ============================================================

pairwise.to_csv(
    PAIRWISE_OUTPUT,
    sep="\t",
    index=False
)

# ============================================================
# Significant pairs only
# ============================================================

significant = pairwise[
    pairwise["Significant"]
].copy()

significant.to_csv(
    SIGNIFICANT_OUTPUT,
    sep="\t",
    index=False
)

# ============================================================
# Print results
# ============================================================

print("\n")
print("=" * 70)
print("DUNN-HOLM PAIRWISE RESULTS")
print("=" * 70)

for _, row in pairwise.iterrows():

    p = row["Holm_adjusted_p"]

    if p == 0:

        p_text = "<1e-300"

    elif p < 0.001:

        p_text = f"{p:.6e}"

    else:

        p_text = f"{p:.6f}"

    print(
        f"{row['Population_1']} vs "
        f"{row['Population_2']}: "
        f"{p_text} "
        f"({row['Significance']})"
    )

# ============================================================
# Summary
# ============================================================

print("\n")
print("=" * 70)
print("SUMMARY")
print("=" * 70)

print(
    f"Total pairwise comparisons: {len(pairwise)}"
)

print(
    f"Significant comparisons: "
    f"{len(significant)}"
)

print(
    f"Non-significant comparisons: "
    f"{len(pairwise) - len(significant)}"
)

# ============================================================
# Output paths
# ============================================================

print("\n")
print("=" * 70)
print("OUTPUT FILES")
print("=" * 70)

print(
    f"P-value matrix:\n{PVALUES_OUTPUT}"
)

print(
    f"\nPairwise table:\n{PAIRWISE_OUTPUT}"
)

print(
    f"\nSignificant pairs:\n{SIGNIFICANT_OUTPUT}"
)

print("\nStatus: SUCCESS")
print("=" * 70)
