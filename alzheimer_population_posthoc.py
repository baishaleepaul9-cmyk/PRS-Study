import os
import pandas as pd
import numpy as np
import scikit_posthocs as sp
import matplotlib.pyplot as plt


# ============================================================
# ALZHEIMER'S PRS POPULATION POST-HOC ANALYSIS
# ============================================================
#
# Purpose:
# Identify which pairs of 1000 Genomes super-populations
# differ significantly in Alzheimer's PRS distribution.
#
# Statistical framework:
#   Kruskal-Wallis significant overall test
#   ->
#   Dunn's post-hoc test
#   ->
#   Holm multiple-testing correction
#
# No PRS values are recalculated.
# No VCFs are processed.
# No PLINK commands are executed.
# ============================================================


# ============================================================
# 1. PATHS
# ============================================================

INPUT_FILE = (
    r".\results\Alzheimer\population_analysis"
    r"\Alzheimer_PRS_with_population_labels.tsv"
)

OUT_DIR = r".\results\Alzheimer\population_analysis"

os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================
# 2. START
# ============================================================

print("=" * 75)
print("ALZHEIMER'S PRS POPULATION POST-HOC ANALYSIS")
print("=" * 75)

print("\nInput:")
print(INPUT_FILE)

print("\nOutput directory:")
print(OUT_DIR)


# ============================================================
# 3. CHECK INPUT
# ============================================================

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        f"Input file not found:\n{INPUT_FILE}"
    )


# ============================================================
# 4. LOAD DATA
# ============================================================

print("\n" + "-" * 75)
print("LOADING DATA")
print("-" * 75)

df = pd.read_csv(
    INPUT_FILE,
    sep="\t"
)

PRS_COL = "ALZHEIMERS_PRS"
POP_COL = "super_pop"

required_columns = [
    "#IID",
    PRS_COL,
    POP_COL
]

for col in required_columns:
    if col not in df.columns:
        raise ValueError(
            f"Required column missing: {col}"
        )


df[PRS_COL] = pd.to_numeric(
    df[PRS_COL],
    errors="coerce"
)

df[POP_COL] = df[POP_COL].astype(str).str.strip()


if df[PRS_COL].isna().any():
    raise ValueError(
        "Missing Alzheimer's PRS values detected."
    )


if df[POP_COL].isna().any():
    raise ValueError(
        "Missing population labels detected."
    )


print("Dataset shape:", df.shape)
print("Individuals:", len(df))
print("Missing PRS:", df[PRS_COL].isna().sum())
print("Missing population labels:", df[POP_COL].isna().sum())


# ============================================================
# 5. POPULATION ORDER
# ============================================================

population_order = [
    "AFR",
    "AMR",
    "EAS",
    "EUR",
    "SAS"
]


print("\nPopulation counts:")

for pop in population_order:

    n = (
        df[POP_COL] == pop
    ).sum()

    print(f"{pop}: {n}")


# ============================================================
# 6. DUNN'S POST-HOC TEST
# ============================================================

print("\n" + "=" * 75)
print("DUNN'S POST-HOC TEST")
print("=" * 75)

print("\nMultiple-testing correction: Holm")


dunn_matrix = sp.posthoc_dunn(
    df,
    val_col=PRS_COL,
    group_col=POP_COL,
    p_adjust="holm"
)


# Reorder rows and columns
dunn_matrix = dunn_matrix.loc[
    population_order,
    population_order
]


print("\nAdjusted P-value matrix:")
print(
    dunn_matrix.to_string(
        float_format=lambda x: f"{x:.6e}"
    )
)


# ============================================================
# 7. SAVE DUNN MATRIX
# ============================================================

matrix_file = os.path.join(
    OUT_DIR,
    "Alzheimer_population_Dunn_Holm_pvalues.tsv"
)

dunn_matrix.to_csv(
    matrix_file,
    sep="\t"
)

print("\nSaved:")
print(matrix_file)


# ============================================================
# 8. EXTRACT UNIQUE PAIRWISE COMPARISONS
# ============================================================

pairwise_results = []

for i in range(len(population_order)):

    for j in range(i + 1, len(population_order)):

        pop1 = population_order[i]
        pop2 = population_order[j]

        p_adj = dunn_matrix.loc[
            pop1,
            pop2
        ]

        pairwise_results.append(
            {
                "Population_1": pop1,
                "Population_2": pop2,
                "Holm_adjusted_P_value": p_adj,
                "Significant_at_0.05": p_adj < 0.05
            }
        )


pairwise_df = pd.DataFrame(
    pairwise_results
)


# ============================================================
# 9. ADD SIGNIFICANCE LABELS
# ============================================================

def significance_label(p):

    if p < 0.001:
        return "***"

    elif p < 0.01:
        return "**"

    elif p < 0.05:
        return "*"

    else:
        return "ns"


pairwise_df["Significance"] = (
    pairwise_df["Holm_adjusted_P_value"]
    .apply(significance_label)
)


# ============================================================
# 10. ADD MEAN DIFFERENCE
# ============================================================

means = (
    df
    .groupby(POP_COL)[PRS_COL]
    .mean()
)


pairwise_df["Mean_1"] = (
    pairwise_df["Population_1"]
    .map(means)
)

pairwise_df["Mean_2"] = (
    pairwise_df["Population_2"]
    .map(means)
)

pairwise_df["Mean_Difference"] = (
    pairwise_df["Mean_1"] -
    pairwise_df["Mean_2"]
)


# ============================================================
# 11. SORT BY P-VALUE
# ============================================================

pairwise_df = pairwise_df.sort_values(
    "Holm_adjusted_P_value"
).reset_index(drop=True)


print("\n" + "=" * 75)
print("PAIRWISE COMPARISONS")
print("=" * 75)

print(
    pairwise_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.6e}"
    )
)


# ============================================================
# 12. SAVE PAIRWISE RESULTS
# ============================================================

pairwise_file = os.path.join(
    OUT_DIR,
    "Alzheimer_population_pairwise_Dunn_Holm.tsv"
)

pairwise_df.to_csv(
    pairwise_file,
    sep="\t",
    index=False
)

print("\nSaved:")
print(pairwise_file)


# ============================================================
# 13. SIGNIFICANT PAIRS ONLY
# ============================================================

significant_pairs = pairwise_df[
    pairwise_df["Significant_at_0.05"]
].copy()


significant_file = os.path.join(
    OUT_DIR,
    "Alzheimer_population_significant_pairs.tsv"
)

significant_pairs.to_csv(
    significant_file,
    sep="\t",
    index=False
)


print("\n" + "-" * 75)
print("SIGNIFICANT PAIRWISE COMPARISONS")
print("-" * 75)

if len(significant_pairs) == 0:

    print("No significant pairwise differences after Holm correction.")

else:

    print(
        significant_pairs.to_string(
            index=False,
            float_format=lambda x: f"{x:.6e}"
        )
    )


print("\nSaved:")
print(significant_file)


# ============================================================
# 14. HEATMAP OF ADJUSTED P-VALUES
# ============================================================

print("\n" + "-" * 75)
print("GENERATING P-VALUE HEATMAP")
print("-" * 75)


# Convert very small values to a plotting floor
plot_matrix = dunn_matrix.copy()

plot_matrix = plot_matrix.replace(
    0,
    np.finfo(float).tiny
)


# Use -log10(p)
log_matrix = -np.log10(
    plot_matrix
)


plt.figure(
    figsize=(8, 7)
)

plt.imshow(
    log_matrix.values,
    aspect="auto"
)

plt.xticks(
    range(len(population_order)),
    population_order
)

plt.yticks(
    range(len(population_order)),
    population_order
)

plt.xlabel(
    "Super-population"
)

plt.ylabel(
    "Super-population"
)

plt.title(
    "Dunn's Test: −log10(Holm-adjusted P-values)"
)

plt.colorbar(
    label="−log10(adjusted P)"
)

plt.tight_layout()


heatmap_file = os.path.join(
    OUT_DIR,
    "Alzheimer_population_Dunn_Holm_heatmap.png"
)

plt.savefig(
    heatmap_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print("Saved:")
print(heatmap_file)


# ============================================================
# 15. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("FINAL SUMMARY")
print("=" * 75)

print(
    "Total pairwise comparisons:",
    len(pairwise_df)
)

print(
    "Significant comparisons after Holm correction:",
    len(significant_pairs)
)

print("\nNo PRS values were recalculated.")
print("No VCF files were processed.")
print("No PLINK commands were executed.")

print("\nAnalysis completed successfully.")
