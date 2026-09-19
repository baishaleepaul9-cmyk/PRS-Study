import os
import pandas as pd
import numpy as np
from scipy.stats import kruskal
import matplotlib.pyplot as plt


# ============================================================
# ALZHEIMER'S PRS POPULATION-DIFFERENCE ANALYSIS
# ============================================================
#
# Purpose:
# Evaluate differences in Alzheimer's PRS distributions
# across the five 1000 Genomes super-populations.
#
# This script:
#   1. Loads the finalized genome-wide Alzheimer's PRS
#   2. Loads the official 1000 Genomes Phase 3 population panel
#   3. Validates sample-ID matching
#   4. Merges PRS with population information
#   5. Calculates population-specific descriptive statistics
#   6. Performs a Kruskal-Wallis test
#   7. Saves the merged dataset and statistical results
#   8. Generates publication-quality plots
#
# IMPORTANT:
# This analysis evaluates differences in PRS distributions.
# It does NOT estimate clinical disease risk.
#
# No VCFs are processed.
# No PLINK commands are executed.
# Existing PRS values are not recalculated or modified.
# ============================================================


# ============================================================
# 1. PATHS
# ============================================================

PRS_FILE = r".\results\Alzheimer\Alzheimer_genomewide_PRS_with_risk_groups.tsv"

PANEL_FILE = r".\data\1000G\1000G_phase3_population_panel.txt"

OUT_DIR = r".\results\Alzheimer\population_analysis"

os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================
# 2. START
# ============================================================

print("=" * 75)
print("ALZHEIMER'S PRS POPULATION-DIFFERENCE ANALYSIS")
print("=" * 75)

print("\nInput PRS file:")
print(PRS_FILE)

print("\nInput population panel:")
print(PANEL_FILE)

print("\nOutput directory:")
print(OUT_DIR)


# ============================================================
# 3. CHECK INPUT FILES
# ============================================================

if not os.path.exists(PRS_FILE):
    raise FileNotFoundError(
        f"PRS file not found:\n{PRS_FILE}"
    )

if not os.path.exists(PANEL_FILE):
    raise FileNotFoundError(
        f"Population panel not found:\n{PANEL_FILE}"
    )


# ============================================================
# 4. LOAD DATA
# ============================================================

print("\n" + "-" * 75)
print("LOADING DATA")
print("-" * 75)

prs = pd.read_csv(
    PRS_FILE,
    sep="\t"
)

panel = pd.read_csv(
    PANEL_FILE,
    sep=r"\s+"
)

print("PRS shape:", prs.shape)
print("Population panel shape:", panel.shape)


# ============================================================
# 5. CHECK REQUIRED COLUMNS
# ============================================================

required_prs_columns = [
    "#IID",
    "ALZHEIMERS_PRS"
]

required_panel_columns = [
    "sample",
    "pop",
    "super_pop",
    "gender"
]

for col in required_prs_columns:
    if col not in prs.columns:
        raise ValueError(
            f"Required PRS column missing: {col}"
        )

for col in required_panel_columns:
    if col not in panel.columns:
        raise ValueError(
            f"Required population-panel column missing: {col}"
        )


# ============================================================
# 6. STANDARDIZE SAMPLE IDs
# ============================================================

prs["#IID"] = prs["#IID"].astype(str).str.strip()

panel["sample"] = panel["sample"].astype(str).str.strip()


# ============================================================
# 7. VALIDATE SAMPLE IDs
# ============================================================

print("\n" + "-" * 75)
print("SAMPLE-ID VALIDATION")
print("-" * 75)

prs_ids = set(prs["#IID"])
panel_ids = set(panel["sample"])

matched_ids = prs_ids.intersection(panel_ids)

missing_population_labels = prs_ids - panel_ids

extra_panel_ids = panel_ids - prs_ids

print("PRS individuals:", len(prs_ids))
print("Panel individuals:", len(panel_ids))
print("Matched IDs:", len(matched_ids))
print("Missing population labels:", len(missing_population_labels))
print("Extra panel IDs:", len(extra_panel_ids))


if len(missing_population_labels) > 0:
    print("\nERROR: Some PRS individuals have no population label.")

    print("\nFirst missing IDs:")
    for sample_id in sorted(missing_population_labels)[:10]:
        print(sample_id)

    raise ValueError(
        "Population metadata is incomplete."
    )


if len(extra_panel_ids) > 0:
    print(
        "\nNote: The panel contains IDs not present in the PRS dataset."
    )


# ============================================================
# 8. CHECK POPULATION PANEL
# ============================================================

print("\n" + "-" * 75)
print("1000 GENOMES SUPER-POPULATION COUNTS")
print("-" * 75)

population_counts = (
    panel["super_pop"]
    .value_counts()
    .sort_index()
)

print(population_counts)


expected_super_populations = [
    "AFR",
    "AMR",
    "EAS",
    "EUR",
    "SAS"
]

observed_super_populations = sorted(
    panel["super_pop"].dropna().unique()
)

print("\nObserved super-populations:")
print(observed_super_populations)


# ============================================================
# 9. MERGE PRS WITH POPULATION INFORMATION
# ============================================================

print("\n" + "-" * 75)
print("MERGING PRS WITH POPULATION INFORMATION")
print("-" * 75)

merged = prs.merge(
    panel[
        [
            "sample",
            "pop",
            "super_pop",
            "gender"
        ]
    ],
    left_on="#IID",
    right_on="sample",
    how="left",
    validate="one_to_one"
)

print("Merged dataset shape:", merged.shape)


# ============================================================
# 10. CHECK MERGE
# ============================================================

if merged["super_pop"].isna().any():

    missing_count = merged["super_pop"].isna().sum()

    raise ValueError(
        f"{missing_count} individuals have missing super-population labels."
    )


if merged["pop"].isna().any():

    missing_count = merged["pop"].isna().sum()

    raise ValueError(
        f"{missing_count} individuals have missing population labels."
    )


print("Population merge: SUCCESS")
print("Individuals after merge:", len(merged))


# ============================================================
# 11. CHECK PRS VALUES
# ============================================================

PRS_COL = "ALZHEIMERS_PRS"

merged[PRS_COL] = pd.to_numeric(
    merged[PRS_COL],
    errors="coerce"
)

missing_prs = merged[PRS_COL].isna().sum()

print("\nMissing Alzheimer's PRS values:", missing_prs)

if missing_prs > 0:
    raise ValueError(
        "Missing or invalid Alzheimer's PRS values detected."
    )


# ============================================================
# 12. SUPER-POPULATION COUNTS IN PRS DATASET
# ============================================================

print("\n" + "-" * 75)
print("SUPER-POPULATION COUNTS IN ANALYSIS DATASET")
print("-" * 75)

analysis_population_counts = (
    merged["super_pop"]
    .value_counts()
    .sort_index()
)

print(analysis_population_counts)


# ============================================================
# 13. POPULATION-SPECIFIC DESCRIPTIVE STATISTICS
# ============================================================

print("\n" + "=" * 75)
print("POPULATION-SPECIFIC ALZHEIMER'S PRS SUMMARY")
print("=" * 75)

summary = (
    merged
    .groupby("super_pop")[PRS_COL]
    .agg(
        N="count",
        Mean="mean",
        SD="std",
        Median="median",
        Min="min",
        Max="max"
    )
    .reset_index()
)


# Standard error
summary["SE"] = (
    summary["SD"] /
    np.sqrt(summary["N"])
)


# Reorder populations
summary["super_pop"] = pd.Categorical(
    summary["super_pop"],
    categories=expected_super_populations,
    ordered=True
)

summary = (
    summary
    .sort_values("super_pop")
    .reset_index(drop=True)
)


print(
    summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)


# ============================================================
# 14. SAVE DESCRIPTIVE STATISTICS
# ============================================================

summary_file = os.path.join(
    OUT_DIR,
    "Alzheimer_population_PRS_by_superpopulation.tsv"
)

summary.to_csv(
    summary_file,
    sep="\t",
    index=False
)

print("\nSaved:")
print(summary_file)


# ============================================================
# 15. KRUSKAL-WALLIS TEST
# ============================================================

print("\n" + "=" * 75)
print("KRUSKAL-WALLIS TEST")
print("=" * 75)

kw_groups = []

kw_population_names = []

for population in expected_super_populations:

    values = merged.loc[
        merged["super_pop"] == population,
        PRS_COL
    ].dropna().values

    if len(values) == 0:
        print(
            f"WARNING: No observations found for {population}"
        )
        continue

    kw_groups.append(values)
    kw_population_names.append(population)

    print(
        f"{population}: n = {len(values)}"
    )


if len(kw_groups) < 2:
    raise ValueError(
        "At least two populations are required for Kruskal-Wallis testing."
    )


H_statistic, p_value = kruskal(
    *kw_groups
)


print("\nKruskal-Wallis H statistic:")
print(f"{H_statistic:.6f}")

print("\nP-value:")
print(f"{p_value:.10e}")


if p_value < 0.05:

    significance_result = (
        "Statistically significant difference among "
        "super-population PRS distributions."
    )

else:

    significance_result = (
        "No statistically significant difference among "
        "super-population PRS distributions."
    )


print("\nInterpretation:")
print(significance_result)


# ============================================================
# 16. SAVE KRUSKAL-WALLIS RESULTS
# ============================================================

kw_results = pd.DataFrame(
    {
        "Test": ["Kruskal-Wallis"],
        "H_statistic": [H_statistic],
        "P_value": [p_value],
        "Number_of_groups": [len(kw_groups)],
        "Interpretation": [significance_result]
    }
)

kw_file = os.path.join(
    OUT_DIR,
    "Alzheimer_population_Kruskal_Wallis.tsv"
)

kw_results.to_csv(
    kw_file,
    sep="\t",
    index=False
)

print("\nSaved:")
print(kw_file)


# ============================================================
# 17. SAVE MERGED INDIVIDUAL-LEVEL DATA
# ============================================================

merged_file = os.path.join(
    OUT_DIR,
    "Alzheimer_PRS_with_population_labels.tsv"
)

merged.to_csv(
    merged_file,
    sep="\t",
    index=False
)

print("\nSaved:")
print(merged_file)


# ============================================================
# 18. BOXPLOT
# ============================================================

print("\n" + "-" * 75)
print("GENERATING BOXPLOT")
print("-" * 75)

plot_data = []

for population in expected_super_populations:

    values = merged.loc[
        merged["super_pop"] == population,
        PRS_COL
    ].dropna().values

    plot_data.append(values)


plt.figure(figsize=(9, 6))

plt.boxplot(
    plot_data,
    tick_labels=expected_super_populations,
    showfliers=False
)

plt.xlabel(
    "1000 Genomes Super-population"
)

plt.ylabel(
    "Alzheimer's PRS"
)

plt.title(
    "Alzheimer's PRS Distribution Across "
    "1000 Genomes Super-populations"
)

plt.tight_layout()


boxplot_file = os.path.join(
    OUT_DIR,
    "Alzheimer_PRS_by_superpopulation_boxplot.png"
)

plt.savefig(
    boxplot_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved:")
print(boxplot_file)


# ============================================================
# 19. MEAN ± SD PLOT
# ============================================================

print("\n" + "-" * 75)
print("GENERATING MEAN ± SD PLOT")
print("-" * 75)

summary_plot = summary.copy()

# Convert categorical ordering to strings for plotting
summary_plot["super_pop"] = (
    summary_plot["super_pop"]
    .astype(str)
)

x_positions = np.arange(
    len(summary_plot)
)

plt.figure(figsize=(9, 6))

plt.errorbar(
    x_positions,
    summary_plot["Mean"],
    yerr=summary_plot["SD"],
    fmt="o",
    capsize=5
)

plt.xticks(
    x_positions,
    summary_plot["super_pop"]
)

plt.xlabel(
    "1000 Genomes Super-population"
)

plt.ylabel(
    "Mean Alzheimer's PRS ± SD"
)

plt.title(
    "Mean Alzheimer's PRS Across "
    "1000 Genomes Super-populations"
)

plt.tight_layout()


mean_plot_file = os.path.join(
    OUT_DIR,
    "Alzheimer_PRS_population_mean_SD.png"
)

plt.savefig(
    mean_plot_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved:")
print(mean_plot_file)


# ============================================================
# 20. FINAL VALIDATION
# ============================================================

print("\n" + "=" * 75)
print("FINAL VALIDATION")
print("=" * 75)

print("Total individuals:", len(merged))
print("Population labels missing:", merged["super_pop"].isna().sum())
print("PRS values missing:", merged[PRS_COL].isna().sum())

print("\nPopulation counts:")
print(
    merged["super_pop"]
    .value_counts()
    .sort_index()
)

print("\nOverall PRS mean:")
print(f"{merged[PRS_COL].mean():.6f}")

print("\nOverall PRS SD:")
print(f"{merged[PRS_COL].std():.6f}")


# ============================================================
# 21. COMPLETE
# ============================================================

print("\n" + "=" * 75)
print("ANALYSIS COMPLETED SUCCESSFULLY")
print("=" * 75)

print("\nOutput files:")

print("1.", summary_file)
print("2.", kw_file)
print("3.", merged_file)
print("4.", boxplot_file)
print("5.", mean_plot_file)

print("\nNo PLINK commands were executed.")
print("No VCF files were processed.")
print("No existing PRS values were recalculated.")
