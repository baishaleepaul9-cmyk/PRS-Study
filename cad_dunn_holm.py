import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

# ============================================================
# PATHS
# ============================================================

BASE = Path(r"C:\PRS_Study")
CAD_DIR = BASE / "results" / "CAD"

INPUT_FILE = CAD_DIR / "CAD_PRS_with_population_labels.tsv"

OUTPUT_ALL = CAD_DIR / "CAD_population_Dunn_Holm_pvalues.tsv"
OUTPUT_PAIRS = CAD_DIR / "CAD_population_pairwise_Dunn_Holm.tsv"
OUTPUT_SIGNIFICANT = CAD_DIR / "CAD_population_significant_pairs.tsv"


# ============================================================
# SETTINGS
# ============================================================

POPULATIONS = ["AFR", "AMR", "EAS", "EUR", "SAS"]


# ============================================================
# 1. LOAD DATA
# ============================================================

print("=" * 90)
print("CAD DUNN–HOLM POST-HOC ANALYSIS")
print("=" * 90)

print("\nLoading population-labelled CAD PRS...")

df = pd.read_csv(INPUT_FILE, sep="\t")

print(f"Individuals: {len(df)}")
print(f"Populations: {sorted(df['super_pop'].unique())}")


# ============================================================
# 2. PREPARE GROUPS
# ============================================================

groups = {
    pop: df.loc[df["super_pop"] == pop, "PRS"].dropna()
    for pop in POPULATIONS
}

print("\nPopulation sample sizes:")

for pop in POPULATIONS:
    print(f"{pop}: {len(groups[pop])}")


# ============================================================
# 3. PAIRWISE MANN–WHITNEY U TESTS
# ============================================================

print("\n" + "-" * 90)
print("PAIRWISE COMPARISONS")
print("-" * 90)

results = []

for i in range(len(POPULATIONS)):
    for j in range(i + 1, len(POPULATIONS)):

        pop1 = POPULATIONS[i]
        pop2 = POPULATIONS[j]

        x = groups[pop1]
        y = groups[pop2]

        U, p = mannwhitneyu(
            x,
            y,
            alternative="two-sided"
        )

        results.append({
            "Population_1": pop1,
            "Population_2": pop2,
            "N_1": len(x),
            "N_2": len(y),
            "U_statistic": U,
            "Raw_p_value": p
        })

results_df = pd.DataFrame(results)


# ============================================================
# 4. HOLM MULTIPLE-TEST CORRECTION
# ============================================================

reject, p_adjusted, _, _ = multipletests(
    results_df["Raw_p_value"],
    method="holm"
)

results_df["Holm_adjusted_p"] = p_adjusted
results_df["Significant"] = reject


# ============================================================
# 5. DISPLAY RESULTS
# ============================================================

print("\nAll pairwise comparisons:")

display_cols = [
    "Population_1",
    "Population_2",
    "U_statistic",
    "Raw_p_value",
    "Holm_adjusted_p",
    "Significant"
]

print(
    results_df[display_cols].to_string(index=False)
)


# ============================================================
# 6. SIGNIFICANT PAIRS
# ============================================================

significant_df = results_df[
    results_df["Significant"] == True
].copy()

print("\n" + "-" * 90)
print("SIGNIFICANT PAIRS AFTER HOLM CORRECTION")
print("-" * 90)

if len(significant_df) > 0:
    print(
        significant_df[
            [
                "Population_1",
                "Population_2",
                "Holm_adjusted_p"
            ]
        ].to_string(index=False)
    )
else:
    print("No significant pairwise differences.")


# ============================================================
# 7. SAVE RESULTS
# ============================================================

results_df.to_csv(
    OUTPUT_ALL,
    sep="\t",
    index=False
)

results_df[
    [
        "Population_1",
        "Population_2",
        "N_1",
        "N_2",
        "U_statistic",
        "Raw_p_value",
        "Holm_adjusted_p",
        "Significant"
    ]
].to_csv(
    OUTPUT_PAIRS,
    sep="\t",
    index=False
)

significant_df.to_csv(
    OUTPUT_SIGNIFICANT,
    sep="\t",
    index=False
)


# ============================================================
# 8. SUMMARY
# ============================================================

print("\n" + "=" * 90)
print("CAD DUNN–HOLM ANALYSIS COMPLETE")
print("=" * 90)

print(f"\nTotal pairwise comparisons: {len(results_df)}")
print(f"Significant comparisons: {len(significant_df)}")
print(
    f"Non-significant comparisons: "
    f"{len(results_df) - len(significant_df)}"
)

print("\nOutputs:")

print(f"\nAll p-values:")
print(OUTPUT_ALL)

print(f"\nPairwise results:")
print(OUTPUT_PAIRS)

print(f"\nSignificant pairs:")
print(OUTPUT_SIGNIFICANT)
