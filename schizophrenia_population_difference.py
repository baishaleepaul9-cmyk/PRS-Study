import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import kruskal


# ============================================================
# SCHIZOPHRENIA — POPULATION-SPECIFIC PRS ANALYSIS
# ============================================================

BASE = Path(r"C:\PRS_Study")

# ------------------------------------------------------------
# Input files
# ------------------------------------------------------------

PRS_FILE = (
    BASE
    / "results"
    / "Schizophrenia"
    / "Schizophrenia_genomewide_PRS_with_risk_groups.tsv"
)

POPULATION_FILE = (
    BASE
    / "data"
    / "1000G"
    / "1000G_phase3_population_panel.txt"
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

POP_STATS_FILE = (
    OUTPUT_DIR
    / "Schizophrenia_population_PRS_by_superpopulation.tsv"
)

KW_FILE = (
    OUTPUT_DIR
    / "Schizophrenia_population_Kruskal_Wallis.tsv"
)

LABELED_FILE = (
    OUTPUT_DIR
    / "Schizophrenia_PRS_with_population_labels.tsv"
)


# ============================================================
# 1. START
# ============================================================

print("=" * 70)
print("SCHIZOPHRENIA POPULATION-SPECIFIC PRS ANALYSIS")
print("=" * 70)


# ============================================================
# 2. CHECK INPUT FILES
# ============================================================

if not PRS_FILE.exists():
    raise FileNotFoundError(
        f"\nPRS file not found:\n{PRS_FILE}"
    )

if not POPULATION_FILE.exists():
    raise FileNotFoundError(
        f"\nPopulation panel not found:\n{POPULATION_FILE}"
    )

print("\nPRS file:")
print(PRS_FILE)

print("\nPopulation panel:")
print(POPULATION_FILE)


# ============================================================
# 3. LOAD PRS DATA
# ============================================================

prs = pd.read_csv(
    PRS_FILE,
    sep="\t"
)

print("\nPRS data loaded.")
print(f"Shape: {prs.shape}")
print(f"Columns: {list(prs.columns)}")


# ============================================================
# 4. LOAD 1000 GENOMES POPULATION PANEL
# ============================================================

population = pd.read_csv(
    POPULATION_FILE,
    sep="\t"
)

print("\nPopulation panel loaded.")
print(f"Shape: {population.shape}")
print(f"Columns: {list(population.columns)}")


# ============================================================
# 5. CHECK REQUIRED COLUMNS
# ============================================================

required_prs_columns = [
    "IID",
    "PRS"
]

required_population_columns = [
    "sample",
    "pop",
    "super_pop",
    "gender"
]

for col in required_prs_columns:

    if col not in prs.columns:
        raise ValueError(
            f"Required PRS column '{col}' not found."
        )

for col in required_population_columns:

    if col not in population.columns:
        raise ValueError(
            f"Required population column '{col}' not found."
        )


# ============================================================
# 6. PREPARE SAMPLE IDs
# ============================================================

prs["IID"] = prs["IID"].astype(str)

population["sample"] = (
    population["sample"]
    .astype(str)
)


# ============================================================
# 7. CHECK SAMPLE ID MATCHING
# ============================================================

prs_ids = set(prs["IID"])

population_ids = set(
    population["sample"]
)

matched_ids = prs_ids.intersection(
    population_ids
)

missing_population_labels = (
    prs_ids - population_ids
)

extra_population_ids = (
    population_ids - prs_ids
)

print("\n" + "-" * 70)
print("SAMPLE ID MATCHING")
print("-" * 70)

print(
    f"PRS individuals              : "
    f"{len(prs_ids)}"
)

print(
    f"Population-panel individuals : "
    f"{len(population_ids)}"
)

print(
    f"Matched individuals          : "
    f"{len(matched_ids)}"
)

print(
    f"PRS IDs without population label : "
    f"{len(missing_population_labels)}"
)

print(
    f"Population IDs without PRS       : "
    f"{len(extra_population_ids)}"
)


# ============================================================
# 8. MERGE PRS WITH POPULATION INFORMATION
# ============================================================

merged = prs.merge(
    population[
        [
            "sample",
            "pop",
            "super_pop",
            "gender"
        ]
    ],
    left_on="IID",
    right_on="sample",
    how="inner",
    validate="one_to_one"
)

print("\nMerged dataset:")
print(f"Shape: {merged.shape}")


# ============================================================
# 9. VERIFY MERGE
# ============================================================

if len(merged) != len(prs):

    print(
        "\nWARNING: Not all PRS individuals were retained "
        "after merging."
    )

else:

    print(
        "\nAll PRS individuals successfully matched "
        "to population information."
    )


# ============================================================
# 10. SAVE LABELED DATASET
# ============================================================

merged.to_csv(
    LABELED_FILE,
    sep="\t",
    index=False
)

print("\nSaved population-labeled PRS file:")
print(LABELED_FILE)


# ============================================================
# 11. POPULATION COUNTS
# ============================================================

population_order = [
    "AFR",
    "AMR",
    "EAS",
    "EUR",
    "SAS"
]

print("\n" + "-" * 70)
print("SUPERPOPULATION SAMPLE COUNTS")
print("-" * 70)

population_counts = (
    merged["super_pop"]
    .value_counts()
    .reindex(
        population_order,
        fill_value=0
    )
)

for pop in population_order:

    print(
        f"{pop:<5} : "
        f"{population_counts[pop]}"
    )


# ============================================================
# 12. CALCULATE POPULATION-SPECIFIC PRS STATISTICS
# ============================================================

summary_rows = []

for pop in population_order:

    subset = merged.loc[
        merged["super_pop"] == pop,
        "PRS"
    ].dropna()

    if len(subset) == 0:
        continue

    summary_rows.append({

        "Disease": "Schizophrenia",

        "PRS_ID": "PGS002785",

        "Superpopulation": pop,

        "N": len(subset),

        "Mean_PRS": subset.mean(),

        "SD_PRS": subset.std(),

        "Median_PRS": subset.median(),

        "Min_PRS": subset.min(),

        "Max_PRS": subset.max(),

        "Q1_PRS": subset.quantile(0.25),

        "Q3_PRS": subset.quantile(0.75),

        "SE_PRS": (
            subset.std()
            / np.sqrt(len(subset))
        )
    })


population_stats = pd.DataFrame(
    summary_rows
)


# ============================================================
# 13. SAVE POPULATION STATISTICS
# ============================================================

population_stats.to_csv(
    POP_STATS_FILE,
    sep="\t",
    index=False
)

print("\nPopulation-specific statistics saved:")
print(POP_STATS_FILE)


# ============================================================
# 14. PRINT POPULATION STATISTICS
# ============================================================

print("\n" + "=" * 70)
print("POPULATION-SPECIFIC PRS STATISTICS")
print("=" * 70)

for _, row in population_stats.iterrows():

    print(
        f"\n{row['Superpopulation']}"
    )

    print(
        f"  N       : {int(row['N'])}"
    )

    print(
        f"  Mean    : {row['Mean_PRS']:.6f}"
    )

    print(
        f"  SD      : {row['SD_PRS']:.6f}"
    )

    print(
        f"  Median  : {row['Median_PRS']:.6f}"
    )

    print(
        f"  Min     : {row['Min_PRS']:.6f}"
    )

    print(
        f"  Max     : {row['Max_PRS']:.6f}"
    )

    print(
        f"  Q1      : {row['Q1_PRS']:.6f}"
    )

    print(
        f"  Q3      : {row['Q3_PRS']:.6f}"
    )

    print(
        f"  SE      : {row['SE_PRS']:.6f}"
    )


# ============================================================
# 15. KRUSKAL-WALLIS TEST
# ============================================================
#
# Tests whether PRS distributions differ across the
# five 1000 Genomes superpopulations.
#
# H0:
# PRS distributions are the same across groups.
#
# H1:
# At least one population differs.
# ============================================================

groups = []

valid_populations = []

for pop in population_order:

    values = merged.loc[
        merged["super_pop"] == pop,
        "PRS"
    ].dropna()

    if len(values) > 0:

        groups.append(values.values)

        valid_populations.append(pop)


if len(groups) < 2:

    raise ValueError(
        "At least two superpopulations are required "
        "for the Kruskal-Wallis test."
    )


H_statistic, p_value = kruskal(
    *groups
)


# ============================================================
# 16. SAVE KRUSKAL-WALLIS RESULT
# ============================================================

kw_result = pd.DataFrame([{

    "Disease": "Schizophrenia",

    "PRS_ID": "PGS002785",

    "Test": "Kruskal-Wallis",

    "Groups": len(valid_populations),

    "H_statistic": H_statistic,

    "P_value": p_value

}])


kw_result.to_csv(
    KW_FILE,
    sep="\t",
    index=False
)


# ============================================================
# 17. PRINT KRUSKAL-WALLIS RESULT
# ============================================================

print("\n" + "=" * 70)
print("KRUSKAL-WALLIS TEST")
print("=" * 70)

print(
    f"H statistic : "
    f"{H_statistic:.6f}"
)

print(
    f"P value     : "
    f"{p_value:.10e}"
)

if p_value < 0.001:

    print(
        "\nResult: PRS distributions differ significantly "
        "across superpopulations (p < 0.001)."
    )

elif p_value < 0.05:

    print(
        "\nResult: PRS distributions differ significantly "
        "across superpopulations (p < 0.05)."
    )

else:

    print(
        "\nResult: No statistically significant difference "
        "was detected across superpopulations."
    )


print("\nKruskal-Wallis result saved:")
print(KW_FILE)


# ============================================================
# 18. FINAL VERIFICATION
# ============================================================

print("\n" + "=" * 70)
print("FINAL VERIFICATION")
print("=" * 70)

print(
    f"Total PRS samples       : {len(prs)}"
)

print(
    f"Matched population IDs  : {len(matched_ids)}"
)

print(
    f"Merged samples          : {len(merged)}"
)

print(
    f"Superpopulations        : "
    f"{', '.join(valid_populations)}"
)

print(
    f"Kruskal-Wallis H        : "
    f"{H_statistic:.6f}"
)

print(
    f"Kruskal-Wallis p       : "
    f"{p_value:.10e}"
)

print("\nOutput directory:")
print(OUTPUT_DIR)

print("\n" + "=" * 70)
print("SCHIZOPHRENIA POPULATION ANALYSIS COMPLETE")
print("=" * 70)
