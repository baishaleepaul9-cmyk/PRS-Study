import os
import pandas as pd
import numpy as np
from scipy.stats import kruskal

# ============================================================
# MULTIPLE SCLEROSIS — POPULATION/ANCESTRY ANALYSIS
# PGS002726
#
# Superpopulations:
# AFR = African
# AMR = Ad Mixed American
# EAS = East Asian
# EUR = European
# SAS = South Asian
# ============================================================

BASE = r"C:\PRS_Study"

RESULTS_DIR = os.path.join(
    BASE,
    "results",
    "Multiple_Sclerosis"
)

POPULATION_PANEL = os.path.join(
    BASE,
    "data",
    "1000G",
    "1000G_phase3_population_panel.txt"
)

PRS_FILE = os.path.join(
    RESULTS_DIR,
    "MS_genomewide_PRS_with_risk_groups.tsv"
)

OUTPUT_DIR = os.path.join(
    RESULTS_DIR,
    "population_analysis"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

# ============================================================
# Output files
# ============================================================

LABELED_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "MS_PRS_with_population_labels.tsv"
)

SUMMARY_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "MS_population_PRS_by_superpopulation.tsv"
)

KW_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "MS_population_Kruskal_Wallis.tsv"
)

# ============================================================
# STEP 1: Load PRS
# ============================================================

if not os.path.exists(PRS_FILE):

    raise FileNotFoundError(
        f"MS PRS file not found:\n{PRS_FILE}"
    )

prs = pd.read_csv(
    PRS_FILE,
    sep="\t"
)

required_prs = [
    "IID",
    "PRS"
]

for col in required_prs:

    if col not in prs.columns:

        raise ValueError(
            f"Required PRS column '{col}' not found."
            f"\nColumns found: {list(prs.columns)}"
        )

prs["PRS"] = pd.to_numeric(
    prs["PRS"],
    errors="coerce"
)

if prs["PRS"].isna().any():

    raise ValueError(
        "Missing PRS values detected."
    )

print("=" * 70)
print("MULTIPLE SCLEROSIS POPULATION ANALYSIS")
print("=" * 70)

print(
    f"PRS individuals: {len(prs)}"
)

# ============================================================
# STEP 2: Load 1000 Genomes population panel
# ============================================================

if not os.path.exists(POPULATION_PANEL):

    raise FileNotFoundError(
        f"Population panel not found:\n"
        f"{POPULATION_PANEL}"
    )

panel = pd.read_csv(
    POPULATION_PANEL,
    sep="\t"
)

print(
    f"Population panel individuals: {len(panel)}"
)

# ============================================================
# Check panel columns
# ============================================================

required_panel = [
    "sample",
    "pop",
    "super_pop",
    "gender"
]

for col in required_panel:

    if col not in panel.columns:

        raise ValueError(
            f"Required population-panel column "
            f"'{col}' not found."
            f"\nColumns found: {list(panel.columns)}"
        )

# ============================================================
# STEP 3: Rename sample ID to IID
# ============================================================

panel = panel.rename(
    columns={
        "sample": "IID"
    }
)

# ============================================================
# STEP 4: Check exact ID matching
# ============================================================

prs_ids = set(
    prs["IID"]
)

panel_ids = set(
    panel["IID"]
)

matched_ids = (
    prs_ids
    & panel_ids
)

missing_population_labels = (
    prs_ids
    - panel_ids
)

extra_panel_ids = (
    panel_ids
    - prs_ids
)

print("\nSample matching:")
print(
    f"PRS individuals: {len(prs_ids)}"
)
print(
    f"Panel individuals: {len(panel_ids)}"
)
print(
    f"Matched individuals: {len(matched_ids)}"
)
print(
    f"Missing population labels: "
    f"{len(missing_population_labels)}"
)
print(
    f"Extra panel individuals: "
    f"{len(extra_panel_ids)}"
)

if len(matched_ids) != len(prs_ids):

    raise ValueError(
        "Not all PRS individuals have population labels."
    )

# ============================================================
# STEP 5: Merge PRS + population labels
# ============================================================

merged = prs.merge(
    panel[
        [
            "IID",
            "pop",
            "super_pop",
            "gender"
        ]
    ],
    on="IID",
    how="left",
    validate="one_to_one"
)

# ============================================================
# Check population labels
# ============================================================

if merged["super_pop"].isna().any():

    raise ValueError(
        "Some individuals have missing superpopulation labels."
    )

# ============================================================
# Standard population order
# ============================================================

population_order = [
    "AFR",
    "AMR",
    "EAS",
    "EUR",
    "SAS"
]

observed_populations = sorted(
    merged["super_pop"].unique()
)

print("\nObserved superpopulations:")
print(observed_populations)

for population in population_order:

    if population not in observed_populations:

        raise ValueError(
            f"Expected superpopulation {population} "
            f"was not found."
        )

# ============================================================
# Save labeled individual-level data
# ============================================================

merged.to_csv(
    LABELED_OUTPUT,
    sep="\t",
    index=False
)

# ============================================================
# STEP 6: Population-specific descriptive statistics
# ============================================================

summary_rows = []

for population in population_order:

    values = merged.loc[
        merged["super_pop"] == population,
        "PRS"
    ]

    n = len(values)

    mean = values.mean()

    sd = values.std()

    median = values.median()

    minimum = values.min()

    maximum = values.max()

    q1 = values.quantile(0.25)

    q3 = values.quantile(0.75)

    se = sd / np.sqrt(n)

    summary_rows.append(
        {
            "Superpopulation": population,
            "N": n,
            "Mean_PRS": mean,
            "SD": sd,
            "Median": median,
            "Min": minimum,
            "Q1": q1,
            "Q3": q3,
            "Max": maximum,
            "SE": se
        }
    )

summary = pd.DataFrame(
    summary_rows
)

# ============================================================
# Print summary
# ============================================================

print("\n")
print("=" * 70)
print("POPULATION-SPECIFIC PRS")
print("=" * 70)

print(
    summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.8f}"
    )
)

# ============================================================
# Save summary
# ============================================================

summary.to_csv(
    SUMMARY_OUTPUT,
    sep="\t",
    index=False
)

# ============================================================
# STEP 7: Kruskal-Wallis test
# ============================================================

groups = [
    merged.loc[
        merged["super_pop"] == population,
        "PRS"
    ].values
    for population in population_order
]

H_statistic, p_value = kruskal(
    *groups
)

# ============================================================
# Report p-value safely
# ============================================================

if p_value == 0:

    p_display = "<1e-300"

elif p_value < 0.001:

    p_display = "<0.001"

else:

    p_display = f"{p_value:.10e}"

kw_result = pd.DataFrame(
    [
        {
            "Test": "Kruskal-Wallis",
            "H_statistic": H_statistic,
            "P_value": p_value,
            "P_value_reported": p_display,
            "Groups": len(population_order),
            "N_total": len(merged)
        }
    ]
)

# ============================================================
# Print Kruskal-Wallis result
# ============================================================

print("\n")
print("=" * 70)
print("KRUSKAL-WALLIS TEST")
print("=" * 70)

print(
    f"H statistic: {H_statistic:.8f}"
)

print(
    f"Raw p-value: {p_value:.10e}"
)

print(
    f"Reported p-value: {p_display}"
)

# ============================================================
# Save test result
# ============================================================

kw_result.to_csv(
    KW_OUTPUT,
    sep="\t",
    index=False
)

# ============================================================
# Final verification
# ============================================================

if len(merged) != 2504:

    raise ValueError(
        f"Expected 2504 merged individuals, "
        f"found {len(merged)}."
    )

population_counts = (
    merged["super_pop"]
    .value_counts()
    .reindex(population_order)
)

print("\n")
print("=" * 70)
print("POPULATION SAMPLE COUNTS")
print("=" * 70)

print(
    population_counts
)

print("\n")
print("=" * 70)
print("OUTPUT FILES")
print("=" * 70)

print(
    f"Individual-level:\n{LABELED_OUTPUT}"
)

print(
    f"\nPopulation summary:\n{SUMMARY_OUTPUT}"
)

print(
    f"\nKruskal-Wallis:\n{KW_OUTPUT}"
)

print("\nStatus: SUCCESS")
print("=" * 70)
