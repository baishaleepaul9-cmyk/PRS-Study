import os
import pandas as pd
import numpy as np
from scipy.stats import kruskal

# ============================================================
# PARKINSON'S DISEASE — POPULATION PRS DIFFERENCES
# ============================================================

PROJECT = r"C:\PRS_Study"

RESULTS_DIR = os.path.join(
    PROJECT,
    "results",
    "Parkinsons"
)

POPULATION_PANEL = os.path.join(
    PROJECT,
    "data",
    "1000G",
    "1000G_phase3_population_panel.txt"
)

PRS_FILE = os.path.join(
    RESULTS_DIR,
    "Parkinsons_genomewide_PRS_with_risk_groups.tsv"
)

POP_DIR = os.path.join(
    RESULTS_DIR,
    "population_analysis"
)

os.makedirs(
    POP_DIR,
    exist_ok=True
)

OUTPUT_LABELLED = os.path.join(
    POP_DIR,
    "Parkinsons_PRS_with_population_labels.tsv"
)

OUTPUT_STATS = os.path.join(
    POP_DIR,
    "Parkinsons_population_PRS_by_superpopulation.tsv"
)

OUTPUT_KW = os.path.join(
    POP_DIR,
    "Parkinsons_population_Kruskal_Wallis.tsv"
)

# ============================================================
# Load files
# ============================================================

print("=" * 80)
print("PARKINSON'S DISEASE — POPULATION PRS ANALYSIS")
print("=" * 80)

print("\nLoading Parkinson's genome-wide PRS...")

prs = pd.read_csv(
    PRS_FILE,
    sep="\t"
)

print(
    f"PRS individuals: {len(prs):,}"
)

print("\nLoading 1000 Genomes population panel...")

panel = pd.read_csv(
    POPULATION_PANEL,
    sep=r"\s+"
)

print(
    f"Population panel individuals: {len(panel):,}"
)

# ============================================================
# Check columns
# ============================================================

required_prs = [
    "IID",
    "PRS"
]

for col in required_prs:

    if col not in prs.columns:

        raise ValueError(
            f"Missing PRS column: {col}"
        )

required_panel = [
    "sample",
    "pop",
    "super_pop",
    "gender"
]

for col in required_panel:

    if col not in panel.columns:

        raise ValueError(
            f"Missing population column: {col}"
        )

# ============================================================
# Rename population sample ID
# ============================================================

panel = panel.rename(
    columns={
        "sample": "IID"
    }
)

# ============================================================
# Check duplicate IDs
# ============================================================

if prs["IID"].duplicated().any():

    raise ValueError(
        "Duplicate IIDs found in PRS file."
    )

if panel["IID"].duplicated().any():

    raise ValueError(
        "Duplicate sample IDs found in population panel."
    )

# ============================================================
# Check exact ID matching
# ============================================================

prs_ids = set(
    prs["IID"]
)

panel_ids = set(
    panel["IID"]
)

missing_labels = prs_ids - panel_ids
extra_panel = panel_ids - prs_ids

print("\n" + "-" * 80)
print("SAMPLE ID MATCHING")
print("-" * 80)

print(
    f"PRS individuals:       {len(prs_ids):,}"
)

print(
    f"Panel individuals:     {len(panel_ids):,}"
)

print(
    f"Missing population labels: {len(missing_labels)}"
)

print(
    f"Extra panel individuals:   {len(extra_panel)}"
)

if missing_labels:

    raise ValueError(
        "Some PRS individuals do not have population labels."
    )

if extra_panel:

    raise ValueError(
        "Population panel contains individuals not present in PRS."
    )

# ============================================================
# Merge
# ============================================================

print("\n" + "-" * 80)
print("MERGING PRS WITH POPULATION LABELS")
print("-" * 80)

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
    how="inner",
    validate="one_to_one"
)

print(
    f"Merged individuals: {len(merged):,}"
)

if len(merged) != 2504:

    raise ValueError(
        "Merged dataset does not contain 2504 individuals."
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

# Check unexpected populations

unexpected = set(
    merged["super_pop"]
) - set(
    population_order
)

if unexpected:

    raise ValueError(
        f"Unexpected superpopulation labels: {unexpected}"
    )

# ============================================================
# Population statistics
# ============================================================

print("\n" + "-" * 80)
print("POPULATION-SPECIFIC PRS STATISTICS")
print("-" * 80)

stats_rows = []

for population in population_order:

    subset = merged[
        merged["super_pop"] == population
    ]["PRS"].astype(float)

    n = len(subset)

    mean = subset.mean()
    sd = subset.std()
    median = subset.median()
    minimum = subset.min()
    maximum = subset.max()

    q1 = subset.quantile(0.25)
    q3 = subset.quantile(0.75)

    se = sd / np.sqrt(n)

    stats_rows.append(
        {
            "Superpopulation": population,
            "N": n,
            "Mean_PRS": mean,
            "SD_PRS": sd,
            "Median_PRS": median,
            "Min_PRS": minimum,
            "Q1_PRS": q1,
            "Q3_PRS": q3,
            "Max_PRS": maximum,
            "SE": se
        }
    )

    print(
        f"\n{population}"
    )

    print(
        f"  N:       {n}"
    )

    print(
        f"  Mean:    {mean:.8f}"
    )

    print(
        f"  SD:      {sd:.8f}"
    )

    print(
        f"  Median:  {median:.8f}"
    )

    print(
        f"  Min:     {minimum:.8f}"
    )

    print(
        f"  Max:     {maximum:.8f}"
    )

population_stats = pd.DataFrame(
    stats_rows
)

# ============================================================
# Kruskal-Wallis test
# ============================================================

print("\n" + "-" * 80)
print("KRUSKAL-WALLIS TEST")
print("-" * 80)

groups = []

for population in population_order:

    values = merged[
        merged["super_pop"] == population
    ]["PRS"].astype(float)

    groups.append(values)

H, p_value = kruskal(
    *groups
)

print(
    f"Kruskal-Wallis H = {H:.8f}"
)

print(
    f"Raw p-value      = {p_value:.8e}"
)

if p_value < 0.001:

    print(
        "Interpretation: p < 0.001"
    )

else:

    print(
        f"Interpretation: p = {p_value:.8f}"
    )

kw_results = pd.DataFrame(
    [
        {
            "Test": "Kruskal-Wallis",
            "H_statistic": H,
            "P_value": p_value,
            "Interpretation": (
                "p < 0.001"
                if p_value < 0.001
                else f"p = {p_value:.8f}"
            )
        }
    ]
)

# ============================================================
# Save outputs
# ============================================================

merged.to_csv(
    OUTPUT_LABELLED,
    sep="\t",
    index=False
)

population_stats.to_csv(
    OUTPUT_STATS,
    sep="\t",
    index=False
)

kw_results.to_csv(
    OUTPUT_KW,
    sep="\t",
    index=False
)

# ============================================================
# Final
# ============================================================

print("\n" + "=" * 80)
print("PARKINSON'S POPULATION PRS ANALYSIS COMPLETE")
print("=" * 80)

print(
    f"\nPopulation-labelled PRS:\n{OUTPUT_LABELLED}"
)

print(
    f"\nPopulation statistics:\n{OUTPUT_STATS}"
)

print(
    f"\nKruskal-Wallis result:\n{OUTPUT_KW}"
)

print("\nSTATUS: SUCCESS")
