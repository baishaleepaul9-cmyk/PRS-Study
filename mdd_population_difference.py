import os
import pandas as pd
import numpy as np
from scipy.stats import kruskal

# ============================================================
# MDD Population-Specific PRS Analysis
# ============================================================

PROJECT = r"C:\PRS_Study"

RESULTS_DIR = os.path.join(
    PROJECT,
    "results",
    "MDD"
)

DATA_DIR = os.path.join(
    PROJECT,
    "data",
    "1000G"
)

# ============================================================
# Input files
# ============================================================

PRS_FILE = os.path.join(
    RESULTS_DIR,
    "MDD_genomewide_PRS_with_risk_groups.tsv"
)

POPULATION_FILE = os.path.join(
    DATA_DIR,
    "1000G_phase3_population_panel.txt"
)

# ============================================================
# Output directory
# ============================================================

POP_DIR = os.path.join(
    RESULTS_DIR,
    "population_analysis"
)

os.makedirs(
    POP_DIR,
    exist_ok=True
)

# ============================================================
# Output files
# ============================================================

LABELED_FILE = os.path.join(
    POP_DIR,
    "MDD_PRS_with_population_labels.tsv"
)

STATS_FILE = os.path.join(
    POP_DIR,
    "MDD_population_PRS_by_superpopulation.tsv"
)

KW_FILE = os.path.join(
    POP_DIR,
    "MDD_population_Kruskal_Wallis.tsv"
)

# ============================================================
# Load PRS
# ============================================================

print("\n" + "=" * 80)
print("MDD POPULATION-SPECIFIC PRS ANALYSIS")
print("=" * 80)

if not os.path.exists(PRS_FILE):
    raise FileNotFoundError(
        f"Missing PRS file:\n{PRS_FILE}"
    )

if not os.path.exists(POPULATION_FILE):
    raise FileNotFoundError(
        f"Missing population panel:\n{POPULATION_FILE}"
    )

prs = pd.read_csv(
    PRS_FILE,
    sep="\t"
)

print(
    f"\nPRS individuals: {len(prs):,}"
)

# ============================================================
# Validate PRS
# ============================================================

required_prs_columns = [
    "IID",
    "PRS"
]

for col in required_prs_columns:

    if col not in prs.columns:

        raise ValueError(
            f"Missing PRS column: {col}\n"
            f"Available columns: {prs.columns.tolist()}"
        )

prs["PRS"] = pd.to_numeric(
    prs["PRS"],
    errors="coerce"
)

if len(prs) != 2504:
    raise ValueError(
        f"Expected 2504 PRS individuals, "
        f"found {len(prs)}."
    )

if prs["IID"].nunique() != 2504:
    raise ValueError(
        "PRS file does not contain 2504 unique IIDs."
    )

if prs["PRS"].isna().sum() != 0:
    raise ValueError(
        "Missing PRS values detected."
    )

# ============================================================
# Load population panel
# ============================================================

print(
    "\nLoading 1000 Genomes population panel..."
)

panel = pd.read_csv(
    POPULATION_FILE,
    sep=r"\s+"
)

print(
    f"Population panel records: "
    f"{len(panel):,}"
)

print(
    f"Population panel columns: "
    f"{panel.columns.tolist()}"
)

# ============================================================
# Validate panel
# ============================================================

required_panel_columns = [
    "sample",
    "pop",
    "super_pop",
    "gender"
]

for col in required_panel_columns:

    if col not in panel.columns:

        raise ValueError(
            f"Missing population-panel column: {col}"
        )

panel = panel[
    required_panel_columns
].copy()

panel.rename(
    columns={
        "sample": "IID",
        "pop": "Population",
        "super_pop": "Superpopulation",
        "gender": "Gender"
    },
    inplace=True
)

# ============================================================
# Check panel IDs
# ============================================================

prs_ids = set(
    prs["IID"]
)

panel_ids = set(
    panel["IID"]
)

matched_ids = (
    prs_ids & panel_ids
)

missing_in_panel = (
    prs_ids - panel_ids
)

extra_panel_ids = (
    panel_ids - prs_ids
)

print(
    f"\nPRS individuals:        {len(prs_ids):,}"
)

print(
    f"Panel individuals:      {len(panel_ids):,}"
)

print(
    f"Matched individuals:    {len(matched_ids):,}"
)

print(
    f"Missing population labels: "
    f"{len(missing_in_panel):,}"
)

print(
    f"Extra panel individuals: "
    f"{len(extra_panel_ids):,}"
)

if len(missing_in_panel) != 0:

    raise ValueError(
        "Some PRS individuals are missing "
        "from the population panel."
    )

if len(extra_panel_ids) != 0:

    raise ValueError(
        "Population panel contains IDs not "
        "present in the PRS dataset."
    )

# ============================================================
# Merge population labels
# ============================================================

merged = prs.merge(
    panel,
    on="IID",
    how="inner",
    validate="one_to_one"
)

print(
    f"\nMerged individuals: "
    f"{len(merged):,}"
)

if len(merged) != 2504:

    raise ValueError(
        f"Expected 2504 merged individuals, "
        f"found {len(merged)}."
    )

if merged["Superpopulation"].isna().sum() != 0:

    raise ValueError(
        "Missing superpopulation labels detected."
    )

# ============================================================
# Save labeled dataset
# ============================================================

merged.to_csv(
    LABELED_FILE,
    sep="\t",
    index=False
)

# ============================================================
# Population counts
# ============================================================

print(
    "\nSuperpopulation counts:"
)

population_counts = (
    merged[
        "Superpopulation"
    ]
    .value_counts()
    .sort_index()
)

print(
    population_counts
)

# ============================================================
# Population-specific descriptive statistics
# ============================================================

population_order = [
    "AFR",
    "AMR",
    "EAS",
    "EUR",
    "SAS"
]

stats_rows = []

for population in population_order:

    values = merged.loc[
        merged["Superpopulation"]
        == population,
        "PRS"
    ]

    if len(values) == 0:

        print(
            f"WARNING: No samples found for "
            f"{population}"
        )

        continue

    n = len(values)

    mean = values.mean()

    sd = values.std()

    median = values.median()

    minimum = values.min()

    q1 = values.quantile(0.25)

    q3 = values.quantile(0.75)

    maximum = values.max()

    se = sd / np.sqrt(n)

    stats_rows.append(
        {
            "Superpopulation": population,
            "N": n,
            "Mean": mean,
            "SD": sd,
            "Median": median,
            "Min": minimum,
            "Q1": q1,
            "Q3": q3,
            "Max": maximum,
            "SE": se
        }
    )

stats = pd.DataFrame(
    stats_rows
)

# ============================================================
# Save population statistics
# ============================================================

stats.to_csv(
    STATS_FILE,
    sep="\t",
    index=False
)

# ============================================================
# Print population statistics
# ============================================================

print(
    "\n" + "=" * 80
)

print(
    "MDD PRS BY SUPERPOPULATION"
)

print(
    "=" * 80
)

for _, row in stats.iterrows():

    print(
        f"\n{row['Superpopulation']}"
    )

    print(
        f"  N:       {int(row['N'])}"
    )

    print(
        f"  Mean:    {row['Mean']:.8f}"
    )

    print(
        f"  SD:      {row['SD']:.8f}"
    )

    print(
        f"  Median:  {row['Median']:.8f}"
    )

    print(
        f"  Min:     {row['Min']:.8f}"
    )

    print(
        f"  Q1:      {row['Q1']:.8f}"
    )

    print(
        f"  Q3:      {row['Q3']:.8f}"
    )

    print(
        f"  Max:     {row['Max']:.8f}"
    )

    print(
        f"  SE:      {row['SE']:.8f}"
    )

# ============================================================
# Kruskal–Wallis test
# ============================================================

print(
    "\n" + "=" * 80
)

print(
    "KRUSKAL–WALLIS TEST"
)

print(
    "=" * 80
)

groups = []

group_names = []

for population in population_order:

    values = merged.loc[
        merged["Superpopulation"]
        == population,
        "PRS"
    ].dropna()

    if len(values) > 0:

        groups.append(
            values.values
        )

        group_names.append(
            population
        )

if len(groups) < 2:

    raise ValueError(
        "At least two populations are required "
        "for Kruskal–Wallis testing."
    )

H, p_value = kruskal(
    *groups
)

print(
    f"\nPopulations tested: "
    f"{', '.join(group_names)}"
)

print(
    f"Kruskal–Wallis H: "
    f"{H:.8f}"
)

print(
    f"P-value: "
    f"{p_value:.10e}"
)

if p_value < 0.001:

    interpretation = "p < 0.001"

else:

    interpretation = f"p = {p_value:.6g}"

print(
    f"Interpretation: "
    f"{interpretation}"
)

# ============================================================
# Save Kruskal–Wallis result
# ============================================================

kw_result = pd.DataFrame(
    [{
        "Test": "Kruskal-Wallis",
        "Populations": ";".join(group_names),
        "H_statistic": H,
        "P_value": p_value,
        "Interpretation": interpretation
    }]
)

kw_result.to_csv(
    KW_FILE,
    sep="\t",
    index=False
)

# ============================================================
# Final report
# ============================================================

print(
    "\n" + "=" * 80
)

print(
    "OUTPUT FILES"
)

print(
    "=" * 80
)

print(
    f"\nPopulation-labeled PRS:\n"
    f"{LABELED_FILE}"
)

print(
    f"\nPopulation statistics:\n"
    f"{STATS_FILE}"
)

print(
    f"\nKruskal–Wallis result:\n"
    f"{KW_FILE}"
)

print(
    "\n" + "=" * 80
)

print(
    "MDD POPULATION ANALYSIS COMPLETE"
)

print(
    "=" * 80
)
