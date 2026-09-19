import os
import itertools
import pandas as pd
import numpy as np
from scipy.stats import rankdata, norm

# ============================================================
# MDD Dunn's Pairwise Test + Holm Correction
# ============================================================

PROJECT = r"C:\PRS_Study"

POP_DIR = os.path.join(
    PROJECT,
    "results",
    "MDD",
    "population_analysis"
)

INPUT_FILE = os.path.join(
    POP_DIR,
    "MDD_PRS_with_population_labels.tsv"
)

PVALUES_FILE = os.path.join(
    POP_DIR,
    "MDD_population_Dunn_Holm_pvalues.tsv"
)

PAIRWISE_FILE = os.path.join(
    POP_DIR,
    "MDD_population_pairwise_Dunn_Holm.tsv"
)

SIGNIFICANT_FILE = os.path.join(
    POP_DIR,
    "MDD_population_significant_pairs.tsv"
)

# ============================================================
# Load data
# ============================================================

print("\n" + "=" * 80)
print("MDD DUNN PAIRWISE TEST + HOLM CORRECTION")
print("=" * 80)

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        f"Missing input file:\n{INPUT_FILE}"
    )

df = pd.read_csv(
    INPUT_FILE,
    sep="\t"
)

required = [
    "IID",
    "PRS",
    "Superpopulation"
]

for col in required:
    if col not in df.columns:
        raise ValueError(
            f"Missing column: {col}"
        )

df["PRS"] = pd.to_numeric(
    df["PRS"],
    errors="coerce"
)

df = df.dropna(
    subset=[
        "PRS",
        "Superpopulation"
    ]
).copy()

population_order = [
    "AFR",
    "AMR",
    "EAS",
    "EUR",
    "SAS"
]

# ============================================================
# Prepare groups
# ============================================================

groups = {}

for pop in population_order:

    values = df.loc[
        df["Superpopulation"] == pop,
        "PRS"
    ].dropna().values

    if len(values) == 0:
        raise ValueError(
            f"No observations found for {pop}"
        )

    groups[pop] = values

    print(
        f"{pop}: {len(values):,} samples"
    )

# ============================================================
# Dunn test
# ============================================================

# Combine all observations
all_values = np.concatenate(
    [groups[p] for p in population_order]
)

# Rank all observations together
ranks = rankdata(
    all_values,
    method="average"
)

# Tie correction
N = len(all_values)

_, counts = np.unique(
    all_values,
    return_counts=True
)

tie_sum = np.sum(
    counts**3 - counts
)

tie_correction = (
    tie_sum /
    (12 * (N - 1))
)

# Mean rank for each population
mean_ranks = {}

start = 0

for pop in population_order:

    n = len(groups[pop])

    mean_ranks[pop] = np.mean(
        ranks[start:start + n]
    )

    start += n

# ============================================================
# Calculate raw pairwise p-values
# ============================================================

pairwise_rows = []

for pop1, pop2 in itertools.combinations(
    population_order,
    2
):

    n1 = len(groups[pop1])
    n2 = len(groups[pop2])

    rank_difference = (
        mean_ranks[pop1]
        - mean_ranks[pop2]
    )

    denominator = np.sqrt(
        (
            N * (N + 1) / 12
            - tie_correction
        )
        *
        (
            1 / n1
            + 1 / n2
        )
    )

    z = (
        rank_difference /
        denominator
    )

    raw_p = 2 * norm.sf(
        abs(z)
    )

    pairwise_rows.append(
        {
            "Population_1": pop1,
            "Population_2": pop2,
            "N1": n1,
            "N2": n2,
            "Mean_Rank_1": mean_ranks[pop1],
            "Mean_Rank_2": mean_ranks[pop2],
            "Rank_Difference": rank_difference,
            "Z": z,
            "Raw_P_value": raw_p
        }
    )

pairwise = pd.DataFrame(
    pairwise_rows
)

# ============================================================
# Holm correction
# ============================================================

m = len(pairwise)

sorted_indices = (
    pairwise["Raw_P_value"]
    .argsort()
    .to_numpy()
)

holm_adjusted = np.empty(m)

for rank, idx in enumerate(
    sorted_indices
):

    holm_adjusted[idx] = min(
        1.0,
        (
            m - rank
        )
        *
        pairwise.loc[
            idx,
            "Raw_P_value"
        ]
    )

# Enforce monotonicity
for i in range(1, m):

    prev_idx = sorted_indices[i - 1]
    curr_idx = sorted_indices[i]

    holm_adjusted[curr_idx] = max(
        holm_adjusted[curr_idx],
        holm_adjusted[prev_idx]
    )

pairwise[
    "Holm_Adjusted_P"
] = holm_adjusted

# ============================================================
# Significance
# ============================================================

pairwise["Significant"] = (
    pairwise["Holm_Adjusted_P"] < 0.05
)

pairwise["Significance"] = np.where(
    pairwise["Holm_Adjusted_P"] < 0.001,
    "p < 0.001",
    np.where(
        pairwise["Holm_Adjusted_P"] < 0.01,
        "p < 0.01",
        np.where(
            pairwise["Holm_Adjusted_P"] < 0.05,
            "p < 0.05",
            "NS"
        )
    )
)

# ============================================================
# Save full pairwise table
# ============================================================

pairwise.to_csv(
    PAIRWISE_FILE,
    sep="\t",
    index=False
)

# ============================================================
# Create p-value matrix
# ============================================================

pvalue_matrix = pd.DataFrame(
    np.nan,
    index=population_order,
    columns=population_order
)

for pop in population_order:
    pvalue_matrix.loc[
        pop,
        pop
    ] = 1.0

for _, row in pairwise.iterrows():

    p1 = row["Population_1"]
    p2 = row["Population_2"]

    p = row["Holm_Adjusted_P"]

    pvalue_matrix.loc[p1, p2] = p
    pvalue_matrix.loc[p2, p1] = p

pvalue_matrix.to_csv(
    PVALUES_FILE,
    sep="\t"
)

# ============================================================
# Significant pairs only
# ============================================================

significant = pairwise[
    pairwise["Significant"]
].copy()

significant.to_csv(
    SIGNIFICANT_FILE,
    sep="\t",
    index=False
)

# ============================================================
# Print results
# ============================================================

print("\n" + "=" * 80)
print("DUNN + HOLM RESULTS")
print("=" * 80)

for _, row in pairwise.iterrows():

    print(
        f"\n{row['Population_1']} vs "
        f"{row['Population_2']}"
    )

    print(
        f"  Z:              "
        f"{row['Z']:.6f}"
    )

    print(
        f"  Raw p:          "
        f"{row['Raw_P_value']:.6e}"
    )

    print(
        f"  Holm-adjusted p:"
        f" {row['Holm_Adjusted_P']:.6e}"
    )

    print(
        f"  Result:         "
        f"{row['Significance']}"
    )

# ============================================================
# Summary
# ============================================================

significant_count = int(
    pairwise["Significant"].sum()
)

total_pairs = len(pairwise)

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print(
    f"Total pairwise comparisons: "
    f"{total_pairs}"
)

print(
    f"Significant comparisons:     "
    f"{significant_count}"
)

print(
    f"Non-significant comparisons: "
    f"{total_pairs - significant_count}"
)

# ============================================================
# Output files
# ============================================================

print("\n" + "=" * 80)
print("OUTPUT FILES")
print("=" * 80)

print(
    f"\nP-value matrix:\n"
    f"{PVALUES_FILE}"
)

print(
    f"\nFull pairwise results:\n"
    f"{PAIRWISE_FILE}"
)

print(
    f"\nSignificant pairs:\n"
    f"{SIGNIFICANT_FILE}"
)

print("\n" + "=" * 80)
print("MDD DUNN + HOLM ANALYSIS COMPLETE")
print("=" * 80)
