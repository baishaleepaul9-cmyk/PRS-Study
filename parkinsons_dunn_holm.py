import os
import pandas as pd
import numpy as np
from scipy.stats import rankdata
from scipy.special import ndtr
from statsmodels.stats.multitest import multipletests

# ============================================================
# PARKINSON'S DISEASE — DUNN POST-HOC TEST + HOLM CORRECTION
# ============================================================

PROJECT = r"C:\PRS_Study"

POP_DIR = os.path.join(
    PROJECT,
    "results",
    "Parkinsons",
    "population_analysis"
)

INPUT_FILE = os.path.join(
    POP_DIR,
    "Parkinsons_PRS_with_population_labels.tsv"
)

PVALUES_FILE = os.path.join(
    POP_DIR,
    "Parkinsons_population_Dunn_Holm_pvalues.tsv"
)

PAIRWISE_FILE = os.path.join(
    POP_DIR,
    "Parkinsons_population_pairwise_Dunn_Holm.tsv"
)

SIGNIFICANT_FILE = os.path.join(
    POP_DIR,
    "Parkinsons_population_significant_pairs.tsv"
)

# ============================================================
# LOAD DATA
# ============================================================

print("=" * 80)
print("PARKINSON'S DISEASE — DUNN POST-HOC TEST + HOLM CORRECTION")
print("=" * 80)

if not os.path.exists(INPUT_FILE):

    raise FileNotFoundError(
        f"Input file not found:\n{INPUT_FILE}"
    )

df = pd.read_csv(
    INPUT_FILE,
    sep="\t"
)

print(
    f"\nIndividuals loaded: {len(df):,}"
)

# ============================================================
# BASIC CHECKS
# ============================================================

required_columns = [
    "IID",
    "PRS",
    "super_pop"
]

for column in required_columns:

    if column not in df.columns:

        raise ValueError(
            f"Required column not found: {column}"
        )

df["PRS"] = pd.to_numeric(
    df["PRS"],
    errors="coerce"
)

if df["PRS"].isna().any():

    raise ValueError(
        "Missing PRS values detected."
    )

if len(df) != 2504:

    raise ValueError(
        f"Expected 2504 individuals, found {len(df)}."
    )

if df["IID"].nunique() != 2504:

    raise ValueError(
        "Duplicate individual IDs detected."
    )

# ============================================================
# POPULATION ORDER
# ============================================================

populations = [
    "AFR",
    "AMR",
    "EAS",
    "EUR",
    "SAS"
]

print("\n" + "-" * 80)
print("POPULATION SAMPLE COUNTS")
print("-" * 80)

groups = []

for population in populations:

    values = df.loc[
        df["super_pop"] == population,
        "PRS"
    ].astype(float).values

    if len(values) == 0:

        raise ValueError(
            f"No individuals found for {population}."
        )

    groups.append(values)

    print(
        f"{population}: N = {len(values)}"
    )

# ============================================================
# COMBINE OBSERVATIONS
# ============================================================

all_values = np.concatenate(
    groups
)

N = len(all_values)
k = len(groups)

# ============================================================
# GLOBAL RANKS
# ============================================================

ranks = rankdata(
    all_values,
    method="average"
)

# ============================================================
# TIE CORRECTION
# ============================================================

_, counts = np.unique(
    all_values,
    return_counts=True
)

tie_sum = np.sum(
    counts[counts > 1] ** 3
    - counts[counts > 1]
)

if N > 1:

    tie_correction = 1 - (
        tie_sum /
        (N**3 - N)
    )

else:

    tie_correction = 1.0

# ============================================================
# MEAN RANKS
# ============================================================

rank_groups = {}

offset = 0

for population, values in zip(
    populations,
    groups
):

    n = len(values)

    population_ranks = ranks[
        offset:
        offset + n
    ]

    rank_groups[population] = {
        "n": n,
        "mean_rank": population_ranks.mean()
    }

    offset += n

# ============================================================
# DUNN PAIRWISE TEST
# ============================================================

print("\n" + "-" * 80)
print("PAIRWISE DUNN TEST")
print("-" * 80)

pair_rows = []
raw_pvalues = []

for i in range(k):

    for j in range(i + 1, k):

        pop1 = populations[i]
        pop2 = populations[j]

        n1 = rank_groups[
            pop1
        ]["n"]

        n2 = rank_groups[
            pop2
        ]["n"]

        r1 = rank_groups[
            pop1
        ]["mean_rank"]

        r2 = rank_groups[
            pop2
        ]["mean_rank"]

        # ----------------------------------------------------
        # Dunn variance
        # ----------------------------------------------------

        variance = (
            N * (N + 1) / 12
        ) * (
            1 / n1 + 1 / n2
        ) * tie_correction

        # ----------------------------------------------------
        # Z statistic
        # ----------------------------------------------------

        z = (
            r1 - r2
        ) / np.sqrt(
            variance
        )

        # ----------------------------------------------------
        # TWO-SIDED P-VALUE
        # ----------------------------------------------------

        p = 2 * ndtr(
            -abs(z)
        )

        raw_pvalues.append(
            p
        )

        pair_rows.append(
            {
                "Population_1": pop1,
                "Population_2": pop2,
                "N_1": n1,
                "N_2": n2,
                "Mean_Rank_1": r1,
                "Mean_Rank_2": r2,
                "Z": z,
                "Raw_P": p
            }
        )

# ============================================================
# HOLM CORRECTION
# ============================================================

reject, adjusted_pvalues, _, _ = multipletests(
    raw_pvalues,
    alpha=0.05,
    method="holm"
)

for row, adjusted_p, significant in zip(
    pair_rows,
    adjusted_pvalues,
    reject
):

    row["Holm_Adjusted_P"] = adjusted_p

    row["Significant"] = (
        "YES"
        if significant
        else
        "NO"
    )

results = pd.DataFrame(
    pair_rows
)

# ============================================================
# PRINT PAIRWISE RESULTS
# ============================================================

print()

for _, row in results.iterrows():

    print(
        f"{row['Population_1']} vs "
        f"{row['Population_2']}: "
        f"raw p = {row['Raw_P']:.6e}, "
        f"Holm p = {row['Holm_Adjusted_P']:.6e}, "
        f"{row['Significant']}"
    )

# ============================================================
# HOLM-ADJUSTED P-VALUE MATRIX
# ============================================================

pvalue_matrix = pd.DataFrame(
    1.0,
    index=populations,
    columns=populations
)

for _, row in results.iterrows():

    pop1 = row[
        "Population_1"
    ]

    pop2 = row[
        "Population_2"
    ]

    adjusted_p = row[
        "Holm_Adjusted_P"
    ]

    pvalue_matrix.loc[
        pop1,
        pop2
    ] = adjusted_p

    pvalue_matrix.loc[
        pop2,
        pop1
    ] = adjusted_p

# ------------------------------------------------------------
# IMPORTANT:
# Do NOT use np.fill_diagonal(pvalue_matrix.values,...)
# because newer NumPy/pandas combinations can expose the
# underlying array as read-only.
# ------------------------------------------------------------

for population in populations:

    pvalue_matrix.loc[
        population,
        population
    ] = 1.0

# ============================================================
# SIGNIFICANT PAIRS
# ============================================================

significant = results[
    results["Significant"] == "YES"
].copy()

# ============================================================
# SAVE RESULTS
# ============================================================

pvalue_matrix.to_csv(
    PVALUES_FILE,
    sep="\t"
)

results.to_csv(
    PAIRWISE_FILE,
    sep="\t",
    index=False
)

significant.to_csv(
    SIGNIFICANT_FILE,
    sep="\t",
    index=False
)

# ============================================================
# SUMMARY
# ============================================================

print("\n" + "-" * 80)
print("SUMMARY")
print("-" * 80)

print(
    f"Total pairwise comparisons: "
    f"{len(results)}"
)

print(
    f"Significant after Holm correction: "
    f"{len(significant)}"
)

print(
    f"Non-significant: "
    f"{len(results) - len(significant)}"
)

print(
    "\nHolm-adjusted p-value matrix:"
)

print(
    pvalue_matrix.to_string()
)

# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 80)
print("PARKINSON'S DUNN + HOLM ANALYSIS COMPLETE")
print("=" * 80)

print(
    f"\nP-value matrix:\n{PVALUES_FILE}"
)

print(
    f"\nPairwise results:\n{PAIRWISE_FILE}"
)

print(
    f"\nSignificant pairs:\n{SIGNIFICANT_FILE}"
)

print(
    "\nSTATUS: SUCCESS"
)
