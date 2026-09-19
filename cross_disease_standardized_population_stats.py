import os
import numpy as np
import pandas as pd
from scipy.stats import kruskal
from scipy.stats import rankdata
from statsmodels.stats.multitest import multipletests

ROOT = r"C:\PRS_Study"

INPUT = os.path.join(
    ROOT,
    "results",
    "combined_analysis",
    "all_7_diseases_standardized_PRS_with_population.tsv"
)

OUT_DIR = os.path.join(
    ROOT,
    "results",
    "combined_analysis",
    "standardized_population_analysis"
)

os.makedirs(OUT_DIR, exist_ok=True)

POPULATIONS = ["AFR", "AMR", "EAS", "EUR", "SAS"]

DISEASES = [
    "Alzheimers",
    "Schizophrenia",
    "Multiple_Sclerosis",
    "MDD",
    "Parkinsons",
    "CAD",
    "T2D"
]

Z_COLUMNS = {
    "Alzheimers": "Alzheimers_Z",
    "Schizophrenia": "Schizophrenia_Z",
    "Multiple_Sclerosis": "Multiple_Sclerosis_Z",
    "MDD": "MDD_Z",
    "Parkinsons": "Parkinsons_Z",
    "CAD": "CAD_Z",
    "T2D": "T2D_Z"
}

print("=" * 80)
print("CROSS-DISEASE STANDARDIZED PRS POPULATION ANALYSIS")
print("=" * 80)

# ============================================================
# LOAD DATA
# ============================================================

if not os.path.exists(INPUT):
    raise FileNotFoundError(
        f"Input file not found:\n{INPUT}"
    )

df = pd.read_csv(INPUT, sep="\t")

print(f"\nInput: {INPUT}")
print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")

# ============================================================
# BASIC VERIFICATION
# ============================================================

if len(df) != 2504:
    raise ValueError(
        f"Expected 2504 individuals, found {len(df)}."
    )

if "Superpopulation" not in df.columns:
    raise ValueError(
        "Superpopulation column not found."
    )

observed_populations = set(
    df["Superpopulation"].astype(str)
)

if observed_populations != set(POPULATIONS):
    raise ValueError(
        f"Unexpected populations: {observed_populations}"
    )

for disease, col in Z_COLUMNS.items():

    if col not in df.columns:
        raise ValueError(
            f"Missing standardized PRS column: {col}"
        )

    if df[col].isna().any():
        raise ValueError(
            f"Missing standardized PRS values for {disease}."
        )

print("Basic verification: PASSED")

# ============================================================
# POPULATION-WISE DESCRIPTIVE STATISTICS
# ============================================================

descriptive_rows = []

for disease in DISEASES:

    z_col = Z_COLUMNS[disease]

    for population in POPULATIONS:

        x = df.loc[
            df["Superpopulation"] == population,
            z_col
        ].dropna()

        n = len(x)
        mean = x.mean()
        sd = x.std(ddof=1)
        median = x.median()
        minimum = x.min()
        maximum = x.max()

        se = sd / np.sqrt(n)

        ci_low = mean - 1.96 * se
        ci_high = mean + 1.96 * se

        descriptive_rows.append({
            "Disease": disease,
            "Population": population,
            "N": n,
            "Mean_Z": mean,
            "SD_Z": sd,
            "Median_Z": median,
            "Min_Z": minimum,
            "Max_Z": maximum,
            "SE_Z": se,
            "CI95_Lower": ci_low,
            "CI95_Upper": ci_high
        })

descriptive = pd.DataFrame(descriptive_rows)

# ============================================================
# SAVE DESCRIPTIVE STATISTICS
# ============================================================

descriptive_output = os.path.join(
    OUT_DIR,
    "all_7_diseases_standardized_population_statistics.tsv"
)

descriptive.to_csv(
    descriptive_output,
    sep="\t",
    index=False,
    float_format="%.10f"
)

# ============================================================
# KRUSKAL-WALLIS + DUNN-HOLM
# ============================================================

kw_rows = []
dunn_rows = []

for disease in DISEASES:

    z_col = Z_COLUMNS[disease]

    groups = []

    for population in POPULATIONS:

        x = df.loc[
            df["Superpopulation"] == population,
            z_col
        ].dropna()

        groups.append(x)

    # --------------------------------------------------------
    # Kruskal-Wallis
    # --------------------------------------------------------

    H, p = kruskal(*groups)

    kw_rows.append({
        "Disease": disease,
        "Kruskal_Wallis_H": H,
        "P_value": p,
        "Significant_P<0.05": p < 0.05
    })

    # --------------------------------------------------------
    # Pairwise Dunn-style rank comparisons
    # --------------------------------------------------------
    #
    # This implementation uses the pooled ranks and applies
    # Holm correction to all 10 population-pair comparisons.
    # --------------------------------------------------------

    values = []
    labels = []

    for population, x in zip(POPULATIONS, groups):

        values.extend(x.tolist())
        labels.extend([population] * len(x))

    values = np.asarray(values)
    labels = np.asarray(labels)

    ranks = rankdata(values)

    rank_df = pd.DataFrame({
        "Population": labels,
        "Rank": ranks
    })

    mean_ranks = (
        rank_df
        .groupby("Population")["Rank"]
        .mean()
    )

    N = len(values)

    # Tie correction
    _, tie_counts = np.unique(values, return_counts=True)

    tie_sum = np.sum(
        tie_counts[tie_counts > 1] ** 3
        - tie_counts[tie_counts > 1]
    )

    tie_correction = 1 - (
        tie_sum / (N**3 - N)
    )

    raw_pairs = []
    raw_pvalues = []

    for i in range(len(POPULATIONS)):

        for j in range(i + 1, len(POPULATIONS)):

            pop1 = POPULATIONS[i]
            pop2 = POPULATIONS[j]

            n1 = len(groups[i])
            n2 = len(groups[j])

            rank_difference = abs(
                mean_ranks[pop1] -
                mean_ranks[pop2]
            )

            denominator = np.sqrt(
                (
                    N * (N + 1) / 12
                )
                * tie_correction
                * (
                    (1 / n1) +
                    (1 / n2)
                )
            )

            z = rank_difference / denominator

            # Two-sided normal approximation
            from scipy.stats import norm

            p_raw = 2 * norm.sf(abs(z))

            raw_pairs.append(
                (pop1, pop2, z)
            )

            raw_pvalues.append(p_raw)

    # Holm correction
    reject, p_adjusted, _, _ = multipletests(
        raw_pvalues,
        method="holm"
    )

    for pair, p_raw, p_adj, sig in zip(
        raw_pairs,
        raw_pvalues,
        p_adjusted,
        reject
    ):

        pop1, pop2, z = pair

        dunn_rows.append({
            "Disease": disease,
            "Population_1": pop1,
            "Population_2": pop2,
            "Z": z,
            "P_raw": p_raw,
            "P_Holm": p_adj,
            "Significant_P<0.05": bool(sig)
        })

# ============================================================
# SAVE KRUSKAL-WALLIS
# ============================================================

kw = pd.DataFrame(kw_rows)

kw_output = os.path.join(
    OUT_DIR,
    "all_7_diseases_standardized_Kruskal_Wallis.tsv"
)

kw.to_csv(
    kw_output,
    sep="\t",
    index=False,
    float_format="%.10f"
)

# ============================================================
# SAVE DUNN-HOLM
# ============================================================

dunn = pd.DataFrame(dunn_rows)

dunn_output = os.path.join(
    OUT_DIR,
    "all_7_diseases_standardized_Dunn_Holm.tsv"
)

dunn.to_csv(
    dunn_output,
    sep="\t",
    index=False,
    float_format="%.10f"
)

# ============================================================
# CREATE P-VALUE MATRICES
# ============================================================

for disease in DISEASES:

    subset = dunn[
        dunn["Disease"] == disease
    ]

    matrix = pd.DataFrame(
        np.ones(
            (len(POPULATIONS), len(POPULATIONS))
        ),
        index=POPULATIONS,
        columns=POPULATIONS
    )

    for _, row in subset.iterrows():

        p = row["P_Holm"]

        matrix.loc[
            row["Population_1"],
            row["Population_2"]
        ] = p

        matrix.loc[
            row["Population_2"],
            row["Population_1"]
        ] = p

    matrix_output = os.path.join(
        OUT_DIR,
        f"{disease}_standardized_Dunn_Holm_pvalues.tsv"
    )

    matrix.to_csv(
        matrix_output,
        sep="\t",
        float_format="%.10f"
    )

# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n")
print("=" * 80)
print("POPULATION-WISE STANDARDIZED PRS")
print("=" * 80)

print(
    descriptive[
        [
            "Disease",
            "Population",
            "N",
            "Mean_Z",
            "SD_Z",
            "CI95_Lower",
            "CI95_Upper"
        ]
    ].to_string(index=False)
)

print("\n")
print("=" * 80)
print("KRUSKAL-WALLIS RESULTS")
print("=" * 80)

print(kw.to_string(index=False))

print("\n")
print("=" * 80)
print("DUNN-HOLM SIGNIFICANT PAIRS")
print("=" * 80)

significant = dunn[
    dunn["Significant_P<0.05"] == True
]

if len(significant) == 0:

    print("No significant pairwise comparisons.")

else:

    print(
        significant.to_string(index=False)
    )

# ============================================================
# FINAL VERIFICATION
# ============================================================

print("\n")
print("=" * 80)
print("FINAL VERIFICATION")
print("=" * 80)

print(
    f"Population-statistic rows: {len(descriptive)} "
    f"(expected 35)"
)

print(
    f"Kruskal-Wallis tests: {len(kw)} "
    f"(expected 7)"
)

print(
    f"Dunn-Holm comparisons: {len(dunn)} "
    f"(expected 70)"
)

if len(descriptive) != 35:
    raise ValueError("Population statistics row count is incorrect.")

if len(kw) != 7:
    raise ValueError("Kruskal-Wallis result count is incorrect.")

if len(dunn) != 70:
    raise ValueError("Dunn-Holm comparison count is incorrect.")

print("\nSTATUS: CROSS-DISEASE STANDARDIZED POPULATION ANALYSIS COMPLETE")

print("\nOutputs:")
print(descriptive_output)
print(kw_output)
print(dunn_output)

print("=" * 80)
