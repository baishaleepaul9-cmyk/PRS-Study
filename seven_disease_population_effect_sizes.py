import os
import numpy as np
import pandas as pd
from scipy.stats import kruskal


# ============================================================
# PATHS
# ============================================================

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
    "strengthening_analysis",
    "effect_sizes"
)

os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================
# DISEASES AND POPULATIONS
# ============================================================

DISEASES = {
    "Alzheimers": {
        "name": "Alzheimer's disease",
        "column": "Alzheimers_Z"
    },
    "Schizophrenia": {
        "name": "Schizophrenia",
        "column": "Schizophrenia_Z"
    },
    "Multiple_Sclerosis": {
        "name": "Multiple sclerosis",
        "column": "Multiple_Sclerosis_Z"
    },
    "MDD": {
        "name": "Major depressive disorder",
        "column": "MDD_Z"
    },
    "Parkinsons": {
        "name": "Parkinson's disease",
        "column": "Parkinsons_Z"
    },
    "CAD": {
        "name": "Coronary artery disease",
        "column": "CAD_Z"
    },
    "T2D": {
        "name": "Type 2 diabetes",
        "column": "T2D_Z"
    }
}

POPULATIONS = [
    "AFR",
    "AMR",
    "EAS",
    "EUR",
    "SAS"
]


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 80)
print("SEVEN-DISEASE POPULATION EFFECT-SIZE ANALYSIS")
print("=" * 80)

if not os.path.exists(INPUT):
    raise FileNotFoundError(
        f"Input file not found:\n{INPUT}"
    )

df = pd.read_csv(INPUT, sep="\t")

print(f"\nInput: {INPUT}")
print(f"Rows: {len(df)}")


# ============================================================
# BASIC VALIDATION
# ============================================================

if len(df) != 2504:
    raise ValueError(
        f"Expected 2504 individuals, found {len(df)}."
    )

if "Superpopulation" not in df.columns:
    raise ValueError(
        "Superpopulation column not found."
    )

observed = set(
    df["Superpopulation"].astype(str)
)

if observed != set(POPULATIONS):
    raise ValueError(
        f"Unexpected populations: {observed}"
    )


for disease, info in DISEASES.items():

    if info["column"] not in df.columns:
        raise ValueError(
            f"Missing PRS column: {info['column']}"
        )

    if df[info["column"]].isna().any():
        raise ValueError(
            f"Missing values detected for {disease}."
        )


# ============================================================
# EFFECT-SIZE FUNCTION
# ============================================================

def epsilon_squared(H, k, N):
    """
    Kruskal-Wallis epsilon-squared effect size.

    epsilon^2 = (H - k + 1) / (N - k)

    k = number of groups
    N = total sample size
    """

    denominator = N - k

    if denominator <= 0:
        return np.nan

    value = (H - k + 1) / denominator

    # Numerical protection
    return max(0.0, value)


# ============================================================
# ANALYSIS
# ============================================================

results = []

population_means = []

for disease_key, info in DISEASES.items():

    disease_name = info["name"]
    z_col = info["column"]

    print("\n" + "-" * 80)
    print(disease_name)
    print("-" * 80)

    groups = []

    means = {}

    for population in POPULATIONS:

        values = df.loc[
            df["Superpopulation"] == population,
            z_col
        ].dropna()

        groups.append(values.values)

        means[population] = values.mean()

    # --------------------------------------------------------
    # Kruskal-Wallis
    # --------------------------------------------------------

    H, p = kruskal(*groups)

    # --------------------------------------------------------
    # Effect size
    # --------------------------------------------------------

    N = sum(len(x) for x in groups)
    k = len(groups)

    epsilon_sq = epsilon_squared(
        H,
        k,
        N
    )

    mean_values = list(means.values())

    mean_range = (
        max(mean_values) -
        min(mean_values)
    )

    results.append({
        "Disease": disease_name,
        "N": N,
        "Groups": k,
        "Kruskal_Wallis_H": H,
        "P_value": p,
        "Epsilon_squared": epsilon_sq,
        "Population_Mean_Range": mean_range,
        "Minimum_Population_Mean": min(mean_values),
        "Maximum_Population_Mean": max(mean_values)
    })

    for population in POPULATIONS:

        population_means.append({
            "Disease": disease_name,
            "Population": population,
            "Mean_Z": means[population]
        })

    print(f"H statistic:       {H:.6f}")
    print(f"P-value:           {p:.6e}")
    print(f"Epsilon-squared:   {epsilon_sq:.6f}")
    print(f"Mean range:        {mean_range:.6f}")


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(results)

means_df = pd.DataFrame(
    population_means
)

effect_file = os.path.join(
    OUT_DIR,
    "seven_disease_population_effect_sizes.tsv"
)

means_file = os.path.join(
    OUT_DIR,
    "seven_disease_population_means_for_effect_size.tsv"
)

results_df.to_csv(
    effect_file,
    sep="\t",
    index=False,
    float_format="%.10f"
)

means_df.to_csv(
    means_file,
    sep="\t",
    index=False,
    float_format="%.10f"
)


# ============================================================
# PRINT FINAL TABLE
# ============================================================

print("\n" + "=" * 80)
print("FINAL EFFECT-SIZE RESULTS")
print("=" * 80)

print(
    results_df[
        [
            "Disease",
            "Kruskal_Wallis_H",
            "P_value",
            "Epsilon_squared",
            "Population_Mean_Range"
        ]
    ].to_string(index=False)
)

print("\nOutput:")
print(effect_file)
print(means_file)

print("\nSTATUS: EFFECT-SIZE ANALYSIS COMPLETE")
print("=" * 80)
