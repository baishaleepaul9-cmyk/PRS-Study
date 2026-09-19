import os
import numpy as np
import pandas as pd

from scipy.stats import (
    kruskal,
    rankdata,
    spearmanr
)


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
    "standardization_sensitivity"
)

os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================
# DISEASES
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
print("SEVEN-DISEASE STANDARDIZATION SENSITIVITY ANALYSIS")
print("=" * 80)

if not os.path.exists(INPUT):
    raise FileNotFoundError(
        f"Input file not found:\n{INPUT}"
    )

df = pd.read_csv(INPUT, sep="\t")

print(f"\nRows: {len(df)}")


# ============================================================
# VALIDATION
# ============================================================

if len(df) != 2504:
    raise ValueError(
        f"Expected 2504 individuals, found {len(df)}."
    )

if "IID" not in df.columns:
    raise ValueError(
        "IID column not found."
    )

if "Superpopulation" not in df.columns:
    raise ValueError(
        "Superpopulation column not found."
    )

if df["IID"].duplicated().any():
    raise ValueError(
        "Duplicate IID values detected."
    )


# ============================================================
# RANK-BASED INVERSE NORMAL TRANSFORMATION
# ============================================================

def inverse_normal_transform(values):
    """
    Rank-based inverse normal transformation.

    Uses:
        q = (rank - 0.5) / N
        INT = Phi^-1(q)

    scipy does not provide a direct inverse-normal
    function in scipy.stats, so scipy.special.ndtri
    is used.
    """

    from scipy.special import ndtri

    values = np.asarray(values, dtype=float)

    ranks = rankdata(
        values,
        method="average"
    )

    n = len(values)

    probabilities = (
        ranks - 0.5
    ) / n

    return ndtri(probabilities)


# ============================================================
# STORAGE
# ============================================================

comparison_rows = []
population_rows = []


# ============================================================
# ANALYZE EACH DISEASE
# ============================================================

for disease_key, info in DISEASES.items():

    disease_name = info["name"]
    z_col = info["column"]

    print("\n" + "-" * 80)
    print(disease_name)
    print("-" * 80)

    original = df[z_col].astype(float)

    # --------------------------------------------------------
    # Alternative transformation
    # --------------------------------------------------------

    int_values = inverse_normal_transform(
        original.values
    )

    int_col = f"{disease_key}_INT"

    df[int_col] = int_values

    # --------------------------------------------------------
    # Correlation between original Z and INT
    # --------------------------------------------------------

    rho, rho_p = spearmanr(
        original,
        df[int_col]
    )

    # --------------------------------------------------------
    # Original Kruskal-Wallis
    # --------------------------------------------------------

    original_groups = []

    for population in POPULATIONS:

        x = df.loc[
            df["Superpopulation"] == population,
            z_col
        ].dropna()

        original_groups.append(
            x.values
        )

    H_original, p_original = kruskal(
        *original_groups
    )

    # --------------------------------------------------------
    # INT Kruskal-Wallis
    # --------------------------------------------------------

    int_groups = []

    for population in POPULATIONS:

        x = df.loc[
            df["Superpopulation"] == population,
            int_col
        ].dropna()

        int_groups.append(
            x.values
        )

    H_int, p_int = kruskal(
        *int_groups
    )

    # --------------------------------------------------------
    # Population means
    # --------------------------------------------------------

    original_means = []
    int_means = []

    for population in POPULATIONS:

        original_mean = df.loc[
            df["Superpopulation"] == population,
            z_col
        ].mean()

        int_mean = df.loc[
            df["Superpopulation"] == population,
            int_col
        ].mean()

        original_means.append(
            original_mean
        )

        int_means.append(
            int_mean
        )

        population_rows.append({
            "Disease": disease_name,
            "Population": population,
            "Original_Mean_Z": original_mean,
            "INT_Mean": int_mean
        })

    # --------------------------------------------------------
    # Population ordering
    # --------------------------------------------------------

    original_order = [
        POPULATIONS[i]
        for i in np.argsort(original_means)
    ]

    int_order = [
        POPULATIONS[i]
        for i in np.argsort(int_means)
    ]

    ordering_identical = (
        original_order == int_order
    )

    # --------------------------------------------------------
    # Store comparison
    # --------------------------------------------------------

    comparison_rows.append({
        "Disease": disease_name,
        "Spearman_Rho_Z_vs_INT": rho,
        "Spearman_P_value": rho_p,

        "Original_KW_H": H_original,
        "Original_KW_P": p_original,

        "INT_KW_H": H_int,
        "INT_KW_P": p_int,

        "Original_Low_to_High_Order":
            ">".join(original_order),

        "INT_Low_to_High_Order":
            ">".join(int_order),

        "Population_Order_Preserved":
            ordering_identical
    })

    print(
        f"Spearman rho: {rho:.6f}"
    )

    print(
        f"Original KW p: {p_original:.6e}"
    )

    print(
        f"INT KW p:      {p_int:.6e}"
    )

    print(
        f"Ordering preserved: "
        f"{ordering_identical}"
    )


# ============================================================
# SAVE COMPARISON RESULTS
# ============================================================

comparison_df = pd.DataFrame(
    comparison_rows
)

population_df = pd.DataFrame(
    population_rows
)

comparison_file = os.path.join(
    OUT_DIR,
    "standardization_sensitivity_summary.tsv"
)

population_file = os.path.join(
    OUT_DIR,
    "standardization_sensitivity_population_means.tsv"
)

comparison_df.to_csv(
    comparison_file,
    sep="\t",
    index=False,
    float_format="%.10f"
)

population_df.to_csv(
    population_file,
    sep="\t",
    index=False,
    float_format="%.10f"
)


# ============================================================
# SAVE TRANSFORMED DATA
# ============================================================

int_columns = [
    f"{key}_INT"
    for key in DISEASES
]

int_output = df[
    ["IID", "Superpopulation"] +
    int_columns
].copy()

int_file = os.path.join(
    OUT_DIR,
    "all_7_diseases_INT_PRS_with_population.tsv"
)

int_output.to_csv(
    int_file,
    sep="\t",
    index=False,
    float_format="%.10f"
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("SENSITIVITY ANALYSIS SUMMARY")
print("=" * 80)

print(
    comparison_df[
        [
            "Disease",
            "Spearman_Rho_Z_vs_INT",
            "Original_KW_P",
            "INT_KW_P",
            "Population_Order_Preserved"
        ]
    ].to_string(index=False)
)

print("\nOutputs:")
print(comparison_file)
print(population_file)
print(int_file)

print("\nSTATUS: STANDARDIZATION SENSITIVITY ANALYSIS COMPLETE")
print("=" * 80)
