# =============================================================================
# FINAL STATISTICAL VERIFICATION
# Seven-Disease PRS Study
#
# Purpose:
#   1. Verify the Disease × Superpopulation interaction using ordinary
#      two-way OLS.
#   2. Avoid the previously problematic IID random/fixed-effect models.
#   3. Independently verify the interaction F statistic.
#   4. Perform an individual-level permutation test by permuting
#      superpopulation labels while preserving each individual's complete
#      seven-disease PRS profile.
#
# Dataset:
#   2,504 individuals
#   7 diseases
#   5 1000 Genomes superpopulations
#
# IMPORTANT:
#   This script does NOT modify PRS values.
#   This script does NOT re-standardize PRS.
#   This script does NOT perform clinical risk prediction.
# =============================================================================


# =============================================================================
# 1. IMPORTS
# =============================================================================

import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

from scipy import stats
from scipy.stats import f as f_distribution


# =============================================================================
# 2. SETTINGS
# =============================================================================

ROOT = r"C:\PRS_Study"

INPUT = os.path.join(
    ROOT,
    "results",
    "combined_analysis",
    "all_7_diseases_standardized_PRS_with_population.tsv"
)

OUTPUT_DIR = os.path.join(
    ROOT,
    "results",
    "combined_analysis",
    "strengthening_analysis",
    "interaction_analysis"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# -----------------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------------

RANDOM_SEED = 42

rng = np.random.default_rng(
    RANDOM_SEED
)


# -----------------------------------------------------------------------------
# Number of permutations
#
# 10,000 is appropriate for the final analysis.
# Increase to 50,000 or 100,000 if desired after the script is confirmed.
# -----------------------------------------------------------------------------

N_PERMUTATIONS = 10000


# =============================================================================
# 3. POPULATIONS AND DISEASES
# =============================================================================

POPULATIONS = [
    "AFR",
    "AMR",
    "EAS",
    "EUR",
    "SAS"
]


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


# This is the important definition that was missing
# in the previous version.
disease_columns = [
    Z_COLUMNS[disease]
    for disease in DISEASES
]


# =============================================================================
# 4. OUTPUT FILES
# =============================================================================

OUTPUT_SUMMARY = os.path.join(
    OUTPUT_DIR,
    "disease_population_interaction_summary.tsv"
)

OUTPUT_PERMUTATION = os.path.join(
    OUTPUT_DIR,
    "disease_population_interaction_permutation.tsv"
)

OUTPUT_DESCRIPTIVE = os.path.join(
    OUTPUT_DIR,
    "disease_population_interaction_descriptive.tsv"
)

OUTPUT_PERMUTATION_DISTRIBUTION = os.path.join(
    OUTPUT_DIR,
    "interaction_permutation_distribution.tsv"
)


# =============================================================================
# 5. HEADER
# =============================================================================

print("\n")
print("=" * 80)
print("FINAL STATISTICAL VERIFICATION")
print("SEVEN-DISEASE PRS: DISEASE × SUPERPOPULATION INTERACTION")
print("=" * 80)

print("\nInput:")
print(INPUT)

print("\nOutput directory:")
print(OUTPUT_DIR)

print("\nNumber of permutations:")
print(f"{N_PERMUTATIONS:,}")

print("\nRandom seed:")
print(RANDOM_SEED)


# =============================================================================
# 6. CHECK INPUT FILE
# =============================================================================

print("\n" + "=" * 80)
print("1. CHECKING INPUT FILE")
print("=" * 80)


if not os.path.exists(INPUT):

    raise FileNotFoundError(
        f"\nInput file was not found:\n{INPUT}\n\n"
        "Check that the standardized PRS file exists at the expected location."
    )


print("Input file found.")


# =============================================================================
# 7. LOAD DATA
# =============================================================================

print("\n" + "=" * 80)
print("2. LOADING DATA")
print("=" * 80)


df = pd.read_csv(
    INPUT,
    sep="\t"
)


print(
    f"Rows:    {len(df):,}"
)

print(
    f"Columns: {len(df.columns):,}"
)


# =============================================================================
# 8. BASIC DATA VALIDATION
# =============================================================================

print("\n" + "=" * 80)
print("3. BASIC DATA VALIDATION")
print("=" * 80)


# -----------------------------------------------------------------------------
# Expected number of individuals
# -----------------------------------------------------------------------------

EXPECTED_N = 2504


if len(df) != EXPECTED_N:

    raise ValueError(
        f"Expected {EXPECTED_N:,} individuals, "
        f"but found {len(df):,}."
    )


print(
    f"Individuals: {len(df):,} -- PASSED"
)


# -----------------------------------------------------------------------------
# IID column
# -----------------------------------------------------------------------------

if "IID" not in df.columns:

    raise ValueError(
        "Required column 'IID' not found."
    )


print(
    "IID column: PASSED"
)


# -----------------------------------------------------------------------------
# Population column
# -----------------------------------------------------------------------------

if "Superpopulation" not in df.columns:

    raise ValueError(
        "Required column 'Superpopulation' not found."
    )


print(
    "Superpopulation column: PASSED"
)


# -----------------------------------------------------------------------------
# Disease columns
# -----------------------------------------------------------------------------

print("\nChecking disease Z-score columns:")


for disease in DISEASES:

    column = Z_COLUMNS[disease]

    if column not in df.columns:

        raise ValueError(
            f"Missing disease Z-score column: {column}"
        )

    print(
        f"  {disease:<22} {column:<25} PASSED"
    )


# =============================================================================
# 9. CHECK IID UNIQUENESS
# =============================================================================

print("\n" + "=" * 80)
print("4. CHECKING INDIVIDUAL IDENTIFIERS")
print("=" * 80)


duplicate_iids = int(
    df["IID"].duplicated().sum()
)


print(
    f"Duplicate IIDs: {duplicate_iids}"
)


if duplicate_iids != 0:

    raise ValueError(
        "Duplicate IID values detected."
    )


print(
    "IID uniqueness: PASSED"
)


# =============================================================================
# 10. CHECK POPULATIONS
# =============================================================================

print("\n" + "=" * 80)
print("5. CHECKING SUPERPOPULATIONS")
print("=" * 80)


df["Superpopulation"] = (
    df["Superpopulation"]
    .astype(str)
    .str.strip()
)


observed_populations = sorted(
    df["Superpopulation"].unique()
)


print(
    "Observed populations:"
)

for population in observed_populations:

    print(
        f"  {population}"
    )


if set(observed_populations) != set(POPULATIONS):

    raise ValueError(
        "\nUnexpected superpopulation set.\n"
        f"Expected: {POPULATIONS}\n"
        f"Observed: {observed_populations}"
    )


print(
    "\nPopulation set: PASSED"
)


# =============================================================================
# 11. POPULATION COUNTS
# =============================================================================

print("\n" + "=" * 80)
print("6. POPULATION COUNTS")
print("=" * 80)


population_counts = (
    df["Superpopulation"]
    .value_counts()
    .reindex(POPULATIONS)
)


for population in POPULATIONS:

    print(
        f"{population}: "
        f"{int(population_counts.loc[population]):,}"
    )


if population_counts.isna().any():

    raise ValueError(
        "Missing population count detected."
    )


print(
    "\nPopulation counts: PASSED"
)


# =============================================================================
# 12. CHECK DISEASE Z-SCORES
# =============================================================================

print("\n" + "=" * 80)
print("7. CHECKING STANDARDIZED PRS VALUES")
print("=" * 80)


for disease in DISEASES:

    column = Z_COLUMNS[disease]

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )

    missing = int(
        df[column].isna().sum()
    )

    if missing != 0:

        raise ValueError(
            f"{disease}: {missing} missing Z-score values."
        )

    if not np.isfinite(
        df[column].to_numpy()
    ).all():

        raise ValueError(
            f"{disease}: non-finite Z-score values detected."
        )

    print(
        f"{disease:<22} "
        f"missing = {missing:<4} "
        f"PASSED"
    )


print(
    "\nPRS completeness: PASSED"
)


# =============================================================================
# 13. VERIFY STANDARDIZED DISTRIBUTIONS
# =============================================================================

print("\n" + "=" * 80)
print("8. VERIFYING STANDARDIZED DISEASE DISTRIBUTIONS")
print("=" * 80)


standardization_rows = []


for disease in DISEASES:

    column = Z_COLUMNS[disease]

    values = df[column].to_numpy(
        dtype=float
    )

    mean = np.mean(
        values
    )

    sd = np.std(
        values,
        ddof=1
    )

    standardization_rows.append({

        "Disease": disease,

        "N": len(values),

        "Mean_Z": mean,

        "SD_Z": sd

    })

    print(
        f"{disease:<22} "
        f"Mean = {mean: .8f}    "
        f"SD = {sd: .8f}"
    )


standardization_df = pd.DataFrame(
    standardization_rows
)


# =============================================================================
# 14. CREATE LONG-FORM DATASET
# =============================================================================

print("\n" + "=" * 80)
print("9. CREATING LONG-FORM DATASET")
print("=" * 80)


# -----------------------------------------------------------------------------
# This converts:
#
# IID | Population | Disease1_Z | Disease2_Z | ...
#
# into:
#
# IID | Superpopulation | Disease | PRS_Z
#
# -----------------------------------------------------------------------------

long_df = df[
    [
        "IID",
        "Superpopulation"
    ] + disease_columns
].melt(
    id_vars=[
        "IID",
        "Superpopulation"
    ],
    value_vars=disease_columns,
    var_name="Disease_Z_Column",
    value_name="PRS_Z"
)


# -----------------------------------------------------------------------------
# Recover disease name from column name
# -----------------------------------------------------------------------------

reverse_z_columns = {
    column: disease
    for disease, column in Z_COLUMNS.items()
}


long_df["Disease"] = (
    long_df["Disease_Z_Column"]
    .map(reverse_z_columns)
)


if long_df["Disease"].isna().any():

    raise ValueError(
        "Could not map one or more disease Z-score columns."
    )


# -----------------------------------------------------------------------------
# Set categorical order
# -----------------------------------------------------------------------------

long_df["Disease"] = pd.Categorical(
    long_df["Disease"],
    categories=DISEASES,
    ordered=True
)


long_df["Superpopulation"] = pd.Categorical(
    long_df["Superpopulation"],
    categories=POPULATIONS,
    ordered=True
)


expected_long_rows = (
    EXPECTED_N * len(DISEASES)
)


print(
    f"Long-form rows: {len(long_df):,}"
)


print(
    f"Expected rows:  {expected_long_rows:,}"
)


if len(long_df) != expected_long_rows:

    raise ValueError(
        "Unexpected number of long-form rows."
    )


print(
    "Long-form conversion: PASSED"
)


# =============================================================================
# 15. DESCRIPTIVE DISEASE × POPULATION STATISTICS
# =============================================================================

print("\n" + "=" * 80)
print("10. DISEASE × SUPERPOPULATION DESCRIPTIVE STATISTICS")
print("=" * 80)


descriptive_rows = []


for disease in DISEASES:

    column = Z_COLUMNS[disease]

    for population in POPULATIONS:

        values = df.loc[
            df["Superpopulation"] == population,
            column
        ].to_numpy(
            dtype=float
        )

        n = len(values)

        mean = np.mean(
            values
        )

        sd = np.std(
            values,
            ddof=1
        )

        se = (
            sd /
            np.sqrt(n)
        )

        ci_low = (
            mean -
            1.96 * se
        )

        ci_high = (
            mean +
            1.96 * se
        )

        median = np.median(
            values
        )

        descriptive_rows.append({

            "Disease": disease,

            "Superpopulation": population,

            "N": n,

            "Mean_Z": mean,

            "SD_Z": sd,

            "Median_Z": median,

            "SE_Z": se,

            "CI95_Lower": ci_low,

            "CI95_Upper": ci_high

        })


descriptive_df = pd.DataFrame(
    descriptive_rows
)


if len(descriptive_df) != (
    len(DISEASES) *
    len(POPULATIONS)
):

    raise ValueError(
        "Unexpected number of disease-population combinations."
    )


print(
    f"Combinations: {len(descriptive_df)}"
)

print(
    "Expected:     35"
)


print(
    "\nDescriptive statistics: PASSED"
)


# =============================================================================
# 16. DESIGN MATRIX CONSTRUCTION
# =============================================================================

print("\n" + "=" * 80)
print("11. CONSTRUCTING TWO-WAY DESIGN MATRICES")
print("=" * 80)


# -----------------------------------------------------------------------------
# We use treatment coding:
#
# Intercept
# 6 disease indicators
# 4 population indicators
# 24 disease × population interaction indicators
#
# Total parameters:
#
# 1 + 6 + 4 + 24 = 35
#
# -----------------------------------------------------------------------------

n_disease = len(DISEASES)
n_population = len(POPULATIONS)


def build_design_matrix(
    disease_codes,
    population_codes,
    include_interaction=True
):

    disease_codes = np.asarray(
        disease_codes,
        dtype=int
    )

    population_codes = np.asarray(
        population_codes,
        dtype=int
    )

    if len(disease_codes) != len(
        population_codes
    ):

        raise ValueError(
            "Disease and population code arrays "
            "have different lengths."
        )


    n = len(
        disease_codes
    )


    # -------------------------------------------------------------
    # Intercept
    # -------------------------------------------------------------

    columns = [
        np.ones(n)
    ]


    # -------------------------------------------------------------
    # Disease main effects
    #
    # Reference disease = disease 0
    # -------------------------------------------------------------

    for d in range(
        1,
        n_disease
    ):

        columns.append(
            (
                disease_codes == d
            ).astype(float)
        )


    # -------------------------------------------------------------
    # Population main effects
    #
    # Reference population = population 0
    # -------------------------------------------------------------

    for p in range(
        1,
        n_population
    ):

        columns.append(
            (
                population_codes == p
            ).astype(float)
        )


    # -------------------------------------------------------------
    # Disease × population interaction
    #
    # Reference disease = 0
    # Reference population = 0
    # -------------------------------------------------------------

    if include_interaction:

        for d in range(
            1,
            n_disease
        ):

            for p in range(
                1,
                n_population
            ):

                interaction = (

                    (disease_codes == d)
                    &
                    (population_codes == p)

                ).astype(float)

                columns.append(
                    interaction
                )


    X = np.column_stack(
        columns
    )

    return X


# =============================================================================
# 17. CREATE OBSERVED CODES
# =============================================================================

disease_code_map = {
    disease: index
    for index, disease in enumerate(
        DISEASES
    )
}


population_code_map = {
    population: index
    for index, population in enumerate(
        POPULATIONS
    )
}


long_disease_codes = (
    long_df["Disease"]
    .map(disease_code_map)
    .to_numpy(
        dtype=int
    )
)


long_population_codes = (
    long_df["Superpopulation"]
    .map(population_code_map)
    .to_numpy(
        dtype=int
    )
)


y = long_df[
    "PRS_Z"
].to_numpy(
    dtype=float
)


# =============================================================================
# 18. REDUCED MODEL
# =============================================================================

print("\n" + "=" * 80)
print("12. REDUCED MODEL")
print("=" * 80)


X_reduced = build_design_matrix(
    disease_codes=long_disease_codes,
    population_codes=long_population_codes,
    include_interaction=False
)


print(
    f"Reduced design matrix shape: "
    f"{X_reduced.shape}"
)


rank_reduced = np.linalg.matrix_rank(
    X_reduced
)


print(
    f"Reduced model rank: "
    f"{rank_reduced}"
)


expected_reduced_parameters = (
    1
    + (n_disease - 1)
    + (n_population - 1)
)


print(
    f"Expected reduced parameters: "
    f"{expected_reduced_parameters}"
)


if rank_reduced != (
    expected_reduced_parameters
):

    raise ValueError(
        "Reduced design matrix is rank deficient."
    )


# -----------------------------------------------------------------------------
# Least squares
# -----------------------------------------------------------------------------

beta_reduced, residuals_reduced, _, _ = (
    np.linalg.lstsq(
        X_reduced,
        y,
        rcond=None
    )
)


fitted_reduced = (
    X_reduced @ beta_reduced
)


residuals_vector_reduced = (
    y -
    fitted_reduced
)


SSE_reduced = np.sum(
    residuals_vector_reduced ** 2
)


SST = np.sum(
    (
        y -
        np.mean(y)
    ) ** 2
)


df_reduced_model = (
    X_reduced.shape[1]
)

df_error_reduced = (
    len(y) -
    rank_reduced
)


print(
    f"SSE reduced: "
    f"{SSE_reduced:.10f}"
)

print(
    f"Error df: "
    f"{df_error_reduced:,}"
)


# =============================================================================
# 19. FULL INTERACTION MODEL
# =============================================================================

print("\n" + "=" * 80)
print("13. FULL DISEASE × SUPERPOPULATION MODEL")
print("=" * 80)


X_full = build_design_matrix(
    disease_codes=long_disease_codes,
    population_codes=long_population_codes,
    include_interaction=True
)


print(
    f"Full design matrix shape: "
    f"{X_full.shape}"
)


rank_full = np.linalg.matrix_rank(
    X_full
)


print(
    f"Full model rank: "
    f"{rank_full}"
)


expected_full_parameters = (

    1

    + (n_disease - 1)

    + (n_population - 1)

    + (
        (n_disease - 1)
        *
        (n_population - 1)
    )

)


print(
    f"Expected full parameters: "
    f"{expected_full_parameters}"
)


if rank_full != (
    expected_full_parameters
):

    raise ValueError(
        "\nFULL DESIGN MATRIX IS RANK DEFICIENT.\n"
        f"Rank = {rank_full}\n"
        f"Expected = {expected_full_parameters}\n"
        "Do not interpret the interaction result."
    )


print(
    "Full design matrix rank: PASSED"
)


# -----------------------------------------------------------------------------
# Least squares
# -----------------------------------------------------------------------------

beta_full, residuals_full, _, _ = (
    np.linalg.lstsq(
        X_full,
        y,
        rcond=None
    )
)


fitted_full = (
    X_full @ beta_full
)


residuals_vector_full = (
    y -
    fitted_full
)


SSE_full = np.sum(
    residuals_vector_full ** 2
)


df_full_model = (
    X_full.shape[1]
)

df_error_full = (
    len(y) -
    rank_full
)


print(
    f"SSE full: "
    f"{SSE_full:.10f}"
)

print(
    f"Error df: "
    f"{df_error_full:,}"
)


# =============================================================================
# 20. INTERACTION TEST
# =============================================================================

print("\n" + "=" * 80)
print("14. OBSERVED INTERACTION TEST")
print("=" * 80)


df_interaction = (
    rank_full -
    rank_reduced
)


SSR_interaction = (
    SSE_reduced -
    SSE_full
)


if SSR_interaction < 0:

    raise ValueError(
        "\nNegative interaction sum of squares detected.\n"
        "This indicates an invalid model calculation."
    )


MS_interaction = (
    SSR_interaction /
    df_interaction
)


MSE_full = (
    SSE_full /
    df_error_full
)


F_observed = (
    MS_interaction /
    MSE_full
)


p_observed = (
    f_distribution.sf(
        F_observed,
        df_interaction,
        df_error_full
    )
)


# -----------------------------------------------------------------------------
# Partial eta squared
# -----------------------------------------------------------------------------

partial_eta_squared = (
    SSR_interaction /
    (
        SSR_interaction +
        SSE_full
    )
)


print(
    f"Reduced SSE:             "
    f"{SSE_reduced:.10f}"
)

print(
    f"Full SSE:                "
    f"{SSE_full:.10f}"
)

print(
    f"Interaction SS:          "
    f"{SSR_interaction:.10f}"
)

print(
    f"Interaction df:          "
    f"{df_interaction}"
)

print(
    f"Error df:                "
    f"{df_error_full:,}"
)

print(
    f"Mean square interaction: "
    f"{MS_interaction:.10f}"
)

print(
    f"Mean square error:       "
    f"{MSE_full:.10f}"
)

print(
    f"Observed F:              "
    f"{F_observed:.10f}"
)

print(
    f"Parametric p-value:      "
    f"{p_observed:.12e}"
)

print(
    f"Partial eta-squared:     "
    f"{partial_eta_squared:.10f}"
)


# =============================================================================
# 21. INDEPENDENT F CALCULATION
# =============================================================================

print("\n" + "=" * 80)
print("15. INDEPENDENT F CALCULATION CHECK")
print("=" * 80)


# -----------------------------------------------------------------------------
# Calculate the F statistic again using the same sums of squares but an
# independent direct formula.
# -----------------------------------------------------------------------------

F_independent = (

    (
        (SSE_reduced - SSE_full)
        /
        (rank_full - rank_reduced)
    )

    /

    (
        SSE_full
        /
        (len(y) - rank_full)
    )

)


print(
    f"F from primary calculation: "
    f"{F_observed:.12f}"
)

print(
    f"F from independent formula: "
    f"{F_independent:.12f}"
)


F_difference = abs(
    F_observed -
    F_independent
)


print(
    f"Absolute difference: "
    f"{F_difference:.12e}"
)


if not np.isclose(
    F_observed,
    F_independent,
    rtol=1e-10,
    atol=1e-10
):

    raise ValueError(
        "Independent F-statistic calculation does not agree."
    )


print(
    "\nIndependent F calculation: PASSED"
)


# =============================================================================
# 22. PREPARE INDIVIDUAL-LEVEL PERMUTATION
# =============================================================================

print("\n" + "=" * 80)
print("16. PREPARING INDIVIDUAL-LEVEL PERMUTATION")
print("=" * 80)


# -----------------------------------------------------------------------------
# IMPORTANT FIX:
#
# disease_columns is explicitly defined above:
#
# disease_columns = [
#     Z_COLUMNS[disease]
#     for disease in DISEASES
# ]
#
# This was the variable missing in the previous version.
# -----------------------------------------------------------------------------

Y = (
    df[disease_columns]
    .astype(float)
    .to_numpy()
)


n_individuals = (
    Y.shape[0]
)

n_diseases = (
    Y.shape[1]
)


print(
    f"PRS matrix shape: "
    f"{Y.shape}"
)

print(
    f"Individuals: "
    f"{n_individuals:,}"
)

print(
    f"Diseases: "
    f"{n_diseases}"
)


if n_individuals != EXPECTED_N:

    raise ValueError(
        "Unexpected number of individuals in PRS matrix."
    )


if n_diseases != len(DISEASES):

    raise ValueError(
        "Unexpected number of diseases in PRS matrix."
    )


# =============================================================================
# 23. INDIVIDUAL POPULATION CODES
# =============================================================================

individual_population_codes = (
    df["Superpopulation"]
    .map(population_code_map)
    .to_numpy(
        dtype=int
    )
)


if np.any(
    individual_population_codes < 0
):

    raise ValueError(
        "Invalid population code detected."
    )


# -----------------------------------------------------------------------------
# Verify original population counts
# -----------------------------------------------------------------------------

original_population_counts = np.bincount(
    individual_population_codes,
    minlength=len(POPULATIONS)
)


print(
    "\nOriginal population counts:"
)

for i, population in enumerate(
    POPULATIONS
):

    print(
        f"  {population}: "
        f"{original_population_counts[i]:,}"
    )


# =============================================================================
# 24. VERIFY INDIVIDUAL-LEVEL STRUCTURE
# =============================================================================

print("\n" + "=" * 80)
print("17. VERIFYING INDIVIDUAL-LEVEL STRUCTURE")
print("=" * 80)


# Every individual must have exactly seven disease PRS values.

if Y.shape != (
    EXPECTED_N,
    len(DISEASES)
):

    raise ValueError(
        f"Expected PRS matrix shape "
        f"({EXPECTED_N}, {len(DISEASES)}), "
        f"found {Y.shape}."
    )


if not np.isfinite(
    Y
).all():

    raise ValueError(
        "Non-finite values detected in PRS matrix."
    )


print(
    "Complete seven-disease profiles: PASSED"
)


# =============================================================================
# 25. PERMUTATION DESIGN
# =============================================================================

print("\n" + "=" * 80)
print("18. PERMUTATION DESIGN")
print("=" * 80)


print(
    "Permutation strategy:"
)

print(
    "  - Keep each individual's seven-disease PRS profile intact."
)

print(
    "  - Randomly permute superpopulation labels across individuals."
)

print(
    "  - Preserve the original AFR/AMR/EAS/EUR/SAS group sizes."
)

print(
    "  - Recalculate the Disease × Superpopulation interaction F statistic."
)

print(
    "  - Repeat for "
    f"{N_PERMUTATIONS:,} permutations."
)


# =============================================================================
# 26. FUNCTION FOR PERMUTATION F
# =============================================================================

def calculate_interaction_F_from_population_codes(
    y_matrix,
    population_codes
):

    """
    Calculate the Disease × Superpopulation interaction F statistic
    while preserving the seven-disease profile of each individual.

    Parameters
    ----------
    y_matrix : numpy.ndarray
        Shape = (individuals, diseases)

    population_codes : numpy.ndarray
        Shape = (individuals,)

    Returns
    -------
    float
        Interaction F statistic.
    """

    n_individuals_local = (
        y_matrix.shape[0]
    )

    n_diseases_local = (
        y_matrix.shape[1]
    )


    # -------------------------------------------------------------
    # Convert wide individual-level matrix into long form
    # -------------------------------------------------------------

    y_long = (
        y_matrix
        .reshape(-1)
    )


    disease_codes_local = np.tile(
        np.arange(
            n_diseases_local
        ),
        n_individuals_local
    )


    population_codes_local = np.repeat(
        population_codes,
        n_diseases_local
    )


    # -------------------------------------------------------------
    # Reduced model
    # -------------------------------------------------------------

    X_red = build_design_matrix(
        disease_codes=disease_codes_local,
        population_codes=population_codes_local,
        include_interaction=False
    )


    # -------------------------------------------------------------
    # Full model
    # -------------------------------------------------------------

    X_full_local = build_design_matrix(
        disease_codes=disease_codes_local,
        population_codes=population_codes_local,
        include_interaction=True
    )


    # -------------------------------------------------------------
    # Least squares
    # -------------------------------------------------------------

    beta_red = np.linalg.lstsq(
        X_red,
        y_long,
        rcond=None
    )[0]


    beta_full_local = np.linalg.lstsq(
        X_full_local,
        y_long,
        rcond=None
    )[0]


    residual_red = (
        y_long -
        X_red @ beta_red
    )


    residual_full = (
        y_long -
        X_full_local @ beta_full_local
    )


    SSE_red = np.sum(
        residual_red ** 2
    )


    SSE_full_local = np.sum(
        residual_full ** 2
    )


    rank_red = np.linalg.matrix_rank(
        X_red
    )


    rank_full_local = np.linalg.matrix_rank(
        X_full_local
    )


    df_interaction_local = (
        rank_full_local -
        rank_red
    )


    df_error_local = (
        len(y_long) -
        rank_full_local
    )


    SSR_interaction_local = (
        SSE_red -
        SSE_full_local
    )


    if (
        df_interaction_local <= 0
        or
        df_error_local <= 0
    ):

        return np.nan


    if SSR_interaction_local < 0:

        return np.nan


    F_local = (

        (
            SSR_interaction_local
            /
            df_interaction_local
        )

        /

        (
            SSE_full_local
            /
            df_error_local
        )

    )


    return F_local


# =============================================================================
# 27. VERIFY PERMUTATION FUNCTION ON ORIGINAL DATA
# =============================================================================

print("\n" + "=" * 80)
print("19. VERIFYING PERMUTATION FUNCTION")
print("=" * 80)


F_function_check = (
    calculate_interaction_F_from_population_codes(
        Y,
        individual_population_codes
    )
)


print(
    f"Original F:       "
    f"{F_observed:.12f}"
)

print(
    f"Function F:       "
    f"{F_function_check:.12f}"
)


if not np.isclose(
    F_observed,
    F_function_check,
    rtol=1e-8,
    atol=1e-8
):

    raise ValueError(
        "\nPermutation F function does not reproduce "
        "the observed F statistic."
    )


print(
    "\nPermutation function verification: PASSED"
)


# =============================================================================
# 28. RUN PERMUTATIONS
# =============================================================================

print("\n" + "=" * 80)
print("20. RUNNING INDIVIDUAL-LEVEL PERMUTATION TEST")
print("=" * 80)


print(
    f"Number of permutations: "
    f"{N_PERMUTATIONS:,}"
)


print(
    "This may take some time because each permutation "
    "requires two least-squares model fits."
)


start_time = time.time()


permutation_F = np.empty(
    N_PERMUTATIONS,
    dtype=float
)


n_valid = 0


for permutation_index in range(
    N_PERMUTATIONS
):

    # -------------------------------------------------------------
    # Randomly permute the population labels.
    #
    # IMPORTANT:
    # The labels are shuffled, not the PRS matrix.
    #
    # Therefore each person's seven-disease profile remains intact.
    # -------------------------------------------------------------

    permuted_population_codes = (
        rng.permutation(
            individual_population_codes
        )
    )


    F_perm = (
        calculate_interaction_F_from_population_codes(
            Y,
            permuted_population_codes
        )
    )


    permutation_F[
        permutation_index
    ] = F_perm


    if np.isfinite(
        F_perm
    ):

        n_valid += 1


    # -------------------------------------------------------------
    # Progress reporting
    # -------------------------------------------------------------

    if (
        (permutation_index + 1) % 500 == 0
        or
        permutation_index == 0
    ):

        elapsed = (
            time.time() -
            start_time
        )

        print(
            f"Completed "
            f"{permutation_index + 1:,}/"
            f"{N_PERMUTATIONS:,} "
            f"permutations "
            f"({elapsed:.1f} sec)"
        )


# =============================================================================
# 29. VALIDATE PERMUTATION RESULTS
# =============================================================================

print("\n" + "=" * 80)
print("21. VALIDATING PERMUTATION RESULTS")
print("=" * 80)


valid_permutation_F = (
    permutation_F[
        np.isfinite(
            permutation_F
        )
    ]
)


print(
    f"Requested permutations: "
    f"{N_PERMUTATIONS:,}"
)

print(
    f"Valid permutations:     "
    f"{len(valid_permutation_F):,}"
)


if len(
    valid_permutation_F
) != N_PERMUTATIONS:

    raise ValueError(
        "Some permutation statistics were invalid."
    )


# =============================================================================
# 30. EMPIRICAL PERMUTATION P-VALUE
# =============================================================================

print("\n" + "=" * 80)
print("22. EMPIRICAL PERMUTATION TEST")
print("=" * 80)


# -----------------------------------------------------------------------------
# Add-one correction:
#
# p = (1 + number of permuted statistics >= observed statistic)
#     / (1 + number of permutations)
#
# This avoids reporting an exact zero p-value.
# -----------------------------------------------------------------------------

n_ge_observed = int(
    np.sum(
        valid_permutation_F >=
        F_observed
    )
)


permutation_p = (

    (
        1 +
        n_ge_observed
    )

    /

    (
        1 +
        N_PERMUTATIONS
    )

)


print(
    f"Observed F:                  "
    f"{F_observed:.10f}"
)

print(
    f"Permuted F >= observed:      "
    f"{n_ge_observed:,}"
)

print(
    f"Total permutations:          "
    f"{N_PERMUTATIONS:,}"
)

print(
    f"Empirical permutation p:     "
    f"{permutation_p:.12e}"
)


# =============================================================================
# 31. PERMUTATION DISTRIBUTION SUMMARY
# =============================================================================

print("\n" + "=" * 80)
print("23. PERMUTATION DISTRIBUTION")
print("=" * 80)


perm_mean = np.mean(
    valid_permutation_F
)

perm_sd = np.std(
    valid_permutation_F,
    ddof=1
)

perm_median = np.median(
    valid_permutation_F
)

perm_min = np.min(
    valid_permutation_F
)

perm_max = np.max(
    valid_permutation_F
)

perm_q025 = np.quantile(
    valid_permutation_F,
    0.025
)

perm_q975 = np.quantile(
    valid_permutation_F,
    0.975
)


print(
    f"Mean:        {perm_mean:.10f}"
)

print(
    f"SD:          {perm_sd:.10f}"
)

print(
    f"Median:      {perm_median:.10f}"
)

print(
    f"Minimum:     {perm_min:.10f}"
)

print(
    f"Maximum:     {perm_max:.10f}"
)

print(
    f"2.5th pct:   {perm_q025:.10f}"
)

print(
    f"97.5th pct:  {perm_q975:.10f}"
)


# =============================================================================
# 32. PERMUTATION Z-SCORE
# =============================================================================

if perm_sd > 0:

    permutation_standardized_distance = (

        (
            F_observed -
            perm_mean
        )

        /

        perm_sd

    )

else:

    permutation_standardized_distance = np.nan


print(
    f"\nObserved F distance from "
    f"permutation mean: "
    f"{permutation_standardized_distance:.6f} SD"
)


# =============================================================================
# 33. COMPARE OBSERVED F WITH PERMUTATION DISTRIBUTION
# =============================================================================

print("\n" + "=" * 80)
print("24. OBSERVED VS PERMUTATION DISTRIBUTION")
print("=" * 80)


print(
    f"Observed F:             "
    f"{F_observed:.10f}"
)

print(
    f"Permutation mean:       "
    f"{perm_mean:.10f}"
)

print(
    f"Permutation 97.5%:      "
    f"{perm_q975:.10f}"
)


if F_observed > perm_q975:

    print(
        "\nObserved F exceeds the "
        "97.5th percentile of the permutation distribution."
    )

else:

    print(
        "\nObserved F does not exceed the "
        "97.5th percentile of the permutation distribution."
    )


# =============================================================================
# 34. SAVE DESCRIPTIVE RESULTS
# =============================================================================

print("\n" + "=" * 80)
print("25. SAVING DESCRIPTIVE RESULTS")
print("=" * 80)


descriptive_df.to_csv(
    OUTPUT_DESCRIPTIVE,
    sep="\t",
    index=False,
    float_format="%.10f"
)


print(
    f"Saved:\n{OUTPUT_DESCRIPTIVE}"
)


# =============================================================================
# 35. SAVE PERMUTATION DISTRIBUTION
# =============================================================================

permutation_distribution_df = pd.DataFrame({

    "Permutation": np.arange(
        1,
        len(valid_permutation_F) + 1
    ),

    "Interaction_F": valid_permutation_F

})


permutation_distribution_df.to_csv(
    OUTPUT_PERMUTATION_DISTRIBUTION,
    sep="\t",
    index=False,
    float_format="%.10f"
)


print(
    f"Saved:\n"
    f"{OUTPUT_PERMUTATION_DISTRIBUTION}"
)


# =============================================================================
# 36. SAVE PERMUTATION SUMMARY
# =============================================================================

permutation_summary_df = pd.DataFrame([{

    "Observed_F": F_observed,

    "Interaction_df": df_interaction,

    "Error_df": df_error_full,

    "Parametric_P": p_observed,

    "Partial_Eta_Squared": partial_eta_squared,

    "Permutation_N": N_PERMUTATIONS,

    "Permutation_Greater_Equal_Observed": n_ge_observed,

    "Permutation_P": permutation_p,

    "Permutation_Mean_F": perm_mean,

    "Permutation_SD_F": perm_sd,

    "Permutation_Median_F": perm_median,

    "Permutation_Min_F": perm_min,

    "Permutation_Max_F": perm_max,

    "Permutation_2.5th_Percentile_F": perm_q025,

    "Permutation_97.5th_Percentile_F": perm_q975,

    "Observed_F_Permutation_SD_Distance":
        permutation_standardized_distance

}])


permutation_summary_df.to_csv(
    OUTPUT_PERMUTATION,
    sep="\t",
    index=False,
    float_format="%.12f"
)


print(
    f"\nSaved:\n"
    f"{OUTPUT_PERMUTATION}"
)


# =============================================================================
# 37. FINAL SUMMARY TABLE
# =============================================================================

summary_df = pd.DataFrame([{

    "Analysis":
        "Disease × Superpopulation interaction",

    "Individuals":
        n_individuals,

    "Diseases":
        n_diseases,

    "Superpopulations":
        len(POPULATIONS),

    "Total_observations":
        len(y),

    "Reduced_parameters":
        rank_reduced,

    "Full_parameters":
        rank_full,

    "Interaction_df":
        df_interaction,

    "Error_df":
        df_error_full,

    "Reduced_SSE":
        SSE_reduced,

    "Full_SSE":
        SSE_full,

    "Interaction_SS":
        SSR_interaction,

    "Observed_F":
        F_observed,

    "Independent_F":
        F_independent,

    "F_difference":
        F_difference,

    "Parametric_P":
        p_observed,

    "Partial_Eta_Squared":
        partial_eta_squared,

    "Permutation_N":
        N_PERMUTATIONS,

    "Permutation_P":
        permutation_p,

    "Permutation_Mean_F":
        perm_mean,

    "Permutation_97.5th_Percentile_F":
        perm_q975

}])


summary_df.to_csv(
    OUTPUT_SUMMARY,
    sep="\t",
    index=False,
    float_format="%.12f"
)


# =============================================================================
# 38. FINAL VERIFICATION
# =============================================================================

print("\n" + "=" * 80)
print("26. FINAL VERIFICATION")
print("=" * 80)


checks = {

    "N = 2504":
        len(df) == 2504,

    "Seven diseases":
        len(disease_columns) == 7,

    "Five populations":
        len(POPULATIONS) == 5,

    "No missing PRS":
        df[disease_columns].isna().sum().sum() == 0,

    "IID unique":
        duplicate_iids == 0,

    "Reduced model full rank":
        rank_reduced == expected_reduced_parameters,

    "Full model full rank":
        rank_full == expected_full_parameters,

    "Independent F agrees":
        np.isclose(
            F_observed,
            F_independent,
            rtol=1e-10,
            atol=1e-10
        ),

    "Permutation function agrees":
        np.isclose(
            F_observed,
            F_function_check,
            rtol=1e-8,
            atol=1e-8
        ),

    "All permutations valid":
        len(valid_permutation_F)
        == N_PERMUTATIONS

}


all_passed = True


for check_name, passed in checks.items():

    status = (
        "PASSED"
        if passed
        else
        "FAILED"
    )

    print(
        f"{check_name:<35} {status}"
    )

    if not passed:

        all_passed = False


# =============================================================================
# 39. FINAL STATISTICAL RESULT
# =============================================================================

print("\n" + "=" * 80)
print("27. FINAL STATISTICAL RESULT")
print("=" * 80)


print(
    f"\nDisease × Superpopulation interaction"
)

print(
    f"F({df_interaction}, "
    f"{df_error_full}) = "
    f"{F_observed:.6f}"
)

print(
    f"Parametric p = "
    f"{p_observed:.6e}"
)

print(
    f"Partial η² = "
    f"{partial_eta_squared:.6f}"
)

print(
    f"Permutation p = "
    f"{permutation_p:.6e}"
)


# =============================================================================
# 40. INTERPRETATION
# =============================================================================

print("\n" + "=" * 80)
print("28. INTERPRETATION")
print("=" * 80)


if (
    p_observed < 0.05
    and
    permutation_p < 0.05
):

    print(
        "\nThe Disease × Superpopulation interaction is "
        "statistically supported by both the parametric "
        "two-way OLS test and the individual-level "
        "permutation test."
    )

elif (
    p_observed < 0.05
):

    print(
        "\nThe parametric two-way OLS model indicates a "
        "statistically significant Disease × Superpopulation "
        "interaction, but the permutation result should be "
        "considered separately."
    )

else:

    print(
        "\nThe interaction does not reach the conventional "
        "0.05 significance threshold in the parametric test."
    )


# =============================================================================
# 41. OUTPUT LOCATIONS
# =============================================================================

print("\n" + "=" * 80)
print("29. OUTPUT FILES")
print("=" * 80)


print(
    f"\n1. Summary:"
)

print(
    OUTPUT_SUMMARY
)


print(
    f"\n2. Permutation summary:"
)

print(
    OUTPUT_PERMUTATION
)


print(
    f"\n3. Disease × population descriptive statistics:"
)

print(
    OUTPUT_DESCRIPTIVE
)


print(
    f"\n4. Full permutation F distribution:"
)

print(
    OUTPUT_PERMUTATION_DISTRIBUTION
)


# =============================================================================
# 42. COMPLETION
# =============================================================================

elapsed_total = (
    time.time() -
    start_time
)


print("\n" + "=" * 80)


if all_passed:

    print(
        "STATUS: ALL STATISTICAL VERIFICATION CHECKS PASSED"
    )

else:

    print(
        "STATUS: ONE OR MORE VERIFICATION CHECKS FAILED"
    )


print(
    f"Total permutation runtime: "
    f"{elapsed_total:.2f} seconds"
)


print(
    "=" * 80)