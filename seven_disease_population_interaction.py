import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import f


# =============================================================================
# SEVEN-DISEASE × SUPERPOPULATION INTERACTION ANALYSIS
# =============================================================================
#
# Purpose:
# Test whether the relationship between disease and standardized PRS differs
# across the five 1000 Genomes superpopulations.
#
# Data:
#   2,504 individuals
#   7 diseases
#   5 superpopulations
#
# Repeated-measures structure:
#   Each individual contributes one standardized PRS value for each disease.
#
# Primary comparison:
#
#   Reduced model:
#       PRS_Z ~ Disease + IID
#
#   Full model:
#       PRS_Z ~ Disease + IID
#                  + Disease × Superpopulation
#
# IID is included as a fixed subject/block effect.
#
# The interaction is tested by comparing the residual sum of squares of the
# reduced and full nested models.
#
# This approach:
#   - handles unequal superpopulation sample sizes
#   - avoids the singular random-effects covariance problem
#   - accounts for repeated observations from the same individual
#   - directly tests Disease × Superpopulation interaction
#
# =============================================================================


# =============================================================================
# 1. PATHS
# =============================================================================

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
    "disease_population_interaction"
)

os.makedirs(
    OUT_DIR,
    exist_ok=True
)


# =============================================================================
# 2. DISEASE DEFINITIONS
# =============================================================================

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


# =============================================================================
# 3. SUPERPOPULATIONS
# =============================================================================

POPULATIONS = [
    "AFR",
    "AMR",
    "EAS",
    "EUR",
    "SAS"
]


# =============================================================================
# 4. HEADER
# =============================================================================

print("=" * 80)
print("SEVEN-DISEASE × SUPERPOPULATION INTERACTION ANALYSIS")
print("=" * 80)


# =============================================================================
# 5. CHECK INPUT
# =============================================================================

if not os.path.exists(INPUT):

    raise FileNotFoundError(
        "\nInput file not found:\n"
        f"{INPUT}"
    )

print("\nInput file:")
print(INPUT)


# =============================================================================
# 6. LOAD DATA
# =============================================================================

df = pd.read_csv(
    INPUT,
    sep="\t"
)

print(
    f"\nRows: {len(df)}"
)

print(
    f"Columns: {len(df.columns)}"
)


# =============================================================================
# 7. VALIDATION
# =============================================================================

print("\n" + "=" * 80)
print("DATA VALIDATION")
print("=" * 80)


# -------------------------------------------------------------------------
# Expected number of individuals
# -------------------------------------------------------------------------

if len(df) != 2504:

    raise ValueError(
        f"Expected 2504 individuals, "
        f"found {len(df)}."
    )


# -------------------------------------------------------------------------
# IID
# -------------------------------------------------------------------------

if "IID" not in df.columns:

    raise ValueError(
        "IID column not found."
    )


# -------------------------------------------------------------------------
# Superpopulation
# -------------------------------------------------------------------------

if "Superpopulation" not in df.columns:

    raise ValueError(
        "Superpopulation column not found."
    )


# -------------------------------------------------------------------------
# Duplicate IID check
# -------------------------------------------------------------------------

if df["IID"].duplicated().any():

    duplicate_ids = (
        df.loc[
            df["IID"].duplicated(),
            "IID"
        ]
        .astype(str)
        .tolist()
    )

    raise ValueError(
        "Duplicate IID values detected.\n"
        f"Examples: {duplicate_ids[:10]}"
    )


# -------------------------------------------------------------------------
# Superpopulation check
# -------------------------------------------------------------------------

observed_populations = set(
    df["Superpopulation"]
    .astype(str)
    .unique()
)

expected_populations = set(
    POPULATIONS
)

if observed_populations != expected_populations:

    raise ValueError(
        "\nUnexpected superpopulations.\n"
        f"Expected: {expected_populations}\n"
        f"Observed: {observed_populations}"
    )


# -------------------------------------------------------------------------
# Disease PRS columns
# -------------------------------------------------------------------------

for disease_key, info in DISEASES.items():

    column = info["column"]

    if column not in df.columns:

        raise ValueError(
            f"Missing PRS column: {column}"
        )

    missing = df[column].isna().sum()

    if missing > 0:

        raise ValueError(
            f"{column} contains "
            f"{missing} missing values."
        )


print("\nValidation: PASSED")


# =============================================================================
# 8. POPULATION COUNTS
# =============================================================================

print("\n" + "-" * 80)
print("SUPERPOPULATION SAMPLE COUNTS")
print("-" * 80)

population_counts = (
    df["Superpopulation"]
    .value_counts()
    .reindex(POPULATIONS)
)

print(
    population_counts.to_string()
)

print(
    f"\nTotal individuals: "
    f"{population_counts.sum()}"
)


# =============================================================================
# 9. CREATE LONG FORMAT
# =============================================================================

print("\n" + "=" * 80)
print("CREATING LONG-FORM DATA")
print("=" * 80)


long_parts = []


for disease_key, info in DISEASES.items():

    column = info["column"]

    temp = df[
        [
            "IID",
            "Superpopulation",
            column
        ]
    ].copy()

    temp = temp.rename(
        columns={
            column: "PRS_Z"
        }
    )

    temp["Disease"] = info["name"]

    temp = temp[
        [
            "IID",
            "Superpopulation",
            "Disease",
            "PRS_Z"
        ]
    ]

    long_parts.append(
        temp
    )


long_df = pd.concat(
    long_parts,
    ignore_index=True
)


print(
    f"\nLong-format observations: "
    f"{len(long_df):,}"
)


# =============================================================================
# 10. EXPECTED OBSERVATION COUNT
# =============================================================================

expected_observations = (
    2504 * len(DISEASES)
)

if len(long_df) != expected_observations:

    raise ValueError(
        f"Expected {expected_observations:,} observations, "
        f"found {len(long_df):,}."
    )


# =============================================================================
# 11. IID × DISEASE UNIQUENESS
# =============================================================================

duplicate_cells = (
    long_df
    .duplicated(
        subset=[
            "IID",
            "Disease"
        ]
    )
    .sum()
)


if duplicate_cells > 0:

    raise ValueError(
        f"Found {duplicate_cells} duplicate "
        f"IID × Disease observations."
    )


# =============================================================================
# 12. MISSING VALUE CHECK
# =============================================================================

if long_df[
    [
        "IID",
        "Superpopulation",
        "Disease",
        "PRS_Z"
    ]
].isna().any().any():

    raise ValueError(
        "Missing values detected in long-format data."
    )


# =============================================================================
# 13. CATEGORICAL VARIABLES
# =============================================================================

long_df["Disease"] = pd.Categorical(
    long_df["Disease"],
    categories=[
        info["name"]
        for info in DISEASES.values()
    ],
    ordered=True
)

long_df["Superpopulation"] = pd.Categorical(
    long_df["Superpopulation"],
    categories=POPULATIONS,
    ordered=True
)


# =============================================================================
# 14. IID AS STRING
# =============================================================================
#
# Treat IID as a categorical subject/block variable.
#
# =============================================================================

long_df["IID"] = (
    long_df["IID"]
    .astype(str)
)


# =============================================================================
# 15. SAVE LONG DATA
# =============================================================================

long_file = os.path.join(
    OUT_DIR,
    "seven_disease_standardized_PRS_long.tsv"
)

long_df.to_csv(
    long_file,
    sep="\t",
    index=False,
    float_format="%.10f"
)


print("\nLong-format dataset saved:")
print(long_file)


# =============================================================================
# 16. CHECK BALANCED REPEATED-MEASURES STRUCTURE
# =============================================================================

print("\n" + "-" * 80)
print("REPEATED-MEASURES STRUCTURE CHECK")
print("-" * 80)


observations_per_iid = (
    long_df
    .groupby(
        "IID",
        observed=False
    )
    .size()
)


if not (
    observations_per_iid == len(DISEASES)
).all():

    raise ValueError(
        "Not every individual has exactly "
        "seven disease observations."
    )


print(
    "Every individual has exactly "
    f"{len(DISEASES)} disease observations."
)


# =============================================================================
# 17. REDUCED MODEL
# =============================================================================
#
# Reduced model:
#
#     PRS_Z ~ Disease + IID
#
# Disease captures systematic differences between diseases.
#
# IID captures repeated-measures / subject-specific baseline variation.
#
# =============================================================================

print("\n" + "=" * 80)
print("REDUCED MODEL")
print("=" * 80)

print(
    "\nFormula:"
)

print(
    "PRS_Z ~ C(Disease) + C(IID)"
)


reduced_model = smf.ols(
    formula=(
        "PRS_Z ~ C(Disease) + C(IID)"
    ),
    data=long_df
)


reduced_result = reduced_model.fit()


print(
    "\nReduced model fitted successfully."
)


print(
    f"Residual SS: "
    f"{reduced_result.ssr:.10f}"
)

print(
    f"Residual df: "
    f"{reduced_result.df_resid:.0f}"
)


# =============================================================================
# 18. FULL MODEL
# =============================================================================
#
# Full model:
#
#     PRS_Z ~ Disease
#          + IID
#          + Disease × Superpopulation
#
# The population main effect is intentionally not included separately because
# Superpopulation is constant for each IID and is therefore completely
# represented by the IID subject effects.
#
# The Disease × Superpopulation interaction remains estimable because disease
# varies within each individual while superpopulation remains between subjects.
#
# =============================================================================

print("\n" + "=" * 80)
print("FULL MODEL")
print("=" * 80)

print(
    "\nFormula:"
)

print(
    "PRS_Z ~ C(Disease) + C(IID) "
    "+ C(Disease):C(Superpopulation)"
)


full_model = smf.ols(
    formula=(
        "PRS_Z ~ C(Disease) + C(IID) "
        "+ C(Disease):C(Superpopulation)"
    ),
    data=long_df
)


full_result = full_model.fit()


print(
    "\nFull model fitted successfully."
)


print(
    f"Residual SS: "
    f"{full_result.ssr:.10f}"
)

print(
    f"Residual df: "
    f"{full_result.df_resid:.0f}"
)


# =============================================================================
# 19. NESTED MODEL COMPARISON
# =============================================================================
#
# Difference in residual SS is attributable to the interaction.
#
# F =
#
#   [(SSR_reduced - SSR_full) / df_difference]
#   ------------------------------------------------
#                SSR_full / df_full
#
# =============================================================================

print("\n" + "=" * 80)
print("DISEASE × SUPERPOPULATION INTERACTION TEST")
print("=" * 80)


SSR_reduced = (
    reduced_result.ssr
)

SSR_full = (
    full_result.ssr
)


df_reduced = (
    reduced_result.df_resid
)

df_full = (
    full_result.df_resid
)


SS_difference = (
    SSR_reduced
    - SSR_full
)


df_difference = (
    df_reduced
    - df_full
)


MS_difference = (
    SS_difference
    / df_difference
)


MS_full_error = (
    SSR_full
    / df_full
)


F_interaction = (
    MS_difference
    / MS_full_error
)


p_interaction = f.sf(
    F_interaction,
    df_difference,
    df_full
)


# =============================================================================
# 20. PARTIAL ETA-SQUARED
# =============================================================================
#
# Partial eta squared for the interaction:
#
#     SS_interaction
#     ---------------------------
#     SS_interaction + SS_error
#
# =============================================================================

partial_eta_squared = (
    SS_difference
    / (
        SS_difference
        + SSR_full
    )
)


# =============================================================================
# 21. SANITY CHECKS
# =============================================================================

print("\n" + "-" * 80)
print("MODEL COMPARISON SANITY CHECKS")
print("-" * 80)


print(
    f"Reduced residual SS: "
    f"{SSR_reduced:.10f}"
)

print(
    f"Full residual SS:    "
    f"{SSR_full:.10f}"
)

print(
    f"SS explained by interaction: "
    f"{SS_difference:.10f}"
)

print(
    f"Reduced residual df: "
    f"{df_reduced:.0f}"
)

print(
    f"Full residual df: "
    f"{df_full:.0f}"
)

print(
    f"Degrees of freedom difference: "
    f"{df_difference:.0f}"
)


# The full model should never have greater residual SS
# than the reduced model.

if SS_difference < -1e-8:

    raise ValueError(
        "\nInvalid model comparison: "
        "full model has greater residual SS "
        "than reduced model."
    )


# Interaction should have 24 df:
#
# (7 - 1) × (5 - 1) = 24

expected_interaction_df = (
    (len(DISEASES) - 1)
    * (len(POPULATIONS) - 1)
)


if df_difference != expected_interaction_df:

    raise ValueError(
        f"\nUnexpected interaction degrees "
        f"of freedom.\n"
        f"Expected: {expected_interaction_df}\n"
        f"Observed: {df_difference}"
    )


if SSR_full <= 0:

    raise ValueError(
        "\nFull-model residual SS is not positive."
    )


if not (
    0 <= partial_eta_squared <= 1
):

    raise ValueError(
        "\nPartial eta-squared is outside "
        "the valid range [0, 1]."
    )


print(
    "\nSanity checks: PASSED"
)


# =============================================================================
# 22. FINAL RESULT
# =============================================================================

print("\n" + "=" * 80)
print("FINAL INTERACTION RESULT")
print("=" * 80)


print(
    f"\nDisease × Superpopulation interaction:"
)

print(
    f"  F = {F_interaction:.6f}"
)

print(
    f"  df = "
    f"{int(df_difference)}, "
    f"{int(df_full)}"
)

print(
    f"  p = {p_interaction:.6e}"
)

print(
    f"  partial η² = "
    f"{partial_eta_squared:.6f}"
)


# =============================================================================
# 23. SAVE INTERACTION RESULT
# =============================================================================

interaction_result = pd.DataFrame({

    "Effect": [
        "Disease × Superpopulation"
    ],

    "Reduced_Model": [
        "PRS_Z ~ C(Disease) + C(IID)"
    ],

    "Full_Model": [
        "PRS_Z ~ C(Disease) + C(IID) + "
        "C(Disease):C(Superpopulation)"
    ],

    "SS_Reduced": [
        SSR_reduced
    ],

    "SS_Full": [
        SSR_full
    ],

    "SS_Interaction": [
        SS_difference
    ],

    "df_Interaction": [
        int(df_difference)
    ],

    "df_Error": [
        int(df_full)
    ],

    "F": [
        F_interaction
    ],

    "P_value": [
        p_interaction
    ],

    "Partial_Eta_Squared": [
        partial_eta_squared
    ]
})


interaction_file = os.path.join(
    OUT_DIR,
    "disease_population_interaction_result.tsv"
)


interaction_result.to_csv(
    interaction_file,
    sep="\t",
    index=False,
    float_format="%.10f"
)


# =============================================================================
# 24. SAVE MODEL SUMMARY
# =============================================================================

model_summary_file = os.path.join(
    OUT_DIR,
    "disease_population_interaction_model_summary.txt"
)


with open(
    model_summary_file,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "SEVEN-DISEASE × SUPERPOPULATION "
        "INTERACTION ANALYSIS\n"
    )

    f.write(
        "=" * 80
        + "\n\n"
    )

    f.write(
        "Reduced model:\n"
    )

    f.write(
        "PRS_Z ~ C(Disease) + C(IID)\n\n"
    )

    f.write(
        "Full model:\n"
    )

    f.write(
        "PRS_Z ~ C(Disease) + C(IID) "
        "+ C(Disease):C(Superpopulation)\n\n"
    )

    f.write(
        f"Reduced residual SS: "
        f"{SSR_reduced:.10f}\n"
    )

    f.write(
        f"Full residual SS: "
        f"{SSR_full:.10f}\n"
    )

    f.write(
        f"Interaction SS: "
        f"{SS_difference:.10f}\n"
    )

    f.write(
        f"Interaction df: "
        f"{int(df_difference)}\n"
    )

    f.write(
        f"Error df: "
        f"{int(df_full)}\n"
    )

    f.write(
        f"F statistic: "
        f"{F_interaction:.10f}\n"
    )

    f.write(
        f"P value: "
        f"{p_interaction:.10e}\n"
    )

    f.write(
        f"Partial eta squared: "
        f"{partial_eta_squared:.10f}\n"
    )


# =============================================================================
# 25. INTERPRETATION
# =============================================================================

print("\n" + "=" * 80)
print("INTERPRETATION")
print("=" * 80)


if p_interaction < 0.05:

    print(
        "\nA statistically significant Disease × "
        "Superpopulation interaction was detected."
    )

    print(
        "This indicates that the pattern of standardized "
        "PRS differences across superpopulations varies "
        "among diseases."
    )

else:

    print(
        "\nThe Disease × Superpopulation interaction "
        "was not statistically significant."
    )

    print(
        "This indicates that the population-associated "
        "pattern did not differ significantly among "
        "the seven diseases under this model."
    )


# =============================================================================
# 26. OUTPUT FILES
# =============================================================================

print("\n" + "=" * 80)
print("OUTPUT FILES")
print("=" * 80)


print(
    "\nLong-format dataset:"
)

print(
    long_file
)


print(
    "\nInteraction result:"
)

print(
    interaction_file
)


print(
    "\nModel summary:"
)

print(
    model_summary_file
)


# =============================================================================
# 27. FINAL STATUS
# =============================================================================

print("\n" + "=" * 80)
print("STATUS: INTERACTION ANALYSIS COMPLETE")
print("=" * 80)