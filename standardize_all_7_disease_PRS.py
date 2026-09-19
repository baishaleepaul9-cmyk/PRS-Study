import os
import pandas as pd
import numpy as np

ROOT = r"C:\PRS_Study\results"
OUT_DIR = os.path.join(ROOT, "combined_analysis")
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# INPUT FILES
# ============================================================

DISEASE_FILES = {
    "Alzheimers":
        r"Alzheimer\Alzheimer_genomewide_PRS_with_risk_groups.tsv",

    "Schizophrenia":
        r"Schizophrenia\Schizophrenia_genomewide_PRS_with_risk_groups.tsv",

    "Multiple_Sclerosis":
        r"Multiple_Sclerosis\MS_genomewide_PRS.tsv",

    "MDD":
        r"MDD\MDD_genomewide_PRS_with_risk_groups.tsv",

    "Parkinsons":
        r"Parkinsons\Parkinsons_genomewide_PRS_with_risk_groups.tsv",

    "CAD":
        r"CAD\CAD_genomewide_PRS_with_risk_groups.tsv",

    "T2D":
        r"T2D\T2D_genomewide_PRS_with_risk_groups.tsv",
}

# ============================================================
# OUTPUT
# ============================================================

OUTPUT = os.path.join(
    OUT_DIR,
    "all_7_diseases_standardized_PRS.tsv"
)

SUMMARY_OUTPUT = os.path.join(
    OUT_DIR,
    "all_7_diseases_standardization_parameters.tsv"
)

# ============================================================
# STORAGE
# ============================================================

standardized = None
parameters = []

print("=" * 80)
print("7-DISEASE PRS STANDARDIZATION")
print("=" * 80)

# ============================================================
# PROCESS EACH DISEASE
# ============================================================

for disease, relative_path in DISEASE_FILES.items():

    path = os.path.join(ROOT, relative_path)

    print(f"\nProcessing: {disease}")
    print(f"File: {path}")

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Required PRS file not found:\n{path}"
        )

    df = pd.read_csv(path, sep="\t")

    # --------------------------------------------------------
    # Identify genome-wide PRS column
    # --------------------------------------------------------

    if "PRS" in df.columns:

        prs_col = "PRS"

    else:

        candidates = [
            col for col in df.columns
            if col.upper().endswith("_PRS")
            and not col.upper().startswith("CHR")
            and "RISK" not in col.upper()
        ]

        if len(candidates) != 1:
            raise ValueError(
                f"Could not uniquely identify PRS column for "
                f"{disease}.\nCandidates: {candidates}\n"
                f"Columns: {list(df.columns)}"
            )

        prs_col = candidates[0]

    # --------------------------------------------------------
    # Identify individual ID
    # --------------------------------------------------------

    if "#IID" in df.columns:
        id_col = "#IID"

    elif "IID" in df.columns:
        id_col = "IID"

    else:
        raise ValueError(
            f"Could not identify individual ID column for {disease}."
        )

    # --------------------------------------------------------
    # Extract
    # --------------------------------------------------------

    temp = df[[id_col, prs_col]].copy()

    temp.columns = ["IID", "PRS"]

    temp["PRS"] = pd.to_numeric(
        temp["PRS"],
        errors="coerce"
    )

    # Check sample count
    if len(temp) != 2504:
        raise ValueError(
            f"{disease}: expected 2504 individuals, "
            f"found {len(temp)}."
        )

    # Check missing
    missing = int(temp["PRS"].isna().sum())

    if missing != 0:
        raise ValueError(
            f"{disease}: {missing} missing PRS values."
        )

    # Check duplicate IDs
    duplicates = int(temp["IID"].duplicated().sum())

    if duplicates != 0:
        raise ValueError(
            f"{disease}: {duplicates} duplicate IID values."
        )

    # --------------------------------------------------------
    # Calculate disease-specific mean and SD
    # --------------------------------------------------------

    mean = temp["PRS"].mean()
    sd = temp["PRS"].std(ddof=1)

    # --------------------------------------------------------
    # Standardize
    # --------------------------------------------------------

    temp[f"{disease}_Z"] = (
        (temp["PRS"] - mean) / sd
    )

    # --------------------------------------------------------
    # Verify standardized distribution
    # --------------------------------------------------------

    z_mean = temp[f"{disease}_Z"].mean()
    z_sd = temp[f"{disease}_Z"].std(ddof=1)

    print(f"PRS column: {prs_col}")
    print(f"N: {len(temp)}")
    print(f"Mean: {mean:.10f}")
    print(f"SD: {sd:.10f}")
    print(f"Z-score mean: {z_mean:.10f}")
    print(f"Z-score SD: {z_sd:.10f}")

    # --------------------------------------------------------
    # Save parameters
    # --------------------------------------------------------

    parameters.append({
        "Disease": disease,
        "N": len(temp),
        "Original_Mean": mean,
        "Original_SD": sd,
        "Z_Mean": z_mean,
        "Z_SD": z_sd,
    })

    # --------------------------------------------------------
    # Merge into combined table
    # --------------------------------------------------------

    z_data = temp[["IID", f"{disease}_Z"]]

    if standardized is None:

        standardized = z_data

    else:

        standardized = standardized.merge(
            z_data,
            on="IID",
            how="inner",
            validate="one_to_one"
        )

# ============================================================
# FINAL VERIFICATION
# ============================================================

print("\n")
print("=" * 80)
print("FINAL VERIFICATION")
print("=" * 80)

print(f"Individuals: {len(standardized)}")
print(f"Columns: {len(standardized.columns)}")

expected_columns = 1 + 7

if len(standardized) != 2504:
    raise ValueError(
        f"Expected 2504 individuals, found {len(standardized)}."
    )

if len(standardized.columns) != expected_columns:
    raise ValueError(
        f"Expected {expected_columns} columns, "
        f"found {len(standardized.columns)}."
    )

if standardized.isna().sum().sum() != 0:
    raise ValueError(
        "Missing values detected in standardized PRS table."
    )

# ============================================================
# SAVE STANDARDIZED DATA
# ============================================================

standardized.to_csv(
    OUTPUT,
    sep="\t",
    index=False,
    float_format="%.10f"
)

parameters_df = pd.DataFrame(parameters)

parameters_df.to_csv(
    SUMMARY_OUTPUT,
    sep="\t",
    index=False,
    float_format="%.10f"
)

# ============================================================
# DISPLAY
# ============================================================

print("STATUS: 7 DISEASES SUCCESSFULLY STANDARDIZED")
print("STATUS: 2504 INDIVIDUALS RETAINED")
print("STATUS: 0 MISSING VALUES")

print("\nStandardization parameters:")
print(parameters_df.to_string(index=False))

print("\nOutput:")
print(OUTPUT)

print("\nParameters:")
print(SUMMARY_OUTPUT)

print("=" * 80)
