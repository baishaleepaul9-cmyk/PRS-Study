from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# CONSOLIDATE ALL 7 DISEASES
# PHASE 1 - STEP 1
#
# IMPORTANT:
# Raw PRS values are NOT standardized or transformed.
# ============================================================


ROOT = Path(r"C:\PRS_Study")

RESULTS = ROOT / "results"

OUT = RESULTS / "combined_analysis"

OUT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# DISEASE DEFINITIONS
# ============================================================

DISEASES = {

    "Alzheimers": {
        "name": "Alzheimer's disease",
        "folder": "Alzheimer",
        "filename": "Alzheimer_genomewide_PRS.tsv",
        "prs_column": "ALZHEIMERS_PRS"
    },

    "Schizophrenia": {
        "name": "Schizophrenia",
        "folder": "Schizophrenia",
        "filename": "Schizophrenia_genomewide_PRS.tsv",
        "prs_column": "PRS"
    },

    "Multiple_Sclerosis": {
        "name": "Multiple sclerosis",
        "folder": "Multiple_Sclerosis",
        "filename": "MS_genomewide_PRS.tsv",
        "prs_column": "PRS"
    },

    "MDD": {
        "name": "Major depressive disorder",
        "folder": "MDD",
        "filename": "MDD_genomewide_PRS.tsv",
        "prs_column": "PRS"
    },

    "Parkinsons": {
        "name": "Parkinson's disease",
        "folder": "Parkinsons",
        "filename": None,
        "prs_column": "PRS"
    },

    "CAD": {
        "name": "Coronary artery disease",
        "folder": "CAD",
        "filename": "CAD_genomewide_PRS.tsv",
        "prs_column": "PRS"
    },

    "T2D": {
        "name": "Type 2 diabetes",
        "folder": "T2D",
        "filename": "T2D_genomewide_PRS.tsv",
        "prs_column": "T2D_PRS"
    }
}


# ============================================================
# FUNCTION: FIND PARKINSON FILE
# ============================================================

def find_parkinson_file(folder):

    candidates = [
        folder / "Parkinson_genomewide_PRS.tsv",
        folder / "Parkinsons_genomewide_PRS.tsv",
        folder / "Parkinson_genomewide_PRS_with_risk_groups.tsv",
        folder / "Parkinsons_genomewide_PRS_with_risk_groups.tsv"
    ]

    for f in candidates:

        if f.exists():

            return f


    # Search only genome-wide PRS files
    matches = list(
        folder.glob("*genomewide*PRS*.tsv")
    )

    if len(matches) == 1:

        return matches[0]

    if len(matches) > 1:

        print("\nPossible Parkinson files:")

        for f in matches:
            print(f)

        raise RuntimeError(
            "Multiple Parkinson genome-wide PRS files found. "
            "Please specify the correct file."
        )


    return None


# ============================================================
# STORAGE
# ============================================================

all_data = []

summary_rows = []

sample_sets = {}


# ============================================================
# START
# ============================================================

print("=" * 75)

print(
    "CONSOLIDATING ALL 7 DISEASE PRS RESULTS"
)

print("=" * 75)

print(
    "\nIMPORTANT: Raw PRS values are preserved exactly."
)


# ============================================================
# PROCESS EACH DISEASE
# ============================================================

for disease_key, info in DISEASES.items():

    print("\n" + "-" * 75)

    print(
        info["name"]
    )

    print("-" * 75)


    folder = (
        RESULTS
        / info["folder"]
    )


    # --------------------------------------------------------
    # Determine file
    # --------------------------------------------------------

    if disease_key == "Parkinsons":

        prs_path = find_parkinson_file(
            folder
        )

        if prs_path is None:

            print(
                "ERROR: Could not find Parkinson genome-wide PRS file."
            )

            continue

    else:

        prs_path = (
            folder
            / info["filename"]
        )


    print(
        f"File: {prs_path}"
    )


    if not prs_path.exists():

        print(
            "ERROR: File not found."
        )

        continue


    # --------------------------------------------------------
    # Read file
    # --------------------------------------------------------

    df = pd.read_csv(
        prs_path,
        sep="\t"
    )


    print(
        f"Rows loaded: {len(df):,}"
    )

    print(
        f"Columns: {df.columns.tolist()}"
    )


    # --------------------------------------------------------
    # Identify sample ID
    # --------------------------------------------------------

    id_candidates = [
        "IID",
        "#IID",
        "ID",
        "Sample",
        "sample",
        "sample_id"
    ]

    id_col = None

    for col in id_candidates:

        if col in df.columns:

            id_col = col

            break


    if id_col is None:

        raise ValueError(
            f"Could not identify sample ID column for "
            f"{info['name']}.\n"
            f"Columns: {df.columns.tolist()}"
        )


    # --------------------------------------------------------
    # Identify PRS
    # --------------------------------------------------------

    requested_prs = info["prs_column"]


    if requested_prs not in df.columns:

        raise ValueError(
            f"Expected PRS column '{requested_prs}' "
            f"not found for {info['name']}.\n"
            f"Available columns: {df.columns.tolist()}"
        )


    prs_col = requested_prs


    print(
        f"ID column: {id_col}"
    )

    print(
        f"PRS column: {prs_col}"
    )


    # --------------------------------------------------------
    # Create clean table
    # --------------------------------------------------------

    clean = df[
        [
            id_col,
            prs_col
        ]
    ].copy()


    clean.columns = [
        "IID",
        "PRS"
    ]


    clean["IID"] = (
        clean["IID"]
        .astype(str)
        .str.strip()
    )


    clean["PRS"] = pd.to_numeric(
        clean["PRS"],
        errors="coerce"
    )


    # --------------------------------------------------------
    # Quality checks
    # --------------------------------------------------------

    n = len(clean)

    unique_iids = (
        clean["IID"]
        .nunique()
    )

    missing_prs = (
        clean["PRS"]
        .isna()
        .sum()
    )

    duplicate_iids = (
        clean["IID"]
        .duplicated()
        .sum()
    )


    print(
        f"N: {n:,}"
    )

    print(
        f"Unique individuals: {unique_iids:,}"
    )

    print(
        f"Missing PRS: {missing_prs:,}"
    )

    print(
        f"Duplicate IDs: {duplicate_iids:,}"
    )


    if n != 2504:

        print(
            f"WARNING: Expected 2504 individuals, "
            f"found {n}."
        )


    if duplicate_iids > 0:

        raise ValueError(
            f"Duplicate sample IDs found for "
            f"{info['name']}."
        )


    if missing_prs > 0:

        raise ValueError(
            f"Missing PRS values found for "
            f"{info['name']}."
        )


    # --------------------------------------------------------
    # Store sample IDs
    # --------------------------------------------------------

    sample_sets[disease_key] = set(
        clean["IID"]
    )


    # --------------------------------------------------------
    # Add disease information
    # --------------------------------------------------------

    clean["Disease"] = info["name"]

    clean["Disease_Key"] = disease_key


    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    valid = clean["PRS"].dropna()


    summary_rows.append({

        "Disease": info["name"],

        "Disease_Key": disease_key,

        "N": len(valid),

        "Mean": valid.mean(),

        "SD": valid.std(
            ddof=1
        ),

        "Median": valid.median(),

        "Min": valid.min(),

        "Q1": valid.quantile(
            0.25
        ),

        "Q3": valid.quantile(
            0.75
        ),

        "Max": valid.max(),

        "P20": valid.quantile(
            0.20
        ),

        "P80": valid.quantile(
            0.80
        ),

        "Missing_PRS": missing_prs,

        "Duplicate_IIDs": duplicate_iids

    })


    all_data.append(
        clean[
            [
                "IID",
                "Disease",
                "Disease_Key",
                "PRS"
            ]
        ]
    )


# ============================================================
# VERIFY ALL 7 WERE LOADED
# ============================================================

print("\n" + "=" * 75)

print(
    "DISEASE LOADING CHECK"
)

print("=" * 75)


loaded_keys = [
    x["Disease_Key"]
    for x in summary_rows
]


print(
    f"Diseases successfully loaded: "
    f"{len(loaded_keys)}/7"
)


for key in loaded_keys:

    print(
        f"  OK: {key}"
    )


missing_keys = [
    key
    for key in DISEASES
    if key not in loaded_keys
]


if missing_keys:

    print(
        "\nMissing diseases:"
    )

    for key in missing_keys:

        print(
            f"  {key}"
        )


    raise RuntimeError(
        "Not all seven diseases were loaded. "
        "No final consolidated files will be produced."
    )


# ============================================================
# COMBINE LONG FORMAT
# ============================================================

master_long = pd.concat(
    all_data,
    ignore_index=True
)


# ============================================================
# SAVE LONG FORMAT
# ============================================================

long_file = (
    OUT
    / "all_7_diseases_raw_PRS_long.tsv"
)


master_long.to_csv(
    long_file,
    sep="\t",
    index=False
)


# ============================================================
# CREATE WIDE FORMAT
# ============================================================

master_wide = (
    master_long
    .pivot(
        index="IID",
        columns="Disease_Key",
        values="PRS"
    )
    .reset_index()
)


master_wide.columns.name = None


wide_file = (
    OUT
    / "all_7_diseases_raw_PRS_wide.tsv"
)


master_wide.to_csv(
    wide_file,
    sep="\t",
    index=False
)


# ============================================================
# DISEASE SUMMARY
# ============================================================

summary = pd.DataFrame(
    summary_rows
)


summary_file = (
    OUT
    / "all_7_diseases_raw_PRS_summary.tsv"
)


summary.to_csv(
    summary_file,
    sep="\t",
    index=False
)


# ============================================================
# SAMPLE OVERLAP
# ============================================================

print("\n" + "=" * 75)

print(
    "SAMPLE OVERLAP CHECK"
)

print("=" * 75)


common_samples = set.intersection(
    *sample_sets.values()
)


union_samples = set.union(
    *sample_sets.values()
)


print(
    f"Unique individuals across all diseases: "
    f"{len(union_samples):,}"
)


print(
    f"Individuals present in ALL 7 diseases: "
    f"{len(common_samples):,}"
)


# ============================================================
# WIDE TABLE CHECK
# ============================================================

print("\n" + "=" * 75)

print(
    "MASTER TABLE CHECK"
)

print("=" * 75)


print(
    f"Master long rows: "
    f"{len(master_long):,}"
)


print(
    f"Master wide individuals: "
    f"{len(master_wide):,}"
)


print(
    f"Master wide columns: "
    f"{master_wide.columns.tolist()}"
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 75)

print(
    "DISEASE-LEVEL RAW PRS SUMMARY"
)

print("=" * 75)


print(
    summary.to_string(
        index=False
    )
)


# ============================================================
# FINAL FILES
# ============================================================

print("\n" + "=" * 75)

print(
    "CONSOLIDATION COMPLETE"
)

print("=" * 75)


print("\nCreated:")

print(
    f"\n1. {long_file}"
)

print(
    f"\n2. {wide_file}"
)

print(
    f"\n3. {summary_file}"
)


print(
    "\nNo PRS values were standardized, "
    "rescaled, inverted, or otherwise transformed."
)


print("\nDone.")
