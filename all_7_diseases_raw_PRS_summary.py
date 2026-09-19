import os
import glob
import pandas as pd
import numpy as np

ROOT = r"C:\PRS_Study\results"

DISEASE_FILES = {
    "Alzheimers": r"Alzheimer\Alzheimer_genomewide_PRS_with_risk_groups.tsv",
    "Schizophrenia": r"Schizophrenia\Schizophrenia_genomewide_PRS_with_risk_groups.tsv",
    "Multiple_Sclerosis": None,  # Automatically located below
    "MDD": r"MDD\MDD_genomewide_PRS_with_risk_groups.tsv",
    "Parkinsons": r"Parkinsons\Parkinsons_genomewide_PRS_with_risk_groups.tsv",
    "CAD": r"CAD\CAD_genomewide_PRS_with_risk_groups.tsv",
    "T2D": r"T2D\T2D_genomewide_PRS_with_risk_groups.tsv",
}

OUT_DIR = os.path.join(ROOT, "combined_analysis")
os.makedirs(OUT_DIR, exist_ok=True)

OUTPUT = os.path.join(
    OUT_DIR,
    "all_7_diseases_raw_PRS_summary.tsv"
)

rows = []

print("=" * 80)
print("7-DISEASE RAW PRS SUMMARY")
print("=" * 80)

for disease, relative_path in DISEASE_FILES.items():

    # ---------------------------------------------------------
    # Locate input file
    # ---------------------------------------------------------

    if disease == "Multiple_Sclerosis":

        ms_dir = os.path.join(ROOT, "Multiple_Sclerosis")

        candidates = glob.glob(
            os.path.join(ms_dir, "*genomewide*PRS*.tsv")
        )

        # Exclude any summary/risk-group files if multiple exist
        candidates = [
            f for f in candidates
            if "summary" not in os.path.basename(f).lower()
        ]

        if not candidates:
            print("\nProcessing: Multiple_Sclerosis")
            print("ERROR: No genome-wide MS PRS file found.")
            continue

        # Prefer the plain genome-wide file
        plain = [
            f for f in candidates
            if "with_risk" not in os.path.basename(f).lower()
        ]

        path = plain[0] if plain else candidates[0]

    else:
        path = os.path.join(ROOT, relative_path)

    print(f"\nProcessing: {disease}")
    print(f"File: {path}")

    if not os.path.exists(path):
        print("WARNING: File not found.")
        continue

    # ---------------------------------------------------------
    # Load file
    # ---------------------------------------------------------

    df = pd.read_csv(path, sep="\t")

    # ---------------------------------------------------------
    # Identify genome-wide PRS column
    # ---------------------------------------------------------

    if "PRS" in df.columns:
        prs_col = "PRS"

    else:
        candidates = [
            col for col in df.columns
            if col.upper().endswith("_PRS")
            and not col.upper().startswith("CHR")
        ]

        if len(candidates) != 1:
            raise ValueError(
                f"Could not uniquely identify genome-wide PRS "
                f"column for {disease}.\n"
                f"Candidates: {candidates}\n"
                f"Columns: {list(df.columns)}"
            )

        prs_col = candidates[0]

    print(f"PRS column: {prs_col}")

    # ---------------------------------------------------------
    # Extract PRS
    # ---------------------------------------------------------

    x = pd.to_numeric(df[prs_col], errors="coerce")

    n_missing = int(x.isna().sum())

    x = x.dropna()

    # ---------------------------------------------------------
    # Calculate statistics
    # ---------------------------------------------------------

    p20 = x.quantile(0.20)
    p80 = x.quantile(0.80)

    low_count = int((x <= p20).sum())
    high_count = int((x >= p80).sum())
    intermediate_count = len(x) - low_count - high_count

    row = {
        "Disease": disease,
        "N": len(x),
        "Missing": n_missing,
        "Mean": x.mean(),
        "SD": x.std(ddof=1),
        "Median": x.median(),
        "Q1": x.quantile(0.25),
        "Q3": x.quantile(0.75),
        "Min": x.min(),
        "Max": x.max(),
        "P20": p20,
        "P80": p80,
        "Low": low_count,
        "Intermediate": intermediate_count,
        "High": high_count,
    }

    rows.append(row)

# -------------------------------------------------------------
# Create final consolidated table
# -------------------------------------------------------------

summary = pd.DataFrame(rows)

order = [
    "Alzheimers",
    "Schizophrenia",
    "Multiple_Sclerosis",
    "MDD",
    "Parkinsons",
    "CAD",
    "T2D",
]

summary["Disease"] = pd.Categorical(
    summary["Disease"],
    categories=order,
    ordered=True
)

summary = summary.sort_values("Disease")

# -------------------------------------------------------------
# Save
# -------------------------------------------------------------

summary.to_csv(
    OUTPUT,
    sep="\t",
    index=False,
    float_format="%.10f"
)

# -------------------------------------------------------------
# Display
# -------------------------------------------------------------

print("\n")
print("=" * 80)
print("FINAL 7-DISEASE RAW PRS SUMMARY")
print("=" * 80)

print(summary.to_string(index=False))

print("\n")
print("=" * 80)
print("VERIFICATION")
print("=" * 80)

print(f"Diseases included: {len(summary)}")
print(f"Output file: {OUTPUT}")

if len(summary) == 7:
    print("STATUS: ALL 7 DISEASES INCLUDED")
else:
    print("WARNING: NOT ALL 7 DISEASES WERE INCLUDED")

print("=" * 80)
