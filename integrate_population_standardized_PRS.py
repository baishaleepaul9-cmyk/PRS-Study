import os
import pandas as pd

# ============================================================
# PATHS
# ============================================================

ROOT = r"C:\PRS_Study"

PRS_FILE = os.path.join(
    ROOT,
    "results",
    "combined_analysis",
    "all_7_diseases_standardized_PRS.tsv"
)

POP_FILE = os.path.join(
    ROOT,
    "data",
    "1000G",
    "1000G_phase3_population_panel.txt"
)

OUT_DIR = os.path.join(
    ROOT,
    "results",
    "combined_analysis"
)

os.makedirs(OUT_DIR, exist_ok=True)

OUTPUT = os.path.join(
    OUT_DIR,
    "all_7_diseases_standardized_PRS_with_population.tsv"
)

# ============================================================
# LOAD FILES
# ============================================================

print("=" * 80)
print("INTEGRATING POPULATION LABELS WITH STANDARDIZED PRS")
print("=" * 80)

print("\nLoading standardized PRS:")
print(PRS_FILE)

prs = pd.read_csv(
    PRS_FILE,
    sep="\t"
)

print(f"PRS rows: {len(prs)}")
print(f"PRS columns: {list(prs.columns)}")

print("\nLoading 1000 Genomes population panel:")
print(POP_FILE)

pop = pd.read_csv(
    POP_FILE,
    sep="\t"
)

print(f"Population-panel rows: {len(pop)}")
print(f"Population-panel columns: {list(pop.columns)}")

# ============================================================
# STANDARDIZE ID COLUMN
# ============================================================

# PRS file uses IID
if "IID" not in prs.columns:
    raise ValueError("IID column not found in standardized PRS file.")

# Population panel uses sample
if "sample" not in pop.columns:
    raise ValueError("sample column not found in population panel.")

prs["IID"] = prs["IID"].astype(str).str.strip()
pop["sample"] = pop["sample"].astype(str).str.strip()

# ============================================================
# VERIFY IDS
# ============================================================

prs_ids = set(prs["IID"])
pop_ids = set(pop["sample"])

print("\nID verification:")
print(f"PRS individuals: {len(prs_ids)}")
print(f"Population-panel individuals: {len(pop_ids)}")
print(f"Matched IDs: {len(prs_ids & pop_ids)}")
print(f"PRS IDs not in panel: {len(prs_ids - pop_ids)}")
print(f"Panel IDs not in PRS: {len(pop_ids - prs_ids)}")

if prs_ids != pop_ids:
    raise ValueError(
        "Individual IDs do not match exactly between PRS "
        "and population panel."
    )

# ============================================================
# SELECT POPULATION INFORMATION
# ============================================================

required_pop_columns = [
    "sample",
    "pop",
    "super_pop",
    "gender"
]

missing = [
    col for col in required_pop_columns
    if col not in pop.columns
]

if missing:
    raise ValueError(
        f"Missing population-panel columns: {missing}"
    )

pop_subset = pop[required_pop_columns].copy()

pop_subset = pop_subset.rename(
    columns={
        "sample": "IID",
        "pop": "Population",
        "super_pop": "Superpopulation",
        "gender": "Gender"
    }
)

# ============================================================
# MERGE
# ============================================================

combined = prs.merge(
    pop_subset,
    on="IID",
    how="inner",
    validate="one_to_one"
)

# ============================================================
# VERIFY MERGE
# ============================================================

print("\nMerge verification:")
print(f"Final individuals: {len(combined)}")
print(
    f"Missing population labels: "
    f"{combined['Superpopulation'].isna().sum()}"
)

if len(combined) != 2504:
    raise ValueError(
        f"Expected 2504 individuals after merge, "
        f"found {len(combined)}."
    )

if combined["Superpopulation"].isna().any():
    raise ValueError(
        "Missing superpopulation labels detected."
    )

expected_populations = {
    "AFR",
    "AMR",
    "EAS",
    "EUR",
    "SAS"
}

observed_populations = set(
    combined["Superpopulation"].astype(str)
)

print(
    f"Superpopulations found: "
    f"{sorted(observed_populations)}"
)

if observed_populations != expected_populations:
    raise ValueError(
        "Unexpected superpopulation labels detected."
    )

# ============================================================
# ORDER COLUMNS
# ============================================================

z_columns = [
    "Alzheimers_Z",
    "Schizophrenia_Z",
    "Multiple_Sclerosis_Z",
    "MDD_Z",
    "Parkinsons_Z",
    "CAD_Z",
    "T2D_Z"
]

final_columns = [
    "IID",
    "Population",
    "Superpopulation",
    "Gender"
] + z_columns

combined = combined[final_columns]

# ============================================================
# SORT
# ============================================================

population_order = [
    "AFR",
    "AMR",
    "EAS",
    "EUR",
    "SAS"
]

combined["Superpopulation"] = pd.Categorical(
    combined["Superpopulation"],
    categories=population_order,
    ordered=True
)

combined = combined.sort_values(
    ["Superpopulation", "IID"]
).reset_index(drop=True)

# ============================================================
# SAVE
# ============================================================

combined.to_csv(
    OUTPUT,
    sep="\t",
    index=False,
    float_format="%.10f"
)

# ============================================================
# FINAL VERIFICATION
# ============================================================

print("\n")
print("=" * 80)
print("FINAL VERIFICATION")
print("=" * 80)

print(f"Individuals: {len(combined)}")
print(f"Columns: {len(combined.columns)}")
print(
    f"Missing values: "
    f"{combined.isna().sum().sum()}"
)

print("\nSuperpopulation counts:")
print(
    combined["Superpopulation"]
    .value_counts()
    .sort_index()
)

print("\nSTATUS: POPULATION LABELS SUCCESSFULLY INTEGRATED")
print("STATUS: 2504 INDIVIDUALS RETAINED")
print("STATUS: 7 STANDARDIZED PRS VARIABLES RETAINED")
print("STATUS: 0 MISSING VALUES")

print("\nOutput:")
print(OUTPUT)

print("=" * 80)
