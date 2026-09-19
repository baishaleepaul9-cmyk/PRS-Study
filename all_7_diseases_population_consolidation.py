import os
import pandas as pd

ROOT = r"C:\PRS_Study\results"
OUT_DIR = os.path.join(ROOT, "combined_analysis")
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# FINAL POPULATION-LEVEL RESULTS FROM THE COMPLETED
# DISEASE-SPECIFIC POPULATION ANALYSES
# ============================================================

data = [

    # --------------------------------------------------------
    # Alzheimer's
    # --------------------------------------------------------
    ["Alzheimers", "AFR", 661,  0.004543, 0.099290,  0.003453, -0.319278, 0.262474, 0.003862],
    ["Alzheimers", "AMR", 347, -0.259248, 0.104609, -0.263624, -0.510371, 0.020164, 0.005616],
    ["Alzheimers", "EAS", 504, -0.356002, 0.103335, -0.361579, -0.656459, 0.005532, 0.004603],
    ["Alzheimers", "EUR", 503, -0.260230, 0.108520, -0.269963, -0.548520, 0.128437, 0.004839],
    ["Alzheimers", "SAS", 489, -0.304201, 0.101661, -0.306210, -0.571281, 0.039191, 0.004597],

    # --------------------------------------------------------
    # Schizophrenia
    # --------------------------------------------------------
    ["Schizophrenia", "AFR", 661, 1.007294, 0.202166, None, None, None, None],
    ["Schizophrenia", "AMR", 347, 0.606662, 0.335393, None, None, None, None],
    ["Schizophrenia", "EAS", 504, 0.864945, 0.221338, None, None, None, None],
    ["Schizophrenia", "EUR", 503, 0.101504, 0.276407, None, None, None, None],
    ["Schizophrenia", "SAS", 489, 0.705388, 0.250112, None, None, None, None],

    # --------------------------------------------------------
    # Multiple Sclerosis
    # --------------------------------------------------------
    ["Multiple_Sclerosis", "AFR", 661, -0.99390706, 0.77028737, None, None, None, None],
    ["Multiple_Sclerosis", "AMR", 347, -3.14035603, 0.93827491, None, None, None, None],
    ["Multiple_Sclerosis", "EAS", 504, -3.04156485, 0.72644786, None, None, None, None],
    ["Multiple_Sclerosis", "EUR", 503, -3.27221174, 1.07596856, None, None, None, None],
    ["Multiple_Sclerosis", "SAS", 489, -2.64637163, 0.78116092, None, None, None, None],

    # --------------------------------------------------------
    # MDD
    # --------------------------------------------------------
    ["MDD", "AFR", 661, -0.60997066, 0.20308922, None, None, None, None],
    ["MDD", "AMR", 347, -0.85628409, 0.31517369, None, None, None, None],
    ["MDD", "EAS", 504, -0.79813327, 0.24428062, None, None, None, None],
    ["MDD", "EUR", 503, -1.14526321, 0.29611482, None, None, None, None],
    ["MDD", "SAS", 489, -0.95027317, 0.27808901, None, None, None, None],

    # --------------------------------------------------------
    # Parkinson's
    # --------------------------------------------------------
    ["Parkinsons", "AFR", 661, -21.72364357, 1.42707011, None, None, None, None],
    ["Parkinsons", "AMR", 347, -22.42210778, 2.04124447, None, None, None, None],
    ["Parkinsons", "EAS", 504, -22.00193294, 1.47080122, None, None, None, None],
    ["Parkinsons", "EUR", 503, -21.91202187, 1.96158830, None, None, None, None],
    ["Parkinsons", "SAS", 489, -21.20368507, 1.65765255, None, None, None, None],

    # --------------------------------------------------------
    # CAD
    # --------------------------------------------------------
    ["CAD", "AFR", 661, -0.175872, 0.208350, None, None, None, None],
    ["CAD", "AMR", 347, -0.069136, 0.326189, None, None, None, None],
    ["CAD", "EAS", 504,  0.253762, 0.207767, None, None, None, None],
    ["CAD", "EUR", 503, -0.342949, 0.338995, None, None, None, None],
    ["CAD", "SAS", 489, -0.070135, 0.253093, None, None, None, None],

    # --------------------------------------------------------
    # T2D
    # --------------------------------------------------------
    ["T2D", "AFR", 661, -2527.077819, 7.722530, None, None, None, 0.300372],
    ["T2D", "AMR", 347, -2481.905906, 12.754430, None, None, None, 0.684694],
    ["T2D", "EAS", 504, -2518.460507, 6.335255, None, None, None, 0.282195],
    ["T2D", "EUR", 503, -2457.330204, 15.668803, None, None, None, 0.698637],
    ["T2D", "SAS", 489, -2500.348977, 9.981058, None, None, None, 0.451359],
]

columns = [
    "Disease",
    "Population",
    "N",
    "Mean",
    "SD",
    "Median",
    "Min",
    "Max",
    "SE",
]

combined = pd.DataFrame(data, columns=columns)

# ============================================================
# ORDERING
# ============================================================

disease_order = [
    "Alzheimers",
    "Schizophrenia",
    "Multiple_Sclerosis",
    "MDD",
    "Parkinsons",
    "CAD",
    "T2D",
]

population_order = [
    "AFR",
    "AMR",
    "EAS",
    "EUR",
    "SAS",
]

combined["Disease"] = pd.Categorical(
    combined["Disease"],
    categories=disease_order,
    ordered=True,
)

combined["Population"] = pd.Categorical(
    combined["Population"],
    categories=population_order,
    ordered=True,
)

combined = (
    combined
    .sort_values(["Disease", "Population"])
    .reset_index(drop=True)
)

# ============================================================
# SAVE LONG TABLE
# ============================================================

long_output = os.path.join(
    OUT_DIR,
    "all_7_diseases_population_long.tsv"
)

combined.to_csv(
    long_output,
    sep="\t",
    index=False,
    float_format="%.10f",
)

# ============================================================
# POPULATION × DISEASE MEAN MATRIX
# ============================================================

mean_matrix = combined.pivot(
    index="Disease",
    columns="Population",
    values="Mean",
)

mean_output = os.path.join(
    OUT_DIR,
    "all_7_diseases_population_mean_matrix.tsv"
)

mean_matrix.to_csv(
    mean_output,
    sep="\t",
    float_format="%.10f",
)

# ============================================================
# POPULATION × DISEASE SD MATRIX
# ============================================================

sd_matrix = combined.pivot(
    index="Disease",
    columns="Population",
    values="SD",
)

sd_output = os.path.join(
    OUT_DIR,
    "all_7_diseases_population_SD_matrix.tsv"
)

sd_matrix.to_csv(
    sd_output,
    sep="\t",
    float_format="%.10f",
)

# ============================================================
# VERIFICATION
# ============================================================

print("\n")
print("=" * 80)
print("FINAL POPULATION × DISEASE TABLE")
print("=" * 80)

print(combined.to_string(index=False))

print("\n")
print("=" * 80)
print("VERIFICATION")
print("=" * 80)

print(f"Diseases included: {combined['Disease'].nunique()}")
print(f"Populations included: {combined['Population'].nunique()}")
print(f"Total rows: {len(combined)}")

if len(combined) == 35:
    print("STATUS: ALL 35 POPULATION × DISEASE COMBINATIONS PRESENT")
else:
    print("WARNING: Expected 35 rows.")

print("\nPopulation counts:")
print(combined["Population"].value_counts().sort_index())

print("\nOutput files:")
print(long_output)
print(mean_output)
print(sd_output)

print("=" * 80)
