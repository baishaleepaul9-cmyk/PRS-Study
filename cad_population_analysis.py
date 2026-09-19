import pandas as pd
from pathlib import Path
from scipy.stats import kruskal

# ============================================================
# PATHS
# ============================================================

BASE = Path(r"C:\PRS_Study")
CAD_DIR = BASE / "results" / "CAD"
PANEL_FILE = BASE / "data" / "1000G" / "1000G_phase3_population_panel.txt"

PRS_FILE = CAD_DIR / "CAD_genomewide_PRS_with_risk_groups.tsv"

OUTPUT_LABELLED = CAD_DIR / "CAD_PRS_with_population_labels.tsv"
OUTPUT_STATS = CAD_DIR / "CAD_population_PRS_by_superpopulation.tsv"
OUTPUT_KW = CAD_DIR / "CAD_population_Kruskal_Wallis.tsv"


# ============================================================
# 1. LOAD CAD PRS
# ============================================================

print("=" * 90)
print("CAD POPULATION / ANCESTRY ANALYSIS")
print("=" * 90)

print("\nLoading CAD genome-wide PRS...")

prs = pd.read_csv(PRS_FILE, sep="\t")

print(f"PRS individuals: {len(prs)}")
print(f"PRS columns: {list(prs.columns)}")


# ============================================================
# 2. LOAD 1000 GENOMES POPULATION PANEL
# ============================================================

print("\nLoading 1000 Genomes population panel...")

panel = pd.read_csv(PANEL_FILE, sep="\t")

print(f"Panel individuals: {len(panel)}")
print(f"Panel columns: {list(panel.columns)}")


# ============================================================
# 3. VERIFY SAMPLE ID MATCHING
# ============================================================

prs_ids = set(prs["IID"])
panel_ids = set(panel["sample"])

matched = prs_ids & panel_ids
missing_population = prs_ids - panel_ids
extra_panel = panel_ids - prs_ids

print("\n" + "-" * 90)
print("SAMPLE ID VERIFICATION")
print("-" * 90)

print(f"PRS individuals: {len(prs_ids)}")
print(f"Panel individuals: {len(panel_ids)}")
print(f"Matched IDs: {len(matched)}")
print(f"Missing population labels: {len(missing_population)}")
print(f"Extra panel IDs: {len(extra_panel)}")


if len(missing_population) > 0:
    print("\nWARNING: Some PRS individuals have no population label.")

if len(extra_panel) > 0:
    print("\nWARNING: Panel contains individuals not present in PRS dataset.")


# ============================================================
# 4. MERGE PRS WITH POPULATION INFORMATION
# ============================================================

print("\nMerging PRS with population information...")

df = prs.merge(
    panel[["sample", "pop", "super_pop", "gender"]],
    left_on="IID",
    right_on="sample",
    how="inner"
)

print(f"Merged individuals: {len(df)}")


# ============================================================
# 5. POPULATION COMPOSITION
# ============================================================

print("\n" + "-" * 90)
print("SUPERPOPULATION COMPOSITION")
print("-" * 90)

population_counts = (
    df["super_pop"]
    .value_counts()
    .sort_index()
)

print(population_counts)


# ============================================================
# 6. POPULATION-SPECIFIC PRS STATISTICS
# ============================================================

print("\n" + "-" * 90)
print("CAD PRS BY SUPERPOPULATION")
print("-" * 90)

stats = (
    df.groupby("super_pop")["PRS"]
    .agg(
        N="count",
        Mean="mean",
        SD="std",
        Median="median",
        Min="min",
        Max="max"
    )
)

stats["Q1"] = (
    df.groupby("super_pop")["PRS"]
    .quantile(0.25)
)

stats["Q3"] = (
    df.groupby("super_pop")["PRS"]
    .quantile(0.75)
)

stats["SE"] = stats["SD"] / stats["N"].pow(0.5)

# Put columns in publication-friendly order
stats = stats[
    [
        "N",
        "Mean",
        "SD",
        "Median",
        "Min",
        "Q1",
        "Q3",
        "Max",
        "SE"
    ]
]

print(stats.to_string())


# ============================================================
# 7. KRUSKAL-WALLIS TEST
# ============================================================

print("\n" + "-" * 90)
print("KRUSKAL-WALLIS TEST")
print("-" * 90)

populations = ["AFR", "AMR", "EAS", "EUR", "SAS"]

groups = [
    df.loc[df["super_pop"] == pop, "PRS"]
    for pop in populations
]

H, p = kruskal(*groups)

print(f"Kruskal-Wallis H = {H:.10f}")
print(f"Kruskal-Wallis p = {p:.10e}")

if p < 0.001:
    print("Result: Significant population-level PRS distribution difference (p < 0.001).")
elif p < 0.05:
    print("Result: Significant population-level PRS distribution difference (p < 0.05).")
else:
    print("Result: No statistically significant difference between superpopulations.")


# ============================================================
# 8. SAVE OUTPUTS
# ============================================================

df.to_csv(
    OUTPUT_LABELLED,
    sep="\t",
    index=False
)

stats.to_csv(
    OUTPUT_STATS,
    sep="\t"
)

kw_results = pd.DataFrame({
    "Test": ["Kruskal-Wallis"],
    "H_statistic": [H],
    "p_value": [p]
})

kw_results.to_csv(
    OUTPUT_KW,
    sep="\t",
    index=False
)


# ============================================================
# 9. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 90)
print("CAD POPULATION ANALYSIS COMPLETE")
print("=" * 90)

print("\nOutputs:")

print(f"\nPopulation-labelled PRS:")
print(OUTPUT_LABELLED)

print(f"\nPopulation statistics:")
print(OUTPUT_STATS)

print(f"\nKruskal-Wallis results:")
print(OUTPUT_KW)

print("\nPopulation counts:")
print(population_counts)

print("\nAnalysis complete.")
