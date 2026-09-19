import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# CAD PUBLICATION FIGURES
# ============================================================

BASE = Path(r"C:\PRS_Study")
CAD_DIR = BASE / "results" / "CAD"

PRS_FILE = CAD_DIR / "CAD_PRS_with_population_labels.tsv"
POP_STATS_FILE = CAD_DIR / "CAD_population_PRS_by_superpopulation.tsv"
DUNN_FILE = CAD_DIR / "CAD_population_Dunn_Holm_pvalues.tsv"

FIG_DIR = (
    CAD_DIR /
    "population_analysis" /
    "publication_figures"
)

FIG_DIR.mkdir(parents=True, exist_ok=True)

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

print("=" * 90)
print("CAD PUBLICATION FIGURES")
print("=" * 90)

print("\nLoading CAD population-labelled PRS...")

df = pd.read_csv(
    PRS_FILE,
    sep="\t"
)

print(f"Individuals loaded: {len(df)}")

print("\nLoading population statistics...")

stats = pd.read_csv(
    POP_STATS_FILE,
    sep="\t"
)

if "super_pop" in stats.columns:
    stats = stats.set_index("super_pop")

print("\nLoading Dunn–Holm results...")

dunn = pd.read_csv(
    DUNN_FILE,
    sep="\t"
)

print(f"Dunn–Holm comparisons: {len(dunn)}")


# ============================================================
# VALIDATION
# ============================================================

required_prs_columns = [
    "IID",
    "PRS",
    "Relative_PRS_Group",
    "super_pop"
]

for column in required_prs_columns:

    if column not in df.columns:
        raise ValueError(
            f"Required column '{column}' not found.\n"
            f"Available columns: {list(df.columns)}"
        )

required_stats_columns = [
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

for column in required_stats_columns:

    if column not in stats.columns:
        raise ValueError(
            f"Required column '{column}' not found "
            f"in population statistics."
        )


# ============================================================
# FIGURE 1
# POPULATION PRS DISTRIBUTIONS
# ============================================================

print("\nCreating Figure 1...")

population_data = [
    df.loc[
        df["super_pop"] == population,
        "PRS"
    ].dropna()
    for population in POPULATIONS
]

plt.figure(figsize=(9, 6))

plt.boxplot(
    population_data,
    tick_labels=POPULATIONS,
    showfliers=False
)

rng = np.random.default_rng(42)

for index, values in enumerate(
    population_data,
    start=1
):

    jittered_x = rng.normal(
        loc=index,
        scale=0.055,
        size=len(values)
    )

    plt.scatter(
        jittered_x,
        values,
        alpha=0.25,
        s=10
    )

plt.xlabel(
    "1000 Genomes Superpopulation"
)

plt.ylabel(
    "CAD PRS"
)

plt.title(
    "CAD PRS Distribution Across 1000 Genomes Superpopulations"
)

plt.tight_layout()

figure1 = (
    FIG_DIR /
    "Figure1_CAD_population_PRS_distribution.png"
)

plt.savefig(
    figure1,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(f"Figure 1 saved: {figure1}")


# ============================================================
# FIGURE 2
# MEAN PRS ± 95% CI
# ============================================================

print("\nCreating Figure 2...")

means = (
    stats.loc[
        POPULATIONS,
        "Mean"
    ]
    .astype(float)
    .values
)

ses = (
    stats.loc[
        POPULATIONS,
        "SE"
    ]
    .astype(float)
    .values
)

ci95 = 1.96 * ses

x = np.arange(
    len(POPULATIONS)
)

plt.figure(figsize=(9, 6))

plt.errorbar(
    x,
    means,
    yerr=ci95,
    fmt="o",
    capsize=5,
    markersize=7
)

plt.axhline(
    0,
    linestyle="--",
    linewidth=1
)

plt.xticks(
    x,
    POPULATIONS
)

plt.xlabel(
    "1000 Genomes Superpopulation"
)

plt.ylabel(
    "Mean CAD PRS ± 95% CI"
)

plt.title(
    "Mean CAD PRS Across 1000 Genomes Superpopulations"
)

plt.tight_layout()

figure2 = (
    FIG_DIR /
    "Figure2_CAD_population_mean_95CI.png"
)

plt.savefig(
    figure2,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(f"Figure 2 saved: {figure2}")


# ============================================================
# FIGURE 3
# OVERALL PRS DISTRIBUTION
# ============================================================

print("\nCreating Figure 3...")

prs_values = df["PRS"].dropna()

plt.figure(figsize=(9, 6))

plt.hist(
    prs_values,
    bins=40,
    edgecolor="black"
)

plt.xlabel(
    "CAD PRS"
)

plt.ylabel(
    "Number of Individuals"
)

plt.title(
    "Overall Distribution of CAD PRS"
)

plt.tight_layout()

figure3 = (
    FIG_DIR /
    "Figure3_CAD_overall_PRS_distribution.png"
)

plt.savefig(
    figure3,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(f"Figure 3 saved: {figure3}")


# ============================================================
# FIGURE 4
# RELATIVE PRS GROUP DISTRIBUTION
# ============================================================

print("\nCreating Figure 4...")

group_order = [
    "Low",
    "Intermediate",
    "High"
]

group_counts = (
    df["Relative_PRS_Group"]
    .value_counts()
    .reindex(group_order)
    .fillna(0)
)

plt.figure(figsize=(8, 6))

plt.bar(
    group_order,
    group_counts.values,
    edgecolor="black"
)

plt.xlabel(
    "Relative PRS Group"
)

plt.ylabel(
    "Number of Individuals"
)

plt.title(
    "Relative PRS Group Distribution"
)

plt.tight_layout()

figure4 = (
    FIG_DIR /
    "Figure4_CAD_relative_PRS_groups.png"
)

plt.savefig(
    figure4,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(f"Figure 4 saved: {figure4}")


# ============================================================
# FIGURE 5
# DUNN–HOLM HEATMAP
# ============================================================

print("\nCreating Figure 5...")

# ------------------------------------------------------------
# Create pairwise p-value matrix
# ------------------------------------------------------------

p_matrix = pd.DataFrame(
    np.nan,
    index=POPULATIONS,
    columns=POPULATIONS
)

for _, row in dunn.iterrows():

    population1 = row["Population_1"]
    population2 = row["Population_2"]

    adjusted_p = float(
        row["Holm_adjusted_p"]
    )

    p_matrix.loc[
        population1,
        population2
    ] = adjusted_p

    p_matrix.loc[
        population2,
        population1
    ] = adjusted_p


# ------------------------------------------------------------
# Convert p-values to -log10 scale
# ------------------------------------------------------------

with np.errstate(
    divide="ignore",
    invalid="ignore"
):

    visual_matrix = -np.log10(
        p_matrix.astype(float)
    )

# Cap visual scale at 50
visual_matrix = visual_matrix.clip(
    lower=0,
    upper=50
)

# IMPORTANT:
# Convert to a new writable NumPy array.
# This avoids NumPy 2.x read-only array errors.

visual_array = np.array(
    visual_matrix.values,
    dtype=float,
    copy=True
)

# Set diagonal to zero
for i in range(
    len(POPULATIONS)
):
    visual_array[i, i] = 0


# ------------------------------------------------------------
# Plot heatmap
# ------------------------------------------------------------

plt.figure(figsize=(8, 7))

image = plt.imshow(
    visual_array,
    aspect="equal"
)

plt.xticks(
    np.arange(len(POPULATIONS)),
    POPULATIONS
)

plt.yticks(
    np.arange(len(POPULATIONS)),
    POPULATIONS
)

plt.xlabel(
    "Superpopulation"
)

plt.ylabel(
    "Superpopulation"
)

plt.title(
    "Dunn–Holm Pairwise Comparisons of CAD PRS"
)


# ------------------------------------------------------------
# Add p-value labels
# ------------------------------------------------------------

for i in range(
    len(POPULATIONS)
):

    for j in range(
        len(POPULATIONS)
    ):

        if i == j:

            text = "—"

        else:

            p = p_matrix.iloc[
                i,
                j
            ]

            if pd.isna(p):

                text = ""

            elif p < 0.001:

                text = "p<0.001"

            else:

                text = f"p={p:.3f}"

        plt.text(
            j,
            i,
            text,
            ha="center",
            va="center",
            fontsize=8
        )


plt.colorbar(
    image,
    label="-log10(Holm-adjusted p)"
)

plt.tight_layout()

figure5 = (
    FIG_DIR /
    "Figure5_CAD_Dunn_Holm_heatmap.png"
)

plt.savefig(
    figure5,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(f"Figure 5 saved: {figure5}")


# ============================================================
# FIGURE 6
# 1000 GENOMES POPULATION COMPOSITION
# ============================================================

print("\nCreating Figure 6...")

population_counts = (
    df["super_pop"]
    .value_counts()
    .reindex(POPULATIONS)
    .fillna(0)
)

plt.figure(figsize=(8, 6))

plt.bar(
    POPULATIONS,
    population_counts.values,
    edgecolor="black"
)

plt.xlabel(
    "1000 Genomes Superpopulation"
)

plt.ylabel(
    "Number of Individuals"
)

plt.title(
    "1000 Genomes Superpopulation Composition"
)

plt.tight_layout()

figure6 = (
    FIG_DIR /
    "Figure6_CAD_population_composition.png"
)

plt.savefig(
    figure6,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(f"Figure 6 saved: {figure6}")


# ============================================================
# FIGURE SUMMARY
# ============================================================

figure_summary = pd.DataFrame({

    "Figure": [
        "Figure 1",
        "Figure 2",
        "Figure 3",
        "Figure 4",
        "Figure 5",
        "Figure 6"
    ],

    "Description": [
        "CAD PRS distribution across 1000 Genomes superpopulations",
        "Mean CAD PRS with 95% confidence intervals",
        "Overall CAD PRS distribution",
        "Relative PRS group distribution",
        "Dunn-Holm pairwise population comparisons",
        "1000 Genomes superpopulation composition"
    ],

    "File": [
        figure1.name,
        figure2.name,
        figure3.name,
        figure4.name,
        figure5.name,
        figure6.name
    ]
})

summary_file = (
    FIG_DIR /
    "CAD_publication_figures_summary.tsv"
)

figure_summary.to_csv(
    summary_file,
    sep="\t",
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 90)
print("CAD PUBLICATION FIGURES COMPLETE")
print("=" * 90)

print("\nOutput directory:")
print(FIG_DIR)

print("\nGenerated files:")

for file in sorted(
    FIG_DIR.glob("Figure*.png")
):

    print(
        f"  {file.name}"
    )

print("\nFigure summary:")
print(summary_file)

print("\nAll six CAD publication figures were generated successfully.")
