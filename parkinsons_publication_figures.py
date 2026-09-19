import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# PARKINSON'S DISEASE — PUBLICATION FIGURES
# ============================================================

PROJECT = r"C:\PRS_Study"

RESULTS_DIR = os.path.join(
    PROJECT,
    "results",
    "Parkinsons"
)

POP_DIR = os.path.join(
    RESULTS_DIR,
    "population_analysis"
)

FIG_DIR = os.path.join(
    POP_DIR,
    "publication_figures"
)

os.makedirs(
    FIG_DIR,
    exist_ok=True
)

# ============================================================
# INPUT FILES
# ============================================================

PRS_FILE = os.path.join(
    RESULTS_DIR,
    "Parkinsons_genomewide_PRS_with_risk_groups.tsv"
)

POP_FILE = os.path.join(
    POP_DIR,
    "Parkinsons_PRS_with_population_labels.tsv"
)

POP_STATS_FILE = os.path.join(
    POP_DIR,
    "Parkinsons_population_PRS_by_superpopulation.tsv"
)

DUNN_FILE = os.path.join(
    POP_DIR,
    "Parkinsons_population_Dunn_Holm_pvalues.tsv"
)

POP_PANEL = os.path.join(
    PROJECT,
    "data",
    "1000G",
    "1000G_phase3_population_panel.txt"
)

# ============================================================
# LOAD DATA
# ============================================================

print("=" * 80)
print("PARKINSON'S DISEASE — PUBLICATION FIGURES")
print("=" * 80)

prs = pd.read_csv(
    PRS_FILE,
    sep="\t"
)

pop = pd.read_csv(
    POP_FILE,
    sep="\t"
)

pop_stats = pd.read_csv(
    POP_STATS_FILE,
    sep="\t"
)

dunn = pd.read_csv(
    DUNN_FILE,
    sep="\t",
    index_col=0
)

panel = pd.read_csv(
    POP_PANEL,
    sep=r"\s+"
)

print(
    f"\nPRS records: {len(prs):,}"
)

print(
    f"Population records: {len(pop):,}"
)

# ============================================================
# COMMON SETTINGS
# ============================================================

population_order = [
    "AFR",
    "AMR",
    "EAS",
    "EUR",
    "SAS"
]

population_names = {
    "AFR": "AFR",
    "AMR": "AMR",
    "EAS": "EAS",
    "EUR": "EUR",
    "SAS": "SAS"
}

# ============================================================
# FIGURE 1
# PRS DISTRIBUTION BY SUPERPOPULATION
# ============================================================

print("\nCreating Figure 1...")

fig, ax = plt.subplots(
    figsize=(9, 6)
)

data = [
    pop.loc[
        pop["super_pop"] == p,
        "PRS"
    ].astype(float)
    for p in population_order
]

ax.boxplot(
    data,
    tick_labels=[
        population_names[p]
        for p in population_order
    ],
    showfliers=False
)

# Individual observations
for i, values in enumerate(data, start=1):

    x = np.random.normal(
        i,
        0.045,
        size=len(values)
    )

    ax.scatter(
        x,
        values,
        alpha=0.18,
        s=8
    )

ax.set_xlabel(
    "1000 Genomes Superpopulation"
)

ax.set_ylabel(
    "Parkinson's Disease PRS"
)

ax.set_title(
    "Parkinson's Disease PRS Distribution Across 1000 Genomes Superpopulations"
)

ax.grid(
    axis="y",
    alpha=0.25
)

plt.tight_layout()

fig1 = os.path.join(
    FIG_DIR,
    "Figure1_Parkinsons_PRS_by_population.png"
)

plt.savefig(
    fig1,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    f"Saved: {fig1}"
)

# ============================================================
# FIGURE 2
# MEAN PRS ± 95% CI
# ============================================================

print("\nCreating Figure 2...")

stats = pop_stats.copy()

stats = stats.set_index(
    "Superpopulation"
).loc[
    population_order
].reset_index()

stats["CI95"] = (
    1.96 *
    stats["SE"]
)

x = np.arange(
    len(stats)
)

fig, ax = plt.subplots(
    figsize=(9, 6)
)

ax.errorbar(
    x,
    stats["Mean_PRS"],
    yerr=stats["CI95"],
    fmt="o",
    markersize=7,
    capsize=5,
    linewidth=1.5
)

ax.set_xticks(
    x
)

ax.set_xticklabels(
    stats["Superpopulation"]
)

ax.set_xlabel(
    "1000 Genomes Superpopulation"
)

ax.set_ylabel(
    "Mean Parkinson's Disease PRS"
)

ax.set_title(
    "Mean Parkinson's Disease PRS Across 1000 Genomes Superpopulations"
)

ax.grid(
    axis="y",
    alpha=0.25
)

plt.tight_layout()

fig2 = os.path.join(
    FIG_DIR,
    "Figure2_Parkinsons_population_mean_95CI.png"
)

plt.savefig(
    fig2,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    f"Saved: {fig2}"
)

# ============================================================
# FIGURE 3
# OVERALL PRS DISTRIBUTION
# ============================================================

print("\nCreating Figure 3...")

values = prs[
    "PRS"
].astype(float)

fig, ax = plt.subplots(
    figsize=(9, 6)
)

ax.hist(
    values,
    bins=40,
    edgecolor="black",
    alpha=0.75
)

ax.axvline(
    values.mean(),
    linestyle="--",
    linewidth=2,
    label=f"Mean = {values.mean():.3f}"
)

ax.axvline(
    values.median(),
    linestyle=":",
    linewidth=2,
    label=f"Median = {values.median():.3f}"
)

ax.set_xlabel(
    "Genome-wide Parkinson's Disease PRS"
)

ax.set_ylabel(
    "Number of Individuals"
)

ax.set_title(
    "Overall Distribution of Parkinson's Disease PRS"
)

ax.legend()

ax.grid(
    axis="y",
    alpha=0.25
)

plt.tight_layout()

fig3 = os.path.join(
    FIG_DIR,
    "Figure3_Parkinsons_overall_PRS_distribution.png"
)

plt.savefig(
    fig3,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    f"Saved: {fig3}"
)

# ============================================================
# FIGURE 4
# RELATIVE PRS GROUP COUNTS
# ============================================================

print("\nCreating Figure 4...")

group_order = [
    "Low",
    "Intermediate",
    "High"
]

group_counts = (
    prs["PRS_Group"]
    .value_counts()
    .reindex(
        group_order,
        fill_value=0
    )
)

fig, ax = plt.subplots(
    figsize=(8, 6)
)

bars = ax.bar(
    group_order,
    group_counts.values,
    edgecolor="black"
)

ax.set_xlabel(
    "Relative PRS Group"
)

ax.set_ylabel(
    "Number of Individuals"
)

ax.set_title(
    "Relative Parkinson's Disease PRS Groups"
)

ax.grid(
    axis="y",
    alpha=0.25
)

for bar, count in zip(
    bars,
    group_counts.values
):

    ax.text(
        bar.get_x()
        + bar.get_width() / 2,
        bar.get_height(),
        str(count),
        ha="center",
        va="bottom"
    )

plt.tight_layout()

fig4 = os.path.join(
    FIG_DIR,
    "Figure4_Parkinsons_relative_PRS_groups.png"
)

plt.savefig(
    fig4,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    f"Saved: {fig4}"
)

# ============================================================
# FIGURE 5
# DUNN-HOLM HEATMAP
# ============================================================

print("\nCreating Figure 5...")

# Ensure correct population order
dunn = dunn.loc[
    population_order,
    population_order
]

# Convert p-values to -log10
pvalues = dunn.astype(float)

neglog = -np.log10(
    pvalues.clip(
        lower=1e-300
    )
)

# Copy to avoid read-only NumPy issue
heat_values = (
    neglog.to_numpy()
    .copy()
)

# Cap visual scale at 50
# Original p-values remain unchanged
heat_values[
    heat_values > 50
] = 50

fig, ax = plt.subplots(
    figsize=(8, 7)
)

im = ax.imshow(
    heat_values,
    aspect="auto"
)

ax.set_xticks(
    np.arange(
        len(population_order)
    )
)

ax.set_yticks(
    np.arange(
        len(population_order)
    )
)

ax.set_xticklabels(
    population_order
)

ax.set_yticklabels(
    population_order
)

ax.set_xlabel(
    "Population"
)

ax.set_ylabel(
    "Population"
)

ax.set_title(
    "Dunn's Post-hoc Test with Holm Correction"
)

cbar = plt.colorbar(
    im,
    ax=ax
)

cbar.set_label(
    r"$-\log_{10}$(Holm-adjusted p-value)"
)

# Add actual adjusted p-values
for i in range(
    len(population_order)
):

    for j in range(
        len(population_order)
    ):

        if i == j:

            text = "—"

        else:

            p = pvalues.iloc[
                i,
                j
            ]

            if p < 0.001:

                text = "<0.001"

            else:

                text = f"{p:.3f}"

        ax.text(
            j,
            i,
            text,
            ha="center",
            va="center",
            fontsize=8
        )

plt.tight_layout()

fig5 = os.path.join(
    FIG_DIR,
    "Figure5_Parkinsons_Dunn_Holm_heatmap.png"
)

plt.savefig(
    fig5,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    f"Saved: {fig5}"
)

# ============================================================
# FIGURE 6
# 1000 GENOMES POPULATION COMPOSITION
# ============================================================

print("\nCreating Figure 6...")

population_counts = (
    panel["super_pop"]
    .value_counts()
    .reindex(
        population_order,
        fill_value=0
    )
)

fig, ax = plt.subplots(
    figsize=(9, 6)
)

bars = ax.bar(
    population_order,
    population_counts.values,
    edgecolor="black"
)

ax.set_xlabel(
    "1000 Genomes Superpopulation"
)

ax.set_ylabel(
    "Number of Individuals"
)

ax.set_title(
    "1000 Genomes Phase 3 Population Composition"
)

ax.grid(
    axis="y",
    alpha=0.25
)

for bar, count in zip(
    bars,
    population_counts.values
):

    ax.text(
        bar.get_x()
        + bar.get_width() / 2,
        bar.get_height(),
        str(count),
        ha="center",
        va="bottom"
    )

plt.tight_layout()

fig6 = os.path.join(
    FIG_DIR,
    "Figure6_Parkinsons_population_composition.png"
)

plt.savefig(
    fig6,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    f"Saved: {fig6}"
)

# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 80)
print("PARKINSON'S PUBLICATION FIGURES COMPLETE")
print("=" * 80)

print("\nFigures saved in:")

print(
    FIG_DIR
)

print("\nGenerated:")

print("1. Figure1_Parkinsons_PRS_by_population.png")
print("2. Figure2_Parkinsons_population_mean_95CI.png")
print("3. Figure3_Parkinsons_overall_PRS_distribution.png")
print("4. Figure4_Parkinsons_relative_PRS_groups.png")
print("5. Figure5_Parkinsons_Dunn_Holm_heatmap.png")
print("6. Figure6_Parkinsons_population_composition.png")

print("\nSTATUS: SUCCESS")
