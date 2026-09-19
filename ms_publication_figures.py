import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# MULTIPLE SCLEROSIS — PUBLICATION FIGURES
# PGS002726
# ============================================================

BASE = r"C:\PRS_Study"

RESULTS_DIR = os.path.join(
    BASE,
    "results",
    "Multiple_Sclerosis"
)

POP_DIR = os.path.join(
    RESULTS_DIR,
    "population_analysis"
)

OUTPUT_DIR = os.path.join(
    POP_DIR,
    "publication_figures"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

# ============================================================
# Input files
# ============================================================

PRS_FILE = os.path.join(
    RESULTS_DIR,
    "MS_genomewide_PRS_with_risk_groups.tsv"
)

POP_FILE = os.path.join(
    POP_DIR,
    "MS_PRS_with_population_labels.tsv"
)

POP_SUMMARY_FILE = os.path.join(
    POP_DIR,
    "MS_population_PRS_by_superpopulation.tsv"
)

DUNN_FILE = os.path.join(
    POP_DIR,
    "MS_population_Dunn_Holm_pvalues.tsv"
)

# ============================================================
# Load data
# ============================================================

prs = pd.read_csv(
    PRS_FILE,
    sep="\t"
)

pop = pd.read_csv(
    POP_FILE,
    sep="\t"
)

summary = pd.read_csv(
    POP_SUMMARY_FILE,
    sep="\t"
)

dunn = pd.read_csv(
    DUNN_FILE,
    sep="\t",
    index_col=0
)

# ============================================================
# Population order
# ============================================================

population_order = [
    "AFR",
    "AMR",
    "EAS",
    "EUR",
    "SAS"
]

# ============================================================
# Figure 1
# Population PRS distributions
# ============================================================

print("Creating Figure 1...")

fig, ax = plt.subplots(
    figsize=(8, 6)
)

data = [
    pop.loc[
        pop["super_pop"] == p,
        "PRS"
    ].values
    for p in population_order
]

ax.boxplot(
    data,
    tick_labels=population_order,
    showfliers=False
)

# Individual observations
rng = np.random.default_rng(42)

for i, values in enumerate(data, start=1):

    jitter = rng.normal(
        i,
        0.045,
        size=len(values)
    )

    ax.scatter(
        jitter,
        values,
        alpha=0.18,
        s=10
    )

ax.set_xlabel(
    "1000 Genomes Superpopulation"
)

ax.set_ylabel(
    "Multiple Sclerosis PRS"
)

ax.set_title(
    "Multiple Sclerosis PRS Distribution by Superpopulation"
)

ax.grid(
    axis="y",
    alpha=0.25
)

plt.tight_layout()

fig1 = os.path.join(
    OUTPUT_DIR,
    "Figure1_MS_PRS_by_Superpopulation.png"
)

plt.savefig(
    fig1,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ============================================================
# Figure 2
# Mean PRS ± 95% CI
# ============================================================

print("Creating Figure 2...")

fig, ax = plt.subplots(
    figsize=(8, 6)
)

plot_summary = (
    summary
    .set_index("Superpopulation")
    .loc[population_order]
    .reset_index()
)

means = plot_summary["Mean_PRS"].values
ses = plot_summary["SE"].values

ci = 1.96 * ses

x = np.arange(
    len(population_order)
)

ax.errorbar(
    x,
    means,
    yerr=ci,
    fmt="o",
    capsize=5,
    markersize=7
)

ax.axhline(
    0,
    linestyle="--",
    linewidth=1
)

ax.set_xticks(
    x
)

ax.set_xticklabels(
    population_order
)

ax.set_xlabel(
    "1000 Genomes Superpopulation"
)

ax.set_ylabel(
    "Mean Multiple Sclerosis PRS (95% CI)"
)

ax.set_title(
    "Mean Multiple Sclerosis PRS by Superpopulation"
)

ax.grid(
    axis="y",
    alpha=0.25
)

plt.tight_layout()

fig2 = os.path.join(
    OUTPUT_DIR,
    "Figure2_MS_Mean_PRS_95CI.png"
)

plt.savefig(
    fig2,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ============================================================
# Figure 3
# Overall PRS distribution
# ============================================================

print("Creating Figure 3...")

fig, ax = plt.subplots(
    figsize=(8, 6)
)

ax.hist(
    prs["PRS"],
    bins=40,
    edgecolor="black"
)

ax.axvline(
    prs["PRS"].median(),
    linestyle="--",
    linewidth=1.5,
    label=f"Median = {prs['PRS'].median():.3f}"
)

ax.set_xlabel(
    "Multiple Sclerosis PRS"
)

ax.set_ylabel(
    "Number of Individuals"
)

ax.set_title(
    "Genome-wide Multiple Sclerosis PRS Distribution"
)

ax.legend()

ax.grid(
    axis="y",
    alpha=0.25
)

plt.tight_layout()

fig3 = os.path.join(
    OUTPUT_DIR,
    "Figure3_MS_Overall_PRS_Distribution.png"
)

plt.savefig(
    fig3,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ============================================================
# Figure 4
# Relative PRS group counts
# ============================================================

print("Creating Figure 4...")

group_order = [
    "Low",
    "Intermediate",
    "High"
]

group_counts = (
    prs["Risk_Group"]
    .value_counts()
    .reindex(group_order)
    .fillna(0)
)

fig, ax = plt.subplots(
    figsize=(7, 6)
)

bars = ax.bar(
    group_order,
    group_counts.values
)

ax.set_xlabel(
    "Relative PRS Group"
)

ax.set_ylabel(
    "Number of Individuals"
)

ax.set_title(
    "Relative Multiple Sclerosis PRS Groups"
)

ax.grid(
    axis="y",
    alpha=0.25
)

for bar, value in zip(
    bars,
    group_counts.values
):

    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{int(value)}",
        ha="center",
        va="bottom"
    )

plt.tight_layout()

fig4 = os.path.join(
    OUTPUT_DIR,
    "Figure4_MS_Relative_PRS_Groups.png"
)

plt.savefig(
    fig4,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ============================================================
# Figure 5
# Dunn-Holm heatmap
#
# We cap the visual -log10(p) scale at 50 so that extremely
# small p-values do not dominate the visualization.
#
# Original adjusted p-values remain unchanged in the TSV.
# ============================================================

print("Creating Figure 5...")

p_matrix = dunn.loc[
    population_order,
    population_order
].astype(float)

# Convert p-values to -log10 scale.
# Protect against exact zero.
safe_p = p_matrix.clip(
    lower=np.finfo(float).tiny
)

log_p = -np.log10(
    safe_p
)

# Cap visual scale.
visual_log_p = log_p.clip(
    upper=50
)

# Mask diagonal.
mask = np.eye(
    len(population_order),
    dtype=bool
)

visual_log_p = visual_log_p.mask(
    mask
)

fig, ax = plt.subplots(
    figsize=(8, 7)
)

im = ax.imshow(
    visual_log_p.values,
    aspect="auto"
)

cbar = plt.colorbar(
    im,
    ax=ax
)

cbar.set_label(
    "-log10(Holm-adjusted p-value), capped at 50"
)

ax.set_xticks(
    np.arange(len(population_order))
)

ax.set_yticks(
    np.arange(len(population_order))
)

ax.set_xticklabels(
    population_order
)

ax.set_yticklabels(
    population_order
)

ax.set_xlabel(
    "Superpopulation"
)

ax.set_ylabel(
    "Superpopulation"
)

ax.set_title(
    "Dunn-Holm Pairwise Comparisons of MS PRS"
)

# Add p-value annotations
for i in range(len(population_order)):

    for j in range(len(population_order)):

        if i == j:
            continue

        p = p_matrix.iloc[i, j]

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
    OUTPUT_DIR,
    "Figure5_MS_Dunn_Holm_Heatmap.png"
)

plt.savefig(
    fig5,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ============================================================
# Figure 6
# 1000 Genomes population composition
# ============================================================

print("Creating Figure 6...")

population_counts = (
    pop["super_pop"]
    .value_counts()
    .reindex(population_order)
)

fig, ax = plt.subplots(
    figsize=(8, 6)
)

bars = ax.bar(
    population_order,
    population_counts.values
)

ax.set_xlabel(
    "1000 Genomes Superpopulation"
)

ax.set_ylabel(
    "Number of Individuals"
)

ax.set_title(
    "1000 Genomes Phase 3 Superpopulation Composition"
)

ax.grid(
    axis="y",
    alpha=0.25
)

for bar, value in zip(
    bars,
    population_counts.values
):

    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{int(value)}",
        ha="center",
        va="bottom"
    )

plt.tight_layout()

fig6 = os.path.join(
    OUTPUT_DIR,
    "Figure6_MS_1000G_Population_Composition.png"
)

plt.savefig(
    fig6,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ============================================================
# Final output
# ============================================================

print("\n")
print("=" * 70)
print("MULTIPLE SCLEROSIS PUBLICATION FIGURES COMPLETE")
print("=" * 70)

print(f"\nOutput directory:")
print(OUTPUT_DIR)

print("\nFigures created:")

print(
    "1.",
    fig1
)

print(
    "2.",
    fig2
)

print(
    "3.",
    fig3
)

print(
    "4.",
    fig4
)

print(
    "5.",
    fig5
)

print(
    "6.",
    fig6
)

print("\nStatus: SUCCESS")
print("=" * 70)
