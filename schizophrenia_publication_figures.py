import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# SCHIZOPHRENIA — PUBLICATION-QUALITY FIGURES
# ============================================================

BASE = Path(r"C:\PRS_Study")

RESULTS = (
    BASE
    / "results"
    / "Schizophrenia"
)

POP_DIR = (
    RESULTS
    / "population_analysis"
)

FIG_DIR = (
    POP_DIR
    / "publication_figures"
)

FIG_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# INPUT FILES
# ============================================================

PRS_FILE = (
    RESULTS
    / "Schizophrenia_genomewide_PRS_with_risk_groups.tsv"
)

POP_FILE = (
    POP_DIR
    / "Schizophrenia_PRS_with_population_labels.tsv"
)

POP_STATS_FILE = (
    POP_DIR
    / "Schizophrenia_population_PRS_by_superpopulation.tsv"
)

DUNN_FILE = (
    POP_DIR
    / "Schizophrenia_population_Dunn_Holm_pvalues.tsv"
)


# ============================================================
# 1. START
# ============================================================

print("=" * 70)
print("SCHIZOPHRENIA PUBLICATION FIGURES")
print("=" * 70)


# ============================================================
# 2. CHECK INPUT FILES
# ============================================================

required_files = [
    PRS_FILE,
    POP_FILE,
    POP_STATS_FILE,
    DUNN_FILE
]

for file in required_files:

    if not file.exists():

        raise FileNotFoundError(
            f"\nRequired file not found:\n{file}"
        )

print("\nAll required input files found.")


# ============================================================
# 3. LOAD DATA
# ============================================================

prs = pd.read_csv(
    PRS_FILE,
    sep="\t"
)

pop = pd.read_csv(
    POP_FILE,
    sep="\t"
)

stats = pd.read_csv(
    POP_STATS_FILE,
    sep="\t"
)

dunn = pd.read_csv(
    DUNN_FILE,
    sep="\t",
    index_col=0
)


# ============================================================
# 4. CHECK REQUIRED COLUMNS
# ============================================================

if "PRS" not in prs.columns:
    raise ValueError(
        "PRS column not found in genome-wide PRS file."
    )

if "PRS_Risk_Group" not in prs.columns:
    raise ValueError(
        "PRS_Risk_Group column not found."
    )

if "PRS" not in pop.columns:
    raise ValueError(
        "PRS column not found in population-labeled file."
    )

if "super_pop" not in pop.columns:
    raise ValueError(
        "super_pop column not found in population-labeled file."
    )

if "Superpopulation" not in stats.columns:
    raise ValueError(
        "Superpopulation column not found in population statistics."
    )

if "Mean_PRS" not in stats.columns:
    raise ValueError(
        "Mean_PRS column not found in population statistics."
    )

if "SE_PRS" not in stats.columns:
    raise ValueError(
        "SE_PRS column not found in population statistics."
    )


# ============================================================
# 5. DEFINE ORDERS
# ============================================================

population_order = [
    "AFR",
    "AMR",
    "EAS",
    "EUR",
    "SAS"
]

group_order = [
    "Low",
    "Intermediate",
    "High"
]


# ============================================================
# 6. PREPARE DATA
# ============================================================

prs["PRS"] = pd.to_numeric(
    prs["PRS"],
    errors="coerce"
)

pop["PRS"] = pd.to_numeric(
    pop["PRS"],
    errors="coerce"
)

stats["Mean_PRS"] = pd.to_numeric(
    stats["Mean_PRS"],
    errors="coerce"
)

stats["SE_PRS"] = pd.to_numeric(
    stats["SE_PRS"],
    errors="coerce"
)

pop = pop[
    pop["super_pop"].isin(
        population_order
    )
].copy()

print(
    f"\nPRS samples: {len(prs)}"
)

print(
    f"Population-labeled samples: {len(pop)}"
)


# ============================================================
# FIGURE 1
# PRS DISTRIBUTION BY SUPERPOPULATION
# Boxplot + individual observations
# ============================================================

print("\nCreating Figure 1...")

data = []

for population in population_order:

    values = pop.loc[
        pop["super_pop"] == population,
        "PRS"
    ].dropna()

    data.append(
        values.to_numpy()
    )


fig, ax = plt.subplots(
    figsize=(9, 6)
)


# IMPORTANT:
# Matplotlib 3.9+ / current versions use
# tick_labels instead of labels.

ax.boxplot(
    data,
    tick_labels=population_order,
    patch_artist=False,
    showfliers=False
)


# ------------------------------------------------------------
# Add individual observations
# ------------------------------------------------------------

rng = np.random.default_rng(
    42
)

for i, values in enumerate(
    data,
    start=1
):

    jitter = rng.uniform(
        -0.08,
        0.08,
        size=len(values)
    )

    ax.scatter(
        np.full(
            len(values),
            i
        ) + jitter,
        values,
        s=8,
        alpha=0.20
    )


ax.set_xlabel(
    "1000 Genomes Superpopulation",
    fontsize=12
)

ax.set_ylabel(
    "Schizophrenia PRS",
    fontsize=12
)

ax.set_title(
    "Schizophrenia PRS Distribution by Superpopulation",
    fontsize=14
)

ax.grid(
    axis="y",
    linestyle="--",
    alpha=0.3
)

plt.tight_layout()


fig1 = (
    FIG_DIR
    / "Figure1_Schizophrenia_PRS_by_Superpopulation_Boxplot.png"
)

plt.savefig(
    fig1,
    dpi=600,
    bbox_inches="tight"
)

plt.close()

print(
    f"Saved: {fig1}"
)


# ============================================================
# FIGURE 2
# MEAN PRS WITH 95% CONFIDENCE INTERVAL
# Dot-and-whisker plot
# ============================================================

print("\nCreating Figure 2...")

stats_plot = (
    stats[
        stats["Superpopulation"].isin(
            population_order
        )
    ]
    .set_index(
        "Superpopulation"
    )
    .reindex(
        population_order
    )
)


if stats_plot["Mean_PRS"].isna().any():

    raise ValueError(
        "Missing population mean PRS values."
    )

if stats_plot["SE_PRS"].isna().any():

    raise ValueError(
        "Missing population SE values."
    )


means = stats_plot[
    "Mean_PRS"
].to_numpy()

se = stats_plot[
    "SE_PRS"
].to_numpy()


# 95% CI
ci = 1.96 * se

lower = means - ci

upper = means + ci

y = np.arange(
    len(population_order)
)


fig, ax = plt.subplots(
    figsize=(9, 6)
)


ax.errorbar(
    means,
    y,
    xerr=[
        means - lower,
        upper - means
    ],
    fmt="o",
    markersize=7,
    capsize=4,
    linewidth=1.5
)


# Reference line at zero
ax.axvline(
    0,
    linestyle="--",
    linewidth=1
)


ax.set_yticks(
    y
)

ax.set_yticklabels(
    population_order
)

ax.set_xlabel(
    "Mean Schizophrenia PRS",
    fontsize=12
)

ax.set_ylabel(
    "1000 Genomes Superpopulation",
    fontsize=12
)

ax.set_title(
    "Mean Schizophrenia PRS with 95% Confidence Intervals",
    fontsize=14
)

ax.grid(
    axis="x",
    linestyle="--",
    alpha=0.3
)

plt.tight_layout()


fig2 = (
    FIG_DIR
    / "Figure2_Schizophrenia_Mean_PRS_95CI.png"
)

plt.savefig(
    fig2,
    dpi=600,
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
].dropna()

if len(values) == 0:

    raise ValueError(
        "No valid PRS values available."
    )


mean_value = values.mean()

median_value = values.median()


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
    mean_value,
    linestyle="--",
    linewidth=2,
    label=f"Mean = {mean_value:.3f}"
)


ax.axvline(
    median_value,
    linestyle=":",
    linewidth=2,
    label=f"Median = {median_value:.3f}"
)


ax.set_xlabel(
    "Genome-wide Schizophrenia PRS",
    fontsize=12
)

ax.set_ylabel(
    "Number of Individuals",
    fontsize=12
)

ax.set_title(
    "Overall Distribution of Schizophrenia PRS",
    fontsize=14
)

ax.legend()

ax.grid(
    axis="y",
    linestyle="--",
    alpha=0.3
)

plt.tight_layout()


fig3 = (
    FIG_DIR
    / "Figure3_Schizophrenia_Overall_PRS_Distribution.png"
)

plt.savefig(
    fig3,
    dpi=600,
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

group_counts = (
    prs[
        "PRS_Risk_Group"
    ]
    .value_counts()
    .reindex(
        group_order,
        fill_value=0
    )
)

group_percentages = (
    group_counts
    / len(prs)
    * 100
)


fig, ax = plt.subplots(
    figsize=(8, 6)
)


bars = ax.bar(
    group_order,
    group_counts.to_numpy(),
    edgecolor="black"
)


for bar, count, percentage in zip(
    bars,
    group_counts.to_numpy(),
    group_percentages.to_numpy()
):

    ax.text(
        bar.get_x()
        + bar.get_width() / 2,
        bar.get_height(),
        f"{count}\n({percentage:.1f}%)",
        ha="center",
        va="bottom",
        fontsize=10
    )


ax.set_xlabel(
    "Relative PRS Group",
    fontsize=12
)

ax.set_ylabel(
    "Number of Individuals",
    fontsize=12
)

ax.set_title(
    "Relative Schizophrenia PRS Group Distribution",
    fontsize=14
)

ax.grid(
    axis="y",
    linestyle="--",
    alpha=0.3
)

plt.tight_layout()


fig4 = (
    FIG_DIR
    / "Figure4_Schizophrenia_Relative_PRS_Groups.png"
)

plt.savefig(
    fig4,
    dpi=600,
    bbox_inches="tight"
)

plt.close()

print(
    f"Saved: {fig4}"
)


# ============================================================
# FIGURE 5
# DUNN-HOLM P-VALUE HEATMAP
# ============================================================

print("\nCreating Figure 5...")

# Reorder matrix
dunn = dunn.reindex(
    index=population_order,
    columns=population_order
)

# Convert to numeric
dunn = dunn.apply(
    pd.to_numeric,
    errors="coerce"
)


if dunn.isna().all().all():

    raise ValueError(
        "Dunn-Holm matrix contains no usable p-values."
    )


# ------------------------------------------------------------
# Prepare values for -log10 transformation
# ------------------------------------------------------------

pvalues = dunn.copy()

# Replace zero values with smallest positive float
pvalues[
    pvalues <= 0
] = np.finfo(float).tiny


log_p = (
    -np.log10(
        pvalues
    )
)


# Diagonal = 0
for i in range(
    len(population_order)
):

    log_p.iloc[i, i] = 0


# ------------------------------------------------------------
# Create heatmap
# ------------------------------------------------------------

fig, ax = plt.subplots(
    figsize=(8, 7)
)


im = ax.imshow(
    log_p.to_numpy(),
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
    "Superpopulation",
    fontsize=12
)

ax.set_ylabel(
    "Superpopulation",
    fontsize=12
)

ax.set_title(
    "Dunn–Holm Pairwise Comparisons of Schizophrenia PRS",
    fontsize=14
)


# ------------------------------------------------------------
# Add p-value text
# ------------------------------------------------------------

for i in range(
    len(population_order)
):

    for j in range(
        len(population_order)
    ):

        if i == j:

            text = "—"

        else:

            p = dunn.iloc[i, j]

            if pd.isna(p):

                text = "NA"

            elif p < 0.001:

                text = "<0.001"

            else:

                text = f"{p:.3f}"

        ax.text(
            j,
            i,
            text,
            ha="center",
            va="center",
            fontsize=9
        )


cbar = fig.colorbar(
    im,
    ax=ax
)

cbar.set_label(
    "−log10(Holm-adjusted p-value)",
    rotation=90
)


plt.tight_layout()


fig5 = (
    FIG_DIR
    / "Figure5_Schizophrenia_Dunn_Holm_Heatmap.png"
)

plt.savefig(
    fig5,
    dpi=600,
    bbox_inches="tight"
)

plt.close()

print(
    f"Saved: {fig5}"
)


# ============================================================
# FIGURE 6
# 1000 GENOMES SUPERPOPULATION SAMPLE COMPOSITION
# ============================================================

print("\nCreating Figure 6...")

sample_counts = (
    pop[
        "super_pop"
    ]
    .value_counts()
    .reindex(
        population_order,
        fill_value=0
)

)

sample_percentages = (
    sample_counts
    / sample_counts.sum()
    * 100
)


fig, ax = plt.subplots(
    figsize=(8, 6)
)


bars = ax.bar(
    population_order,
    sample_counts.to_numpy(),
    edgecolor="black"
)


for bar, count, percentage in zip(
    bars,
    sample_counts.to_numpy(),
    sample_percentages.to_numpy()
):

    ax.text(
        bar.get_x()
        + bar.get_width() / 2,
        bar.get_height(),
        f"{count}\n({percentage:.1f}%)",
        ha="center",
        va="bottom",
        fontsize=10
    )


ax.set_xlabel(
    "1000 Genomes Superpopulation",
    fontsize=12
)

ax.set_ylabel(
    "Number of Individuals",
    fontsize=12
)

ax.set_title(
    "1000 Genomes Superpopulation Composition",
    fontsize=14
)

ax.grid(
    axis="y",
    linestyle="--",
    alpha=0.3
)

plt.tight_layout()


fig6 = (
    FIG_DIR
    / "Figure6_Schizophrenia_1000G_Superpopulation_Composition.png"
)

plt.savefig(
    fig6,
    dpi=600,
    bbox_inches="tight"
)

plt.close()

print(
    f"Saved: {fig6}"
)


# ============================================================
# 11. FIGURE SUMMARY TABLE
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

        "PRS distribution by 1000 Genomes superpopulation",

        "Mean PRS with 95% confidence intervals",

        "Overall genome-wide PRS distribution",

        "Relative PRS group distribution",

        "Dunn-Holm pairwise population comparison",

        "1000 Genomes superpopulation sample composition"
    ],

    "File": [

        fig1.name,
        fig2.name,
        fig3.name,
        fig4.name,
        fig5.name,
        fig6.name
    ]
})


SUMMARY_FILE = (
    FIG_DIR
    / "Schizophrenia_publication_figures_summary.tsv"
)


figure_summary.to_csv(
    SUMMARY_FILE,
    sep="\t",
    index=False
)


# ============================================================
# 12. FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("PUBLICATION FIGURES COMPLETE")
print("=" * 70)

print("\nFigures created:")

for file in [
    fig1,
    fig2,
    fig3,
    fig4,
    fig5,
    fig6
]:

    print(
        f"  {file.name}"
    )


print("\nFigure summary:")
print(SUMMARY_FILE)

print("\nOutput directory:")
print(FIG_DIR)

print("\n" + "=" * 70)
print("SCHIZOPHRENIA PUBLICATION FIGURE GENERATION COMPLETE")
print("=" * 70)
