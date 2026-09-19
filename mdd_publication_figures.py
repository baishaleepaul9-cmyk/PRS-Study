import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# MDD PUBLICATION FIGURES
# ============================================================

PROJECT = r"C:\PRS_Study"

RESULTS_DIR = os.path.join(
    PROJECT,
    "results",
    "MDD"
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
    "MDD_genomewide_PRS_with_risk_groups.tsv"
)

POP_STATS_FILE = os.path.join(
    POP_DIR,
    "MDD_population_PRS_by_superpopulation.tsv"
)

LABELED_FILE = os.path.join(
    POP_DIR,
    "MDD_PRS_with_population_labels.tsv"
)

DUNN_FILE = os.path.join(
    POP_DIR,
    "MDD_population_Dunn_Holm_pvalues.tsv"
)

# ============================================================
# Population order
# ============================================================

POP_ORDER = [
    "AFR",
    "AMR",
    "EAS",
    "EUR",
    "SAS"
]

GROUP_ORDER = [
    "Low",
    "Intermediate",
    "High"
]


# ============================================================
# Helper function
# ============================================================

def save_figure(fig, filename):

    output_path = os.path.join(
        OUTPUT_DIR,
        filename
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(
        f"Saved: {output_path}"
    )


# ============================================================
# Load data
# ============================================================

print("\n" + "=" * 80)
print("MDD PUBLICATION FIGURES")
print("=" * 80)

for file_path in [
    PRS_FILE,
    POP_STATS_FILE,
    LABELED_FILE,
    DUNN_FILE
]:

    if not os.path.exists(file_path):

        raise FileNotFoundError(
            f"Required file not found:\n{file_path}"
        )


prs_df = pd.read_csv(
    PRS_FILE,
    sep="\t"
)

stats_df = pd.read_csv(
    POP_STATS_FILE,
    sep="\t"
)

labeled_df = pd.read_csv(
    LABELED_FILE,
    sep="\t"
)

dunn_df = pd.read_csv(
    DUNN_FILE,
    sep="\t",
    index_col=0
)


# ============================================================
# Validate PRS data
# ============================================================

required_prs_columns = [
    "IID",
    "PRS",
    "Risk_Group"
]

for column in required_prs_columns:

    if column not in prs_df.columns:

        raise ValueError(
            f"Missing column in PRS file: {column}"
        )


required_population_columns = [
    "IID",
    "PRS",
    "Superpopulation"
]

for column in required_population_columns:

    if column not in labeled_df.columns:

        raise ValueError(
            f"Missing column in population file: {column}"
        )


if len(prs_df) != 2504:

    raise ValueError(
        f"Expected 2504 PRS records, "
        f"found {len(prs_df)}."
    )


if len(labeled_df) != 2504:

    raise ValueError(
        f"Expected 2504 population-labeled records, "
        f"found {len(labeled_df)}."
    )


prs_df["PRS"] = pd.to_numeric(
    prs_df["PRS"],
    errors="coerce"
)

labeled_df["PRS"] = pd.to_numeric(
    labeled_df["PRS"],
    errors="coerce"
)


# ============================================================
# FIGURE 1
# Population-specific PRS distributions
# ============================================================

print("\nCreating Figure 1...")

data = []

for population in POP_ORDER:

    values = labeled_df.loc[
        labeled_df["Superpopulation"] == population,
        "PRS"
    ].dropna()

    if len(values) == 0:

        raise ValueError(
            f"No PRS values found for {population}"
        )

    data.append(
        values.to_numpy()
    )


fig, ax = plt.subplots(
    figsize=(9, 6)
)


# New Matplotlib API:
# tick_labels instead of labels

ax.boxplot(
    data,
    tick_labels=POP_ORDER,
    patch_artist=False,
    showfliers=False
)


# Individual observations

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
        s=7,
        alpha=0.18
    )


ax.set_xlabel(
    "1000 Genomes Superpopulation"
)

ax.set_ylabel(
    "MDD Polygenic Risk Score (PRS)"
)

ax.set_title(
    "MDD PRS Distribution Across 1000 Genomes Superpopulations"
)

ax.grid(
    axis="y",
    alpha=0.25
)

save_figure(
    fig,
    "Figure1_MDD_PRS_by_Superpopulation.png"
)


# ============================================================
# FIGURE 2
# Mean PRS ± 95% CI
# ============================================================

print("\nCreating Figure 2...")

stats_plot = (
    stats_df
    .set_index("Superpopulation")
    .reindex(POP_ORDER)
    .reset_index()
)

if stats_plot["Mean"].isna().any():

    raise ValueError(
        "Missing population mean values."
    )


x = np.arange(
    len(stats_plot)
)

means = stats_plot[
    "Mean"
].to_numpy()

se = stats_plot[
    "SE"
].to_numpy()

ci95 = 1.96 * se


fig, ax = plt.subplots(
    figsize=(9, 6)
)


ax.errorbar(
    x,
    means,
    yerr=ci95,
    fmt="o",
    markersize=7,
    capsize=5,
    linewidth=1.5
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
    POP_ORDER
)

ax.set_xlabel(
    "1000 Genomes Superpopulation"
)

ax.set_ylabel(
    "Mean MDD PRS ± 95% CI"
)

ax.set_title(
    "Mean MDD PRS Across 1000 Genomes Superpopulations"
)

ax.grid(
    axis="y",
    alpha=0.25
)

save_figure(
    fig,
    "Figure2_MDD_Mean_PRS_95CI.png"
)


# ============================================================
# FIGURE 3
# Overall genome-wide PRS distribution
# ============================================================

print("\nCreating Figure 3...")

prs = prs_df[
    "PRS"
].dropna()

if len(prs) != 2504:

    raise ValueError(
        f"Expected 2504 non-missing PRS values, "
        f"found {len(prs)}."
    )


p20 = prs.quantile(
    0.20
)

p80 = prs.quantile(
    0.80
)


fig, ax = plt.subplots(
    figsize=(9, 6)
)


ax.hist(
    prs,
    bins=40,
    edgecolor="black",
    alpha=0.8
)


ax.axvline(
    p20,
    linestyle="--",
    linewidth=1.5,
    label=f"20th percentile ({p20:.3f})"
)


ax.axvline(
    p80,
    linestyle="--",
    linewidth=1.5,
    label=f"80th percentile ({p80:.3f})"
)


ax.set_xlabel(
    "Genome-wide MDD PRS"
)

ax.set_ylabel(
    "Number of Individuals"
)

ax.set_title(
    "Genome-wide MDD PRS Distribution"
)

ax.legend()

ax.grid(
    axis="y",
    alpha=0.25
)

save_figure(
    fig,
    "Figure3_MDD_Overall_PRS_Distribution.png"
)


# ============================================================
# FIGURE 4
# Relative PRS group counts
# ============================================================

print("\nCreating Figure 4...")

group_counts = (
    prs_df[
        "Risk_Group"
    ]
    .value_counts()
    .reindex(
        GROUP_ORDER,
        fill_value=0
    )
)


if group_counts.sum() != 2504:

    raise ValueError(
        "Risk-group counts do not sum to 2504."
    )


fig, ax = plt.subplots(
    figsize=(8, 6)
)


bars = ax.bar(
    GROUP_ORDER,
    group_counts.to_numpy()
)


ax.set_xlabel(
    "Relative PRS Group"
)

ax.set_ylabel(
    "Number of Individuals"
)

ax.set_title(
    "Relative MDD PRS Group Distribution"
)


for bar, count in zip(
    bars,
    group_counts.to_numpy()
):

    percentage = (
        count / len(prs_df)
    ) * 100

    ax.text(
        bar.get_x()
        + bar.get_width() / 2,
        bar.get_height(),
        f"{count}\n({percentage:.1f}%)",
        ha="center",
        va="bottom"
    )


ax.grid(
    axis="y",
    alpha=0.25
)


save_figure(
    fig,
    "Figure4_MDD_Relative_PRS_Groups.png"
)


# ============================================================
# FIGURE 5
# Dunn-Holm pairwise significance heatmap
# ============================================================

print("\nCreating Figure 5...")


# ------------------------------------------------------------
# Reorder p-value matrix
# ------------------------------------------------------------

try:

    dunn_matrix = dunn_df.loc[
        POP_ORDER,
        POP_ORDER
    ].astype(float)

except KeyError as error:

    raise ValueError(
        "Dunn-Holm matrix does not contain "
        "the expected population labels."
    ) from error


# ------------------------------------------------------------
# Convert p-values to -log10(p)
# ------------------------------------------------------------

with np.errstate(
    divide="ignore",
    invalid="ignore"
):

    neglog = -np.log10(
        dunn_matrix
    )


neglog = neglog.replace(
    [np.inf, -np.inf],
    50
)


# Cap visual scale at 50.
# This DOES NOT change the original p-values.

neglog = neglog.clip(
    upper=50
)


# ------------------------------------------------------------
# FIX:
# Create a writable NumPy copy
# ------------------------------------------------------------

neglog_array = (
    neglog.to_numpy()
    .copy()
)


# Set diagonal to zero

np.fill_diagonal(
    neglog_array,
    0
)


# ------------------------------------------------------------
# Create heatmap
# ------------------------------------------------------------

fig, ax = plt.subplots(
    figsize=(8, 7)
)


im = ax.imshow(
    neglog_array,
    aspect="auto"
)


ax.set_xticks(
    np.arange(
        len(POP_ORDER)
    )
)

ax.set_yticks(
    np.arange(
        len(POP_ORDER)
    )
)

ax.set_xticklabels(
    POP_ORDER
)

ax.set_yticklabels(
    POP_ORDER
)


ax.set_xlabel(
    "Superpopulation"
)

ax.set_ylabel(
    "Superpopulation"
)

ax.set_title(
    "Pairwise MDD PRS Differences\n"
    "Dunn Test with Holm Correction"
)


# ------------------------------------------------------------
# Add p-values
# ------------------------------------------------------------

for i in range(
    len(POP_ORDER)
):

    for j in range(
        len(POP_ORDER)
    ):

        if i == j:

            text = "—"

        else:

            p = dunn_matrix.iloc[
                i,
                j
            ]

            if p < 0.001:

                text = "p < 0.001"

            else:

                text = f"p = {p:.3f}"


        ax.text(
            j,
            i,
            text,
            ha="center",
            va="center",
            fontsize=8
        )


cbar = fig.colorbar(
    im,
    ax=ax
)


cbar.set_label(
    "−log10(Holm-adjusted p)"
)


save_figure(
    fig,
    "Figure5_MDD_Dunn_Holm_Heatmap.png"
)


# ============================================================
# FIGURE 6
# 1000 Genomes population composition
# ============================================================

print("\nCreating Figure 6...")


population_counts = (
    labeled_df[
        "Superpopulation"
    ]
    .value_counts()
    .reindex(
        POP_ORDER,
        fill_value=0
    )
)


if population_counts.sum() != 2504:

    raise ValueError(
        "Population counts do not sum to 2504."
    )


fig, ax = plt.subplots(
    figsize=(8, 6)
)


bars = ax.bar(
    POP_ORDER,
    population_counts.to_numpy()
)


ax.set_xlabel(
    "1000 Genomes Superpopulation"
)

ax.set_ylabel(
    "Number of Individuals"
)

ax.set_title(
    "1000 Genomes Sample Composition"
)


for bar, count in zip(
    bars,
    population_counts.to_numpy()
):

    ax.text(
        bar.get_x()
        + bar.get_width() / 2,
        bar.get_height(),
        f"{count:,}",
        ha="center",
        va="bottom"
    )


ax.grid(
    axis="y",
    alpha=0.25
)


save_figure(
    fig,
    "Figure6_MDD_Population_Composition.png"
)


# ============================================================
# FINAL REPORT
# ============================================================

print("\n" + "=" * 80)
print("MDD PUBLICATION FIGURES COMPLETE")
print("=" * 80)

print(
    f"\nFigures saved to:\n"
    f"{OUTPUT_DIR}"
)

print("\nGenerated files:")

figure_files = [
    "Figure1_MDD_PRS_by_Superpopulation.png",
    "Figure2_MDD_Mean_PRS_95CI.png",
    "Figure3_MDD_Overall_PRS_Distribution.png",
    "Figure4_MDD_Relative_PRS_Groups.png",
    "Figure5_MDD_Dunn_Holm_Heatmap.png",
    "Figure6_MDD_Population_Composition.png"
]

for filename in figure_files:

    full_path = os.path.join(
        OUTPUT_DIR,
        filename
    )

    if os.path.exists(full_path):

        print(
            f"  ✓ {filename}"
        )

    else:

        print(
            f"  ✗ {filename} MISSING"
        )


print("\n" + "=" * 80)
