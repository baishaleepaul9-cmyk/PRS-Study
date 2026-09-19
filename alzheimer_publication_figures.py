import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


# ============================================================
# ALZHEIMER'S PRS — PUBLICATION FIGURES
# ============================================================
#
# Generates:
#
# Figure 1:
#   Population-specific PRS boxplot + individual observations
#
# Figure 2:
#   Mean PRS bar graph + 95% confidence intervals
#
# Figure 3:
#   Overall Alzheimer's PRS distribution
#
# Figure 4:
#   Relative PRS group counts
#
# Figure 5:
#   Dunn-Holm adjusted p-value heatmap
#
# Figure 6:
#   1000 Genomes population sample sizes
#
# IMPORTANT:
# - Reads existing results only
# - Does NOT recalculate PRS
# - Does NOT run PLINK
# - Does NOT process VCF files
# - Does NOT modify existing analysis files
# ============================================================


# ============================================================
# 1. PATHS
# ============================================================

DATA_FILE = (
    r".\results\Alzheimer\population_analysis"
    r"\Alzheimer_PRS_with_population_labels.tsv"
)

DUNN_FILE = (
    r".\results\Alzheimer\population_analysis"
    r"\Alzheimer_population_Dunn_Holm_pvalues.tsv"
)

OUT_DIR = (
    r".\results\Alzheimer\population_analysis"
    r"\publication_figures"
)

os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================
# 2. SETTINGS
# ============================================================

PRS_COL = "ALZHEIMERS_PRS"
POP_COL = "super_pop"

POPULATIONS = [
    "AFR",
    "AMR",
    "EAS",
    "EUR",
    "SAS"
]

POPULATION_LABELS = {
    "AFR": "AFR",
    "AMR": "AMR",
    "EAS": "EAS",
    "EUR": "EUR",
    "SAS": "SAS"
}


# ============================================================
# 3. LOAD DATA
# ============================================================

print("=" * 75)
print("ALZHEIMER'S PRS PUBLICATION FIGURES")
print("=" * 75)

if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(DATA_FILE)

df = pd.read_csv(
    DATA_FILE,
    sep="\t"
)

print("\nDataset shape:", df.shape)

if PRS_COL not in df.columns:
    raise ValueError(
        f"Missing column: {PRS_COL}"
    )

if POP_COL not in df.columns:
    raise ValueError(
        f"Missing column: {POP_COL}"
    )

df[PRS_COL] = pd.to_numeric(
    df[PRS_COL],
    errors="coerce"
)

df[POP_COL] = (
    df[POP_COL]
    .astype(str)
    .str.strip()
)

df = df.dropna(
    subset=[PRS_COL, POP_COL]
)

print("Individuals:", len(df))
print("Missing PRS:", df[PRS_COL].isna().sum())
print(
    "Population counts:"
)
print(
    df[POP_COL]
    .value_counts()
    .sort_index()
)


# ============================================================
# FIGURE 1
# BOXplot + INDIVIDUAL OBSERVATIONS
# ============================================================

print("\nGenerating Figure 1...")

fig, ax = plt.subplots(
    figsize=(9, 6)
)

box_data = []

for pop in POPULATIONS:

    values = df.loc[
        df[POP_COL] == pop,
        PRS_COL
    ].values

    box_data.append(values)


# Boxplot
ax.boxplot(
    box_data,
    tick_labels=[
        f"{p}\n(n={len(v)})"
        for p, v in zip(
            POPULATIONS,
            box_data
        )
    ],
    showfliers=False,
    widths=0.55,
    patch_artist=False
)


# Individual observations
rng = np.random.default_rng(42)

for i, values in enumerate(box_data, start=1):

    jitter = rng.uniform(
        -0.10,
        0.10,
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
    "1000 Genomes super-population",
    fontsize=12
)

ax.set_ylabel(
    "Alzheimer's PRS",
    fontsize=12
)

ax.set_title(
    "Alzheimer's PRS Across 1000 Genomes Super-populations",
    fontsize=14,
    pad=12
)

ax.tick_params(
    axis="both",
    labelsize=10
)

ax.grid(
    axis="y",
    alpha=0.25
)

fig.tight_layout()

figure1 = os.path.join(
    OUT_DIR,
    "Figure1_Alzheimer_PRS_population_boxplot_points.png"
)

fig.savefig(
    figure1,
    dpi=600,
    bbox_inches="tight"
)

plt.close(fig)

print("Saved:", figure1)


# ============================================================
# FIGURE 2
# MEAN PRS BAR GRAPH + 95% CI
# ============================================================

print("\nGenerating Figure 2...")

summary_rows = []

for pop in POPULATIONS:

    values = df.loc[
        df[POP_COL] == pop,
        PRS_COL
    ].values

    n = len(values)

    mean = np.mean(values)

    sd = np.std(
        values,
        ddof=1
    )

    se = sd / np.sqrt(n)

    # 95% CI using t distribution
    t_critical = stats.t.ppf(
        0.975,
        df=n - 1
    )

    ci = t_critical * se

    summary_rows.append(
        {
            "Population": pop,
            "N": n,
            "Mean": mean,
            "SD": sd,
            "SE": se,
            "CI95": ci
        }
    )


summary = pd.DataFrame(
    summary_rows
)


fig, ax = plt.subplots(
    figsize=(9, 6)
)

x = np.arange(
    len(POPULATIONS)
)

ax.bar(
    x,
    summary["Mean"],
    yerr=summary["CI95"],
    capsize=5,
    width=0.62
)

ax.axhline(
    0,
    linewidth=0.8
)

ax.set_xticks(x)

ax.set_xticklabels(
    [
        f"{p}\n(n={n})"
        for p, n in zip(
            summary["Population"],
            summary["N"]
        )
    ]
)

ax.set_xlabel(
    "1000 Genomes super-population",
    fontsize=12
)

ax.set_ylabel(
    "Mean Alzheimer's PRS ± 95% CI",
    fontsize=12
)

ax.set_title(
    "Mean Alzheimer's PRS Across Super-populations",
    fontsize=14,
    pad=12
)

ax.tick_params(
    axis="both",
    labelsize=10
)

ax.grid(
    axis="y",
    alpha=0.25
)

fig.tight_layout()

figure2 = os.path.join(
    OUT_DIR,
    "Figure2_Alzheimer_PRS_population_mean_95CI.png"
)

fig.savefig(
    figure2,
    dpi=600,
    bbox_inches="tight"
)

plt.close(fig)

print("Saved:", figure2)


# ============================================================
# SAVE SUMMARY USED FOR FIGURE 2
# ============================================================

summary_file = os.path.join(
    OUT_DIR,
    "Alzheimer_population_figure_summary.tsv"
)

summary.to_csv(
    summary_file,
    sep="\t",
    index=False
)

print(
    "Saved:",
    summary_file
)


# ============================================================
# FIGURE 3
# OVERALL PRS DISTRIBUTION
# ============================================================

print("\nGenerating Figure 3...")

prs_values = df[PRS_COL].values

fig, ax = plt.subplots(
    figsize=(9, 6)
)

ax.hist(
    prs_values,
    bins=40,
    edgecolor="black",
    alpha=0.8
)

mean_prs = np.mean(prs_values)

median_prs = np.median(prs_values)

ax.axvline(
    mean_prs,
    linestyle="--",
    linewidth=1.5,
    label=f"Mean = {mean_prs:.3f}"
)

ax.axvline(
    median_prs,
    linestyle=":",
    linewidth=1.5,
    label=f"Median = {median_prs:.3f}"
)

ax.set_xlabel(
    "Alzheimer's PRS",
    fontsize=12
)

ax.set_ylabel(
    "Number of individuals",
    fontsize=12
)

ax.set_title(
    "Genome-wide Alzheimer's PRS Distribution",
    fontsize=14,
    pad=12
)

ax.legend(
    frameon=False,
    fontsize=10
)

ax.tick_params(
    axis="both",
    labelsize=10
)

ax.grid(
    axis="y",
    alpha=0.25
)

fig.tight_layout()

figure3 = os.path.join(
    OUT_DIR,
    "Figure3_Alzheimer_genomewide_PRS_distribution.png"
)

fig.savefig(
    figure3,
    dpi=600,
    bbox_inches="tight"
)

plt.close(fig)

print("Saved:", figure3)


# ============================================================
# FIGURE 4
# RELATIVE PRS GROUP COUNTS
# ============================================================

print("\nGenerating Figure 4...")

RISK_COL = "ALZHEIMERS_RISK_GROUP"

if RISK_COL in df.columns:

    risk_order = [
        "Low",
        "Intermediate",
        "High"
    ]

    risk_counts = (
        df[RISK_COL]
        .value_counts()
        .reindex(
            risk_order,
            fill_value=0
        )
    )

    risk_percent = (
        risk_counts /
        len(df) *
        100
    )

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    x = np.arange(
        len(risk_order)
    )

    bars = ax.bar(
        x,
        risk_counts.values,
        width=0.6
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        [
            f"{group}\n({risk_percent[group]:.1f}%)"
            for group in risk_order
        ]
    )

    ax.set_xlabel(
        "Relative PRS group",
        fontsize=12
    )

    ax.set_ylabel(
        "Number of individuals",
        fontsize=12
    )

    ax.set_title(
        "Relative Alzheimer's PRS Groups",
        fontsize=14,
        pad=12
    )

    ax.tick_params(
        axis="both",
        labelsize=10
    )

    ax.grid(
        axis="y",
        alpha=0.25
    )

    # Add counts above bars
    for bar, count in zip(
        bars,
        risk_counts.values
    ):

        ax.text(
            bar.get_x() +
            bar.get_width() / 2,
            bar.get_height(),
            str(count),
            ha="center",
            va="bottom",
            fontsize=10
        )

    fig.tight_layout()

    figure4 = os.path.join(
        OUT_DIR,
        "Figure4_Alzheimer_relative_PRS_groups.png"
    )

    fig.savefig(
        figure4,
        dpi=600,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("Saved:", figure4)

else:

    print(
        "Risk-group column not found; Figure 4 skipped."
    )


# ============================================================
# FIGURE 5
# DUNN-HOLM HEATMAP
# ============================================================

print("\nGenerating Figure 5...")

if os.path.exists(DUNN_FILE):

    dunn = pd.read_csv(
        DUNN_FILE,
        sep="\t",
        index_col=0
    )

    dunn = dunn.loc[
        POPULATIONS,
        POPULATIONS
    ]

    # Convert to numeric
    dunn = dunn.apply(
        pd.to_numeric,
        errors="coerce"
    )

    # Replace zero with smallest representable positive value
    dunn_for_plot = dunn.copy()

    dunn_for_plot = dunn_for_plot.mask(
        dunn_for_plot <= 0,
        np.nextafter(
            0,
            1
        )
    )

    # Convert p-values to -log10
    log_p = -np.log10(
        dunn_for_plot
    )

    # Mask diagonal and upper triangle
    mask = np.triu(
        np.ones_like(
            log_p,
            dtype=bool
        )
    )

    log_p_masked = np.ma.masked_where(
        mask,
        log_p.values
    )

    fig, ax = plt.subplots(
        figsize=(8, 7)
    )

    image = ax.imshow(
        log_p_masked,
        aspect="equal"
    )

    ax.set_xticks(
        np.arange(
            len(POPULATIONS)
        )
    )

    ax.set_yticks(
        np.arange(
            len(POPULATIONS)
        )
    )

    ax.set_xticklabels(
        POPULATIONS
    )

    ax.set_yticklabels(
        POPULATIONS
    )

    ax.set_xlabel(
        "1000 Genomes super-population",
        fontsize=11
    )

    ax.set_ylabel(
        "1000 Genomes super-population",
        fontsize=11
    )

    ax.set_title(
        "Dunn's Test with Holm Correction",
        fontsize=14,
        pad=12
    )

    colorbar = fig.colorbar(
        image,
        ax=ax
    )

    colorbar.set_label(
        "−log10(Holm-adjusted p-value)",
        fontsize=10
    )

    # Add adjusted p-values in lower triangle
    for i in range(
        len(POPULATIONS)
    ):

        for j in range(
            len(POPULATIONS)
        ):

            if i > j:

                p = dunn.iloc[
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

    fig.tight_layout()

    figure5 = os.path.join(
        OUT_DIR,
        "Figure5_Alzheimer_Dunn_Holm_heatmap.png"
    )

    fig.savefig(
        figure5,
        dpi=600,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("Saved:", figure5)

else:

    print(
        "Dunn-Holm file not found; Figure 5 skipped."
    )


# ============================================================
# FIGURE 6
# POPULATION SAMPLE SIZE
# ============================================================

print("\nGenerating Figure 6...")

population_counts = (
    df[POP_COL]
    .value_counts()
    .reindex(
        POPULATIONS,
        fill_value=0
    )
)

fig, ax = plt.subplots(
    figsize=(8, 6)
)

x = np.arange(
    len(POPULATIONS)
)

bars = ax.bar(
    x,
    population_counts.values,
    width=0.6
)

ax.set_xticks(x)

ax.set_xticklabels(
    POPULATIONS
)

ax.set_xlabel(
    "1000 Genomes super-population",
    fontsize=12
)

ax.set_ylabel(
    "Number of individuals",
    fontsize=12
)

ax.set_title(
    "1000 Genomes Sample Composition",
    fontsize=14,
    pad=12
)

ax.tick_params(
    axis="both",
    labelsize=10
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
        bar.get_x() +
        bar.get_width() / 2,
        bar.get_height(),
        str(count),
        ha="center",
        va="bottom",
        fontsize=10
    )

fig.tight_layout()

figure6 = os.path.join(
    OUT_DIR,
    "Figure6_1000G_population_sample_sizes.png"
)

fig.savefig(
    figure6,
    dpi=600,
    bbox_inches="tight"
)

plt.close(fig)

print("Saved:", figure6)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("FIGURE GENERATION COMPLETED")
print("=" * 75)

print("\nFigures saved to:")
print(OUT_DIR)

print("\nGenerated figures:")

print("1. Population PRS boxplot + individual observations")
print("2. Mean PRS bar graph + 95% CI")
print("3. Overall genome-wide PRS distribution")
print("4. Relative PRS group counts")
print("5. Dunn-Holm pairwise comparison heatmap")
print("6. 1000 Genomes population sample composition")

print("\nNo PRS values were recalculated.")
print("No VCF files were processed.")
print("No PLINK commands were executed.")

print("\nDone.")
