from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

ROOT = Path(r"C:\PRS_Study")

RESULTS = ROOT / "results" / "T2D"

OUT = (
    RESULTS
    / "population_analysis"
    / "publication_figures"
)

OUT.mkdir(parents=True, exist_ok=True)


# ============================================================
# INPUT FILES
# ============================================================

prs_file = (
    RESULTS
    / "T2D_genomewide_PRS_with_risk_groups.tsv"
)

pop_file = (
    RESULTS
    / "T2D_PRS_with_population_labels.tsv"
)

stats_file = (
    RESULTS
    / "T2D_population_PRS_by_superpopulation.tsv"
)

dunn_file = (
    RESULTS
    / "T2D_population_Dunn_Holm.tsv"
)


# ============================================================
# CHECK FILES
# ============================================================

print("=" * 70)
print("T2D PUBLICATION FIGURE GENERATION")
print("=" * 70)

print("\nChecking input files...")

for f in [
    prs_file,
    pop_file,
    stats_file,
    dunn_file
]:

    if not f.exists():

        raise FileNotFoundError(
            f"\nMissing required file:\n{f}"
        )

    print(f"OK: {f.name}")


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading data...")

prs = pd.read_csv(
    prs_file,
    sep="\t"
)

pop = pd.read_csv(
    pop_file,
    sep="\t"
)

stats = pd.read_csv(
    stats_file,
    sep="\t"
)

dunn = pd.read_csv(
    dunn_file,
    sep="\t"
)

print(f"Genome-wide PRS rows: {len(prs):,}")
print(f"Population-labelled rows: {len(pop):,}")
print(f"Population statistics rows: {len(stats):,}")
print(f"Dunn-Holm rows: {len(dunn):,}")


# ============================================================
# IDENTIFY PRS COLUMN
# ============================================================

if "PRS" in prs.columns:

    prs_col = "PRS"

else:

    possible = [
        c for c in prs.columns
        if "prs" in c.lower()
    ]

    if not possible:

        raise ValueError(
            "Could not identify PRS column.\n"
            f"Available columns: {prs.columns.tolist()}"
        )

    prs_col = possible[0]


if "PRS" in pop.columns:

    pop_prs_col = "PRS"

else:

    possible = [
        c for c in pop.columns
        if "prs" in c.lower()
    ]

    if not possible:

        raise ValueError(
            "Could not identify population PRS column.\n"
            f"Available columns: {pop.columns.tolist()}"
        )

    pop_prs_col = possible[0]


# ============================================================
# IDENTIFY SUPERPOPULATION COLUMN
# ============================================================

pop_col = None

for c in [
    "super_pop",
    "Superpopulation",
    "superpopulation",
    "SUPER_POP"
]:

    if c in pop.columns:

        pop_col = c
        break


if pop_col is None:

    raise ValueError(
        "Could not identify superpopulation column.\n"
        f"Available columns: {pop.columns.tolist()}"
    )


# ============================================================
# CONVERT PRS TO NUMERIC
# ============================================================

prs[prs_col] = pd.to_numeric(
    prs[prs_col],
    errors="coerce"
)

pop[pop_prs_col] = pd.to_numeric(
    pop[pop_prs_col],
    errors="coerce"
)


# ============================================================
# POPULATION ORDER
# ============================================================

ORDER = [
    "AFR",
    "AMR",
    "EAS",
    "EUR",
    "SAS"
]


# ============================================================
# PREPARE POPULATION GROUPS
# ============================================================

groups = []

for population in ORDER:

    values = (
        pop.loc[
            pop[pop_col] == population,
            pop_prs_col
        ]
        .dropna()
        .to_numpy()
    )

    groups.append(values)

    print(
        f"{population}: "
        f"N = {len(values):,}"
    )


# ============================================================
# FIGURE 1
# POPULATION-SPECIFIC PRS DISTRIBUTIONS
# ============================================================

print("\nGenerating Figure 1...")

fig, ax = plt.subplots(
    figsize=(9, 6)
)

ax.boxplot(
    groups,
    tick_labels=ORDER,
    showfliers=False
)

rng = np.random.default_rng(42)

for i, values in enumerate(
    groups,
    start=1
):

    jitter = rng.uniform(
        -0.10,
        0.10,
        len(values)
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
    "T2D Polygenic Risk Score"
)

ax.set_title(
    "T2D Polygenic Risk Score Distributions by Superpopulation"
)

ax.grid(
    axis="y",
    alpha=0.25
)

fig.tight_layout()

fig.savefig(
    OUT
    / "Figure1_T2D_population_PRS_distributions.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)

print("Figure 1 saved.")


# ============================================================
# FIGURE 2
# POPULATION MEAN ± 95% CI
# ============================================================

print("\nGenerating Figure 2...")

print(
    "Population statistics columns:"
)

print(
    stats.columns.tolist()
)


# Identify population column
stats_pop_col = stats.columns[0]


# Identify mean column
mean_col = None

for c in stats.columns:

    if c.lower() == "mean":

        mean_col = c
        break


# Identify standard error column
se_col = None

for c in stats.columns:

    cl = c.lower()

    if cl in [
        "se",
        "sem",
        "standard_error",
        "standard error"
    ]:

        se_col = c
        break


if mean_col is None:

    raise ValueError(
        "Could not identify Mean column."
    )


if se_col is None:

    raise ValueError(
        "Could not identify SE column."
    )


stats2 = stats.copy()

stats2[mean_col] = pd.to_numeric(
    stats2[mean_col],
    errors="coerce"
)

stats2[se_col] = pd.to_numeric(
    stats2[se_col],
    errors="coerce"
)

stats2 = (
    stats2
    .set_index(stats_pop_col)
    .reindex(ORDER)
)


means = (
    stats2[mean_col]
    .to_numpy(dtype=float)
)

se = (
    stats2[se_col]
    .to_numpy(dtype=float)
)

ci95 = 1.96 * se

x = np.arange(
    len(ORDER)
)


fig, ax = plt.subplots(
    figsize=(9, 6)
)

ax.errorbar(
    x,
    means,
    yerr=ci95,
    fmt="o",
    capsize=5,
    markersize=7
)

ax.axhline(
    0,
    linewidth=0.8
)

ax.set_xticks(x)

ax.set_xticklabels(
    ORDER
)

ax.set_xlabel(
    "1000 Genomes Superpopulation"
)

ax.set_ylabel(
    "Mean T2D PRS (95% CI)"
)

ax.set_title(
    "Mean T2D Polygenic Risk Score by Superpopulation"
)

ax.grid(
    axis="y",
    alpha=0.25
)

fig.tight_layout()

fig.savefig(
    OUT
    / "Figure2_T2D_population_mean_95CI.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)

print("Figure 2 saved.")


# ============================================================
# FIGURE 3
# OVERALL PRS DISTRIBUTION
# ============================================================

print("\nGenerating Figure 3...")

values = (
    prs[prs_col]
    .dropna()
    .to_numpy()
)

mean_prs = np.mean(values)

median_prs = np.median(values)

sd_prs = np.std(
    values,
    ddof=1
)


fig, ax = plt.subplots(
    figsize=(9, 6)
)

ax.hist(
    values,
    bins=35,
    edgecolor="black"
)

ax.axvline(
    mean_prs,
    linestyle="--",
    linewidth=1.2,
    label=f"Mean = {mean_prs:.2f}"
)

ax.axvline(
    median_prs,
    linestyle=":",
    linewidth=1.2,
    label=f"Median = {median_prs:.2f}"
)

ax.set_xlabel(
    "Genome-wide T2D PRS"
)

ax.set_ylabel(
    "Number of Individuals"
)

ax.set_title(
    "Overall Distribution of T2D Polygenic Risk Scores"
)

ax.legend()

ax.grid(
    axis="y",
    alpha=0.25
)

fig.tight_layout()

fig.savefig(
    OUT
    / "Figure3_T2D_overall_PRS_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)

print("Figure 3 saved.")


# ============================================================
# FIGURE 4
# RELATIVE PRS GROUP DISTRIBUTION
# ============================================================

print("\nGenerating Figure 4...")


# Find risk-group column
group_col = None

for c in prs.columns:

    if c.lower() in [
        "risk_group",
        "relative_prs_group",
        "relative risk group",
        "group"
    ]:

        group_col = c
        break


# Calculate current empirical thresholds
p20, p80 = np.percentile(
    values,
    [20, 80]
)


if group_col is None:

    prs["_relative_group"] = np.select(

        [
            prs[prs_col] <= p20,
            prs[prs_col] >= p80
        ],

        [
            "Low",
            "High"
        ],

        default="Intermediate"
    )

    group_col = "_relative_group"


counts = (
    prs[group_col]
    .value_counts()
    .reindex(
        [
            "Low",
            "Intermediate",
            "High"
        ]
    )
    .fillna(0)
)


fig, ax = plt.subplots(
    figsize=(8, 6)
)

ax.bar(
    counts.index,
    counts.values,
    edgecolor="black"
)

ax.set_xlabel(
    "Relative PRS Group"
)

ax.set_ylabel(
    "Number of Individuals"
)

ax.set_title(
    "Relative T2D PRS Group Distribution"
)

ax.grid(
    axis="y",
    alpha=0.25
)


for i, v in enumerate(
    counts.values
):

    ax.text(
        i,
        v + max(counts.values) * 0.015,
        f"{int(v)}",
        ha="center"
    )


fig.tight_layout()

fig.savefig(
    OUT
    / "Figure4_T2D_relative_PRS_groups.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)


print("Figure 4 saved.")

print(
    f"Current P20 = {p20:.6f}"
)

print(
    f"Current P80 = {p80:.6f}"
)

print("\nRelative group counts:")

print(counts)


# ============================================================
# FIGURE 5
# DUNN-HOLM PAIRWISE COMPARISONS
# ============================================================

print("\nGenerating Figure 5...")

print(
    "Dunn-Holm columns:"
)

print(
    dunn.columns.tolist()
)


# Your file uses:
#
# Group1
# Group2
# Z
# Raw_P
# Holm_Adjusted_P
# Significant


required = [
    "Group1",
    "Group2",
    "Holm_Adjusted_P"
]


for c in required:

    if c not in dunn.columns:

        raise ValueError(
            f"Missing Dunn-Holm column: {c}\n"
            f"Available columns: {dunn.columns.tolist()}"
        )


# Create 5 × 5 matrix
matrix = np.full(
    (5, 5),
    np.nan
)

np.fill_diagonal(
    matrix,
    0
)


# Fill matrix
for _, row in dunn.iterrows():

    a = str(
        row["Group1"]
    ).strip()

    b = str(
        row["Group2"]
    ).strip()


    if a not in ORDER:

        continue

    if b not in ORDER:

        continue


    try:

        p = float(
            row["Holm_Adjusted_P"]
        )

    except:

        continue


    i = ORDER.index(a)

    j = ORDER.index(b)


    matrix[i, j] = p

    matrix[j, i] = p


# ============================================================
# CONVERT TO -LOG10 P
# ============================================================

display_matrix = np.full_like(
    matrix,
    np.nan
)


for i in range(5):

    for j in range(5):

        p = matrix[i, j]


        if i == j:

            display_matrix[i, j] = 0


        elif np.isfinite(p):

            if p <= 0:

                display_matrix[i, j] = 50

            else:

                display_matrix[i, j] = min(
                    -np.log10(p),
                    50
                )


# ============================================================
# DRAW HEATMAP
# ============================================================

fig, ax = plt.subplots(
    figsize=(8, 7)
)

im = ax.imshow(
    display_matrix,
    aspect="equal"
)


ax.set_xticks(
    range(5)
)

ax.set_yticks(
    range(5)
)

ax.set_xticklabels(
    ORDER
)

ax.set_yticklabels(
    ORDER
)


ax.set_xlabel(
    "Superpopulation"
)

ax.set_ylabel(
    "Superpopulation"
)

ax.set_title(
    "Dunn–Holm Pairwise Comparisons of T2D PRS"
)


# Add p-values
for i in range(5):

    for j in range(5):

        if i == j:

            text = "—"

        elif np.isfinite(
            matrix[i, j]
        ):

            p = matrix[i, j]

            if (
                p <= 0
                or p < 1e-50
            ):

                text = "p<1e-50"

            else:

                text = f"{p:.2e}"

        else:

            text = "NA"


        ax.text(
            j,
            i,
            text,
            ha="center",
            va="center",
            fontsize=8
        )


fig.colorbar(
    im,
    ax=ax,
    label="−log10(adjusted p), capped at 50"
)


fig.tight_layout()

fig.savefig(
    OUT
    / "Figure5_T2D_Dunn_Holm_heatmap.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)

print("Figure 5 saved.")


# ============================================================
# FIGURE 6
# 1000 GENOMES POPULATION COMPOSITION
# ============================================================

print("\nGenerating Figure 6...")


pop_counts = (
    pop[pop_col]
    .value_counts()
    .reindex(ORDER)
    .fillna(0)
)


fig, ax = plt.subplots(
    figsize=(8, 6)
)

ax.bar(
    pop_counts.index,
    pop_counts.values,
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


for i, v in enumerate(
    pop_counts.values
):

    ax.text(
        i,
        v + max(pop_counts.values) * 0.015,
        f"{int(v)}",
        ha="center"
    )


fig.tight_layout()

fig.savefig(
    OUT
    / "Figure6_T2D_population_composition.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)

print("Figure 6 saved.")


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("T2D PUBLICATION FIGURES COMPLETE")
print("=" * 70)

print("\nOutput folder:")
print(OUT)

print("\nGenerated figures:")

for f in sorted(
    OUT.glob("Figure*.png")
):

    print(
        f"  {f.name}"
    )


print("\nCurrent genome-wide T2D PRS:")

print(
    f"N      = {len(values)}"
)

print(
    f"Mean   = {mean_prs:.6f}"
)

print(
    f"SD     = {sd_prs:.6f}"
)

print(
    f"Median = {median_prs:.6f}"
)

print(
    f"P20    = {p20:.6f}"
)

print(
    f"P80    = {p80:.6f}"
)


print("\nPopulation counts:")

print(
    pop_counts
)

print("\nDone.")
