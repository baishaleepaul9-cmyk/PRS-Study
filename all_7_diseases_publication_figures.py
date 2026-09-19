import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# ============================================================
# PATHS
# ============================================================

INPUT_FILE = (
    r"C:\PRS_Study\results\combined_analysis"
    r"\all_7_diseases_standardized_PRS_with_population.tsv"
)

OUTPUT_DIR = (
    r"C:\PRS_Study\results\combined_analysis"
    r"\standardized_population_analysis\publication_figures"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_FILE, sep="\t")

print("=" * 70)
print("PUBLICATION FIGURE GENERATION")
print("=" * 70)

print(f"Input: {INPUT_FILE}")
print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")


# ============================================================
# DISEASES
# ============================================================

disease_columns = {
    "Alzheimer's disease": "Alzheimers_Z",
    "Schizophrenia": "Schizophrenia_Z",
    "Multiple sclerosis": "Multiple_Sclerosis_Z",
    "Major depressive disorder": "MDD_Z",
    "Parkinson's disease": "Parkinsons_Z",
    "Coronary artery disease": "CAD_Z",
    "Type 2 diabetes": "T2D_Z",
}

populations = ["AFR", "AMR", "EAS", "EUR", "SAS"]


# ============================================================
# POPULATION VISUAL SETTINGS
# Grayscale + different hatch patterns for print friendliness
# ============================================================

population_styles = {
    "AFR": {"facecolor": "0.90", "hatch": "///"},
    "AMR": {"facecolor": "0.75", "hatch": "\\\\"},
    "EAS": {"facecolor": "0.60", "hatch": "..."},
    "EUR": {"facecolor": "0.45", "hatch": "---"},
    "SAS": {"facecolor": "0.80", "hatch": "xxx"},
}


# ============================================================
# FIGURE 1
# POLISHED POPULATION DISTRIBUTIONS
# ============================================================

fig, ax = plt.subplots(figsize=(15, 7.5))

positions = []
box_data = []

# Disease centers
disease_centers = []

# Width of each box
box_width = 0.14

for disease_index, (disease, column) in enumerate(disease_columns.items()):

    # Center of each disease group
    center = disease_index * 1.25 + 1

    disease_centers.append(center)

    # Spread five populations around the disease center
    offsets = np.linspace(-0.28, 0.28, len(populations))

    for pop, offset in zip(populations, offsets):

        values = df.loc[
            df["Superpopulation"] == pop,
            column
        ].dropna().values

        box_data.append(values)
        positions.append(center + offset)


# Create all boxes
bp = ax.boxplot(
    box_data,
    positions=positions,
    widths=box_width,
    patch_artist=True,
    showfliers=False,
    whis=1.5,
    medianprops=dict(
        linewidth=1.5,
        color="black"
    ),
    whiskerprops=dict(
        linewidth=0.9,
        color="black"
    ),
    capprops=dict(
        linewidth=0.9,
        color="black"
    ),
    boxprops=dict(
        linewidth=0.9,
        color="black"
    )
)


# Apply population-specific patterns
box_index = 0

for disease in disease_columns:

    for pop in populations:

        style = population_styles[pop]

        bp["boxes"][box_index].set_facecolor(style["facecolor"])
        bp["boxes"][box_index].set_hatch(style["hatch"])
        bp["boxes"][box_index].set_edgecolor("black")
        bp["boxes"][box_index].set_linewidth(0.9)

        box_index += 1


# Zero reference line
ax.axhline(
    0,
    linestyle="--",
    linewidth=1.0,
    color="black",
    alpha=0.65,
    zorder=0
)


# Disease labels
ax.set_xticks(disease_centers)
ax.set_xticklabels(
    list(disease_columns.keys()),
    rotation=25,
    ha="right",
    fontsize=10
)


# Axis labels
ax.set_ylabel(
    "Standardized PRS (Z-score)",
    fontsize=11
)

ax.set_xlabel(
    "Disease",
    fontsize=11
)


# Title
ax.set_title(
    "Population Distribution of Standardized Polygenic Risk Scores",
    fontsize=14,
    pad=12
)


# Grid
ax.yaxis.grid(
    True,
    linestyle=":",
    linewidth=0.7,
    alpha=0.45
)

ax.set_axisbelow(True)


# Population legend
legend_handles = []

for pop in populations:

    style = population_styles[pop]

    legend_handles.append(
        Patch(
            facecolor=style["facecolor"],
            edgecolor="black",
            hatch=style["hatch"],
            label=pop
        )
    )


ax.legend(
    handles=legend_handles,
    title="1000 Genomes Superpopulation",
    loc="upper left",
    frameon=True,
    fontsize=9,
    title_fontsize=9
)


# Clean borders
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)


plt.tight_layout()

figure1_path = os.path.join(
    OUTPUT_DIR,
    "Figure1_Standardized_PRS_Population_Distributions.png"
)

plt.savefig(
    figure1_path,
    dpi=600,
    bbox_inches="tight"
)

plt.close()

print("\nFigure 1 replaced:")
print(figure1_path)


# ============================================================
# FIGURE 2
# MEAN STANDARDIZED PRS Â± 95% CI
# ============================================================

summary_rows = []

for disease, column in disease_columns.items():

    for pop in populations:

        values = df.loc[
            df["Superpopulation"] == pop,
            column
        ].dropna()

        n = len(values)
        mean = values.mean()
        sd = values.std(ddof=1)
        se = sd / np.sqrt(n)

        ci_low = mean - 1.96 * se
        ci_high = mean + 1.96 * se

        summary_rows.append({
            "Disease": disease,
            "Population": pop,
            "N": n,
            "Mean_Z": mean,
            "SD": sd,
            "CI_Low": ci_low,
            "CI_High": ci_high
        })


summary = pd.DataFrame(summary_rows)


# Save figure statistics
stats_path = os.path.join(
    OUTPUT_DIR,
    "standardized_PRS_figure_statistics.tsv"
)

summary.to_csv(
    stats_path,
    sep="\t",
    index=False
)


fig, ax = plt.subplots(figsize=(15, 7.5))

x = np.arange(len(disease_columns))

markers = {
    "AFR": "o",
    "AMR": "s",
    "EAS": "^",
    "EUR": "D",
    "SAS": "P",
}

for pop in populations:

    sub = summary[
        summary["Population"] == pop
    ]

    means = sub["Mean_Z"].values
    lower = sub["CI_Low"].values
    upper = sub["CI_High"].values

    yerr = np.vstack([
        means - lower,
        upper - means
    ])

    ax.errorbar(
        x,
        means,
        yerr=yerr,
        marker=markers[pop],
        linestyle="none",
        capsize=3,
        markersize=5,
        linewidth=1
    )


ax.axhline(
    0,
    linestyle="--",
    linewidth=1.0,
    color="black",
    alpha=0.65
)

ax.set_xticks(x)

ax.set_xticklabels(
    list(disease_columns.keys()),
    rotation=25,
    ha="right",
    fontsize=10
)

ax.set_ylabel(
    "Mean standardized PRS (Z-score)",
    fontsize=11
)

ax.set_xlabel(
    "Disease",
    fontsize=11
)

ax.set_title(
    "Mean Standardized PRS Across 1000 Genomes Superpopulations",
    fontsize=14,
    pad=12
)

ax.yaxis.grid(
    True,
    linestyle=":",
    linewidth=0.7,
    alpha=0.45
)

ax.set_axisbelow(True)

ax.legend(
    populations,
    title="Superpopulation",
    loc="best",
    fontsize=9,
    title_fontsize=9
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()

figure2_path = os.path.join(
    OUTPUT_DIR,
    "Figure2_Mean_Standardized_PRS_95CI.png"
)

plt.savefig(
    figure2_path,
    dpi=600,
    bbox_inches="tight"
)

plt.close()

print("Figure 2 regenerated:")
print(figure2_path)


# ============================================================
# FIGURE 3
# DISEASE Ã— POPULATION HEATMAP
# ============================================================

disease_order = list(disease_columns.keys())

heatmap_data = summary.pivot(
    index="Disease",
    columns="Population",
    values="Mean_Z"
)

heatmap_data = heatmap_data.loc[
    disease_order,
    populations
]

fig, ax = plt.subplots(figsize=(9, 7))

im = ax.imshow(
    heatmap_data.values,
    aspect="auto"
)

ax.set_xticks(
    np.arange(len(populations))
)

ax.set_xticklabels(
    populations,
    fontsize=10
)

ax.set_yticks(
    np.arange(len(heatmap_data.index))
)

ax.set_yticklabels(
    heatmap_data.index,
    fontsize=9
)

ax.set_xlabel(
    "1000 Genomes Superpopulation",
    fontsize=11
)

ax.set_ylabel(
    "Disease",
    fontsize=11
)

ax.set_title(
    "Mean Standardized PRS Across Diseases and Populations",
    fontsize=14,
    pad=12
)


# Numeric annotations
for i in range(heatmap_data.shape[0]):

    for j in range(heatmap_data.shape[1]):

        value = heatmap_data.iloc[i, j]

        ax.text(
            j,
            i,
            f"{value:.2f}",
            ha="center",
            va="center",
            fontsize=9
        )


cbar = fig.colorbar(
    im,
    ax=ax
)

cbar.set_label(
    "Mean standardized PRS (Z-score)",
    fontsize=10
)

plt.tight_layout()

figure3_path = os.path.join(
    OUTPUT_DIR,
    "Figure3_Standardized_PRS_Heatmap.png"
)

plt.savefig(
    figure3_path,
    dpi=600,
    bbox_inches="tight"
)

plt.close()

print("Figure 3 regenerated:")
print(figure3_path)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("FIGURE GENERATION COMPLETE")
print("=" * 70)

print("\nFigures created/replaced:")

print(figure1_path)
print(figure2_path)
print(figure3_path)

print("\nStatistics:")
print(stats_path)

print("\nSTATUS: 3 PUBLICATION FIGURES CREATED")
print("=" * 70)

