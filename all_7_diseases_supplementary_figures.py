import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# ============================================================
# PATHS
# ============================================================

BASE = r"C:\PRS_Study\results\combined_analysis"

INPUT_STANDARDIZED = os.path.join(
    BASE,
    "all_7_diseases_standardized_PRS_with_population.tsv"
)

DUNN_FILE = os.path.join(
    BASE,
    "standardized_population_analysis",
    "all_7_diseases_standardized_Dunn_Holm.tsv"
)

OUTPUT_DIR = os.path.join(
    BASE,
    "standardized_population_analysis",
    "publication_figures"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# DISEASES AND POPULATIONS
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
# LOAD STANDARDIZED DATA
# ============================================================

df = pd.read_csv(
    INPUT_STANDARDIZED,
    sep="\t"
)

print("=" * 70)
print("SUPPLEMENTARY FIGURE GENERATION")
print("=" * 70)

print(f"Input: {INPUT_STANDARDIZED}")
print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")


# ============================================================
# SUPPLEMENTARY FIGURE 1
# DUNN-HOLM PAIRWISE COMPARISON HEATMAPS
# ============================================================

dunn = pd.read_csv(
    DUNN_FILE,
    sep="\t"
)

print("\nDunn-Holm file loaded:")
print(f"Rows: {len(dunn)}")
print(f"Columns: {list(dunn.columns)}")


# ------------------------------------------------------------
# Identify required columns
# ------------------------------------------------------------

disease_col = next(
    c for c in dunn.columns
    if "disease" in c.lower()
)

population_cols = [
    c for c in dunn.columns
    if "population" in c.lower()
]

if len(population_cols) < 2:
    raise ValueError(
        "Could not identify the two population columns "
        "in the Dunn-Holm file."
    )

pop1_col = population_cols[0]
pop2_col = population_cols[1]


p_candidates = [
    c for c in dunn.columns
    if (
        "holm" in c.lower()
        or "adjust" in c.lower()
        or "p_value" in c.lower()
        or c.lower() == "p"
    )
]

if not p_candidates:
    raise ValueError(
        "Could not identify the adjusted p-value column."
    )

p_col = p_candidates[0]

print(f"Disease column: {disease_col}")
print(f"Population columns: {pop1_col}, {pop2_col}")
print(f"P-value column: {p_col}")


# ------------------------------------------------------------
# Create 7 disease panels
# ------------------------------------------------------------

fig, axes = plt.subplots(
    4,
    2,
    figsize=(12, 18)
)

axes = axes.flatten()

last_im = None

for idx, disease in enumerate(disease_columns):

    ax = axes[idx]

    sub = dunn[
        dunn[disease_col].astype(str).str.strip()
        == disease
    ].copy()

    matrix = pd.DataFrame(
        np.nan,
        index=populations,
        columns=populations
    )

    for _, row in sub.iterrows():

        p1 = str(row[pop1_col]).strip()
        p2 = str(row[pop2_col]).strip()

        if p1 not in populations or p2 not in populations:
            continue

        try:
            p = float(row[p_col])
        except (ValueError, TypeError):
            continue

        matrix.loc[p1, p2] = p
        matrix.loc[p2, p1] = p

    # Diagonal
    for pop in populations:
        matrix.loc[pop, pop] = 1.0

    values = matrix.values.astype(float)

    # --------------------------------------------------------
    # Convert p-values to -log10(p)
    # --------------------------------------------------------

    display_values = np.full_like(
        values,
        np.nan,
        dtype=float
    )

    valid = ~np.isnan(values)

    display_values[valid] = -np.log10(
        np.maximum(values[valid], 1e-300)
    )

    # Cap visual scale at 50
    display_values = np.minimum(
        display_values,
        50
    )

    last_im = ax.imshow(
        display_values,
        aspect="equal"
    )

    ax.set_xticks(
        np.arange(len(populations))
    )

    ax.set_xticklabels(
        populations,
        fontsize=9
    )

    ax.set_yticks(
        np.arange(len(populations))
    )

    ax.set_yticklabels(
        populations,
        fontsize=9
    )

    ax.set_title(
        disease,
        fontsize=11,
        pad=8
    )

    # --------------------------------------------------------
    # Annotate actual Holm-adjusted p-values
    # --------------------------------------------------------

    for i in range(len(populations)):

        for j in range(len(populations)):

            if i == j:

                text = "—"

            elif np.isnan(values[i, j]):

                text = "NA"

            else:

                p = values[i, j]

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
                fontsize=7
            )


# Remove unused eighth panel
axes[-1].axis("off")


fig.suptitle(
    "Dunn–Holm Pairwise Comparisons Across 1000 Genomes Superpopulations",
    fontsize=14,
    y=0.995
)


# Shared colorbar
if last_im is not None:

    cbar = fig.colorbar(
        last_im,
        ax=axes[:-1],
        fraction=0.025,
        pad=0.02
    )

    cbar.set_label(
        r"$-\log_{10}$(Holm-adjusted p-value)",
        fontsize=10
    )


fig.text(
    0.5,
    0.01,
    "Cells show Holm-adjusted p-values; color intensity represents statistical significance.",
    ha="center",
    fontsize=9
)


plt.tight_layout(
    rect=[0, 0.025, 1, 0.98]
)


figure1 = os.path.join(
    OUTPUT_DIR,
    "Supplementary_Figure1_Dunn_Holm_Pairwise_Comparisons.png"
)

plt.savefig(
    figure1,
    dpi=600,
    bbox_inches="tight"
)

plt.close()

print("\nCreated:")
print(figure1)


# ============================================================
# SUPPLEMENTARY FIGURE 2
# OVERALL STANDARDIZED PRS DISTRIBUTIONS
# ============================================================

fig, ax = plt.subplots(
    figsize=(13, 7)
)

data = []

for disease, column in disease_columns.items():

    values = (
        df[column]
        .dropna()
        .values
    )

    data.append(values)


# IMPORTANT:
# Matplotlib 3.14 uses tick_labels instead of labels.

bp = ax.boxplot(
    data,
    tick_labels=list(disease_columns.keys()),
    patch_artist=True,
    showfliers=False,
    widths=0.60,
    medianprops=dict(
        color="black",
        linewidth=1.5
    ),
    boxprops=dict(
        color="black",
        linewidth=1
    ),
    whiskerprops=dict(
        color="black",
        linewidth=1
    ),
    capprops=dict(
        color="black",
        linewidth=1
    )
)


# ------------------------------------------------------------
# Grayscale + hatch patterns
# ------------------------------------------------------------

hatches = [
    "",
    "///",
    "\\\\",
    "...",
    "---",
    "xxx",
    "+++"
]

for box, hatch in zip(
    bp["boxes"],
    hatches
):

    box.set_facecolor("0.75")
    box.set_hatch(hatch)
    box.set_edgecolor("black")


ax.axhline(
    0,
    linestyle="--",
    linewidth=1,
    color="black",
    alpha=0.6
)


ax.set_ylabel(
    "Standardized PRS (Z-score)",
    fontsize=11
)

ax.set_xlabel(
    "Disease",
    fontsize=11
)

ax.set_title(
    "Overall Distribution of Standardized Polygenic Risk Scores",
    fontsize=14,
    pad=12
)


ax.set_xticklabels(
    list(disease_columns.keys()),
    rotation=25,
    ha="right"
)


ax.yaxis.grid(
    True,
    linestyle=":",
    linewidth=0.7,
    alpha=0.45
)

ax.set_axisbelow(True)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)


plt.tight_layout()


figure2 = os.path.join(
    OUTPUT_DIR,
    "Supplementary_Figure2_Overall_Standardized_PRS_Distributions.png"
)

plt.savefig(
    figure2,
    dpi=600,
    bbox_inches="tight"
)

plt.close()

print("\nCreated:")
print(figure2)


# ============================================================
# SUPPLEMENTARY FIGURE 3
# RELATIVE PRS GROUP DISTRIBUTION
# ============================================================

group_rows = []


for disease, column in disease_columns.items():

    values = (
        df[column]
        .dropna()
    )

    # Same empirical percentile framework
    # used in the project

    p20 = values.quantile(0.20)
    p80 = values.quantile(0.80)

    low = int(
        (values <= p20).sum()
    )

    high = int(
        (values >= p80).sum()
    )

    intermediate = (
        len(values)
        - low
        - high
    )

    group_rows.extend([
        {
            "Disease": disease,
            "Group": "Low",
            "Count": low
        },
        {
            "Disease": disease,
            "Group": "Intermediate",
            "Count": intermediate
        },
        {
            "Disease": disease,
            "Group": "High",
            "Count": high
        }
    ])


groups = pd.DataFrame(
    group_rows
)


pivot = groups.pivot(
    index="Disease",
    columns="Group",
    values="Count"
)


# Keep exact disease order
pivot = pivot.loc[
    list(disease_columns.keys())
]


pivot = pivot[
    ["Low", "Intermediate", "High"]
]


# ============================================================
# PLOT
# ============================================================

fig, ax = plt.subplots(
    figsize=(13, 7)
)

x = np.arange(
    len(pivot.index)
)

width = 0.24


ax.bar(
    x - width,
    pivot["Low"],
    width,
    label="Low",
    color="0.80",
    edgecolor="black",
    hatch="///"
)


ax.bar(
    x,
    pivot["Intermediate"],
    width,
    label="Intermediate",
    color="0.60",
    edgecolor="black",
    hatch="..."
)


ax.bar(
    x + width,
    pivot["High"],
    width,
    label="High",
    color="0.40",
    edgecolor="black",
    hatch="xxx"
)


ax.set_xticks(
    x
)

ax.set_xticklabels(
    pivot.index,
    rotation=25,
    ha="right"
)


ax.set_xlabel(
    "Disease",
    fontsize=11
)

ax.set_ylabel(
    "Number of Individuals",
    fontsize=11
)


ax.set_title(
    "Relative PRS Group Distribution Across Diseases",
    fontsize=14,
    pad=12
)


ax.legend(
    title="Relative PRS Group",
    frameon=True
)


ax.yaxis.grid(
    True,
    linestyle=":",
    linewidth=0.7,
    alpha=0.45
)

ax.set_axisbelow(True)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)


plt.tight_layout()


figure3 = os.path.join(
    OUTPUT_DIR,
    "Supplementary_Figure3_Relative_PRS_Group_Distribution.png"
)

plt.savefig(
    figure3,
    dpi=600,
    bbox_inches="tight"
)

plt.close()

print("\nCreated:")
print(figure3)


# ============================================================
# FINAL STATUS
# ============================================================

print("\n" + "=" * 70)
print("SUPPLEMENTARY FIGURE GENERATION COMPLETE")
print("=" * 70)

print("\nFigures created:")

print(figure1)
print(figure2)
print(figure3)

print("\nSTATUS: 3 SUPPLEMENTARY FIGURES CREATED")
print("=" * 70)
