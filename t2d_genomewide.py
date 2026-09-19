# ============================================================
# T2D GENOME-WIDE PRS + POPULATION ANALYSIS
# Disease: Type 2 Diabetes
# PGS ID: PGS000036
#
# Uses chromosome-wise PLINK score SUM values.
# No additional p-value thresholding, LD clumping, pruning,
# Bayesian shrinkage, or model fitting is performed.
# ============================================================

import os
import subprocess
import pandas as pd
import numpy as np
from scipy.stats import kruskal, rankdata
from statsmodels.stats.multitest import multipletests


# ============================================================
# 1. PATHS
# ============================================================

ROOT = r"C:\PRS_Study"

PLINK = os.path.join(
    ROOT,
    "plink",
    "plink2.exe"
)

RESULTS = os.path.join(
    ROOT,
    "results",
    "T2D"
)

POP_PANEL = os.path.join(
    ROOT,
    "data",
    "1000G",
    "1000G_phase3_population_panel.txt"
)

os.makedirs(RESULTS, exist_ok=True)


# ============================================================
# 2. HELPER FUNCTION
# ============================================================

def run_command(cmd):
    """
    Run a command and stop if PLINK returns an error.
    """
    print("\nRunning:")
    print(" ".join(cmd))

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True
    )

    if result.stdout:
        print(result.stdout)

    if result.stderr:
        print(result.stderr)

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}"
        )


# ============================================================
# 3. READ PLINK SCORE OUTPUT
# ============================================================

def read_plink_score(score_file, chrom):
    """
    Read PLINK .sscore output and extract the actual
    EFFECT_WEIGHT_SUM column.

    PLINK produces:
        ALLELE_CT
        NAMED_ALLELE_DOSAGE_SUM
        EFFECT_WEIGHT_AVG
        EFFECT_WEIGHT_SUM

    NAMED_ALLELE_DOSAGE_SUM is NOT the PRS.

    EFFECT_WEIGHT_SUM is the unaveraged PRS contribution
    from that chromosome.
    """

    if not os.path.exists(score_file):
        raise FileNotFoundError(
            f"Score file not found for Chr{chrom}: {score_file}"
        )

    df = pd.read_csv(
        score_file,
        sep=r"\s+",
        engine="python"
    )

    print(f"\nChr{chrom} score file columns:")
    print(list(df.columns))

    # --------------------------------------------------------
    # Find IID column
    # --------------------------------------------------------

    iid_candidates = [
        "IID",
        "#IID"
    ]

    iid_col = None

    for col in iid_candidates:
        if col in df.columns:
            iid_col = col
            break

    if iid_col is None:
        raise ValueError(
            f"Chr{chrom}: could not find IID column. "
            f"Available columns: {list(df.columns)}"
        )

    # --------------------------------------------------------
    # IMPORTANT FIX:
    # Select EFFECT_WEIGHT_SUM specifically.
    #
    # Do NOT select NAMED_ALLELE_DOSAGE_SUM.
    # --------------------------------------------------------

    exact_candidates = [
        "EFFECT_WEIGHT_SUM"
    ]

    score_col = None

    for col in exact_candidates:
        if col in df.columns:
            score_col = col
            break

    # --------------------------------------------------------
    # Fallback in case PLINK prefixes the score name
    # --------------------------------------------------------

    if score_col is None:

        possible_score_columns = [
            c for c in df.columns
            if str(c).endswith("_SUM")
            and "NAMED_ALLELE_DOSAGE" not in str(c)
        ]

        if len(possible_score_columns) == 1:
            score_col = possible_score_columns[0]

        else:
            raise ValueError(
                f"Chr{chrom}: could not uniquely identify "
                f"the PRS score sum column.\n"
                f"Available columns: {list(df.columns)}\n"
                f"Candidate columns: {possible_score_columns}"
            )

    print(
        f"Chr{chrom}: using PRS score column -> {score_col}"
    )

    # --------------------------------------------------------
    # Create clean output
    # --------------------------------------------------------

    out = pd.DataFrame({
        "IID": df[iid_col].astype(str),
        "PRS": pd.to_numeric(
            df[score_col],
            errors="coerce"
        )
    })

    # Check missing
    missing = out["PRS"].isna().sum()

    if missing > 0:
        raise ValueError(
            f"Chr{chrom}: {missing} missing PRS values found."
        )

    # Check duplicate individuals
    duplicates = out["IID"].duplicated().sum()

    if duplicates > 0:
        raise ValueError(
            f"Chr{chrom}: {duplicates} duplicate IID records found."
        )

    return out


# ============================================================
# 4. SCORE EACH CHROMOSOME USING SCORE SUMS
# ============================================================

chromosome_scores = []

print("\n" + "=" * 70)
print("T2D CHROMOSOME-WISE PRS RESCORING")
print("=" * 70)

for chrom in range(1, 23):

    chrom_dir = os.path.join(
        RESULTS,
        f"T2D_chr{chrom}"
    )

    # --------------------------------------------------------
    # Standard chromosomes
    # --------------------------------------------------------

    if chrom not in [12, 17]:

        score_prefix = os.path.join(
            chrom_dir,
            f"T2D_chr{chrom}_scoring"
        )

        score_file = os.path.join(
            chrom_dir,
            f"T2D_chr{chrom}_scorefile.tsv"
        )

    # --------------------------------------------------------
    # Chr12 and Chr17 recovered using fast extraction
    # --------------------------------------------------------

    else:

        score_prefix = os.path.join(
            chrom_dir,
            f"T2D_chr{chrom}_fast_scoring"
        )

        score_file = os.path.join(
            chrom_dir,
            f"T2D_chr{chrom}_fast_score.tsv"
        )

    pgen_file = score_prefix + ".pgen"
    pvar_file = score_prefix + ".pvar"
    psam_file = score_prefix + ".psam"

    # --------------------------------------------------------
    # Check extracted PLINK files
    # --------------------------------------------------------

    if not all(
        os.path.exists(x)
        for x in [
            pgen_file,
            pvar_file,
            psam_file
        ]
    ):
        raise FileNotFoundError(
            f"Chr{chrom}: required PLINK files are missing.\n"
            f"PGEN: {pgen_file}\n"
            f"PVAR: {pvar_file}\n"
            f"PSAM: {psam_file}"
        )

    # --------------------------------------------------------
    # Score SUM output
    # --------------------------------------------------------

    sum_prefix = os.path.join(
        chrom_dir,
        f"T2D_chr{chrom}_sum"
    )

    sum_sscore = sum_prefix + ".sscore"

    # --------------------------------------------------------
    # Run only if SUM file does not already exist
    # --------------------------------------------------------

    if not os.path.exists(sum_sscore):

        cmd = [
            PLINK,

            "--pfile",
            score_prefix,

            "--score",
            score_file,
            "1",
            "2",
            "3",
            "header-read",

            # Request the unaveraged score sum
            "cols=+scoresums",

            "--memory",
            "2500",

            "--threads",
            "4",

            "--out",
            sum_prefix
        ]

        run_command(cmd)

    else:

        print(
            f"\nChr{chrom}: SUM score already exists. "
            f"Skipping PLINK scoring."
        )

    # --------------------------------------------------------
    # Read the score
    # --------------------------------------------------------

    chrom_df = read_plink_score(
        sum_sscore,
        chrom
    )

    print(
        f"Chr{chrom}: "
        f"N={len(chrom_df)}, "
        f"mean={chrom_df['PRS'].mean():.8f}, "
        f"SD={chrom_df['PRS'].std():.8f}"
    )

    # Rename chromosome-specific PRS column
    chrom_df = chrom_df.rename(
        columns={
            "PRS": f"PRS_chr{chrom}"
        }
    )

    chromosome_scores.append(
        chrom_df
    )


# ============================================================
# 5. MERGE ALL CHROMOSOMES
# ============================================================

print("\n" + "=" * 70)
print("MERGING CHROMOSOME SCORES")
print("=" * 70)

genomewide = chromosome_scores[0].copy()

for df in chromosome_scores[1:]:

    genomewide = genomewide.merge(
        df,
        on="IID",
        how="inner"
    )

# ------------------------------------------------------------
# Verify sample count
# ------------------------------------------------------------

print(
    f"Individuals after merging: {len(genomewide)}"
)

if len(genomewide) != 2504:

    raise ValueError(
        f"Expected 2504 individuals, "
        f"found {len(genomewide)}."
    )


# ============================================================
# 6. CALCULATE GENOME-WIDE T2D PRS
# ============================================================

prs_columns = [
    f"PRS_chr{chrom}"
    for chrom in range(1, 23)
]

# Verify all chromosome columns exist
missing_columns = [
    col
    for col in prs_columns
    if col not in genomewide.columns
]

if missing_columns:

    raise ValueError(
        f"Missing chromosome score columns: "
        f"{missing_columns}"
    )


genomewide["T2D_PRS"] = genomewide[
    prs_columns
].sum(axis=1)


# ============================================================
# 7. GENOME-WIDE SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("GENOME-WIDE T2D PRS SUMMARY")
print("=" * 70)

N = len(genomewide)

mean_prs = genomewide["T2D_PRS"].mean()
sd_prs = genomewide["T2D_PRS"].std()
median_prs = genomewide["T2D_PRS"].median()
min_prs = genomewide["T2D_PRS"].min()
max_prs = genomewide["T2D_PRS"].max()

q1 = genomewide["T2D_PRS"].quantile(0.25)
q3 = genomewide["T2D_PRS"].quantile(0.75)

missing_prs = genomewide["T2D_PRS"].isna().sum()

print(f"N       : {N}")
print(f"Mean    : {mean_prs:.6f}")
print(f"SD      : {sd_prs:.6f}")
print(f"Median  : {median_prs:.6f}")
print(f"Min     : {min_prs:.6f}")
print(f"Q1      : {q1:.6f}")
print(f"Q3      : {q3:.6f}")
print(f"Max     : {max_prs:.6f}")
print(f"Missing : {missing_prs}")


# ============================================================
# 8. SAVE GENOME-WIDE PRS
# ============================================================

genomewide_file = os.path.join(
    RESULTS,
    "T2D_genomewide_PRS.tsv"
)

genomewide[
    ["IID", "T2D_PRS"]
].to_csv(
    genomewide_file,
    sep="\t",
    index=False
)

print(
    f"\nGenome-wide PRS saved to:\n"
    f"{genomewide_file}"
)


# ============================================================
# 9. RELATIVE PRS GROUPS
# ============================================================

print("\n" + "=" * 70)
print("RELATIVE PRS GROUPING")
print("=" * 70)

p20 = genomewide["T2D_PRS"].quantile(0.20)
p80 = genomewide["T2D_PRS"].quantile(0.80)

print(f"20th percentile: {p20:.8f}")
print(f"80th percentile: {p80:.8f}")


def assign_group(x):

    if x <= p20:
        return "Low"

    elif x >= p80:
        return "High"

    else:
        return "Intermediate"


genomewide["Relative_PRS_Group"] = (
    genomewide["T2D_PRS"]
    .apply(assign_group)
)


# ------------------------------------------------------------
# Group summary
# ------------------------------------------------------------

group_summary = (
    genomewide
    .groupby("Relative_PRS_Group")["T2D_PRS"]
    .agg(
        N="count",
        Mean="mean",
        SD="std",
        Min="min",
        Max="max"
    )
    .reset_index()
)

group_summary["Percentage"] = (
    group_summary["N"] / N * 100
)

# Desired ordering
group_order = [
    "Low",
    "Intermediate",
    "High"
]

group_summary["order"] = (
    group_summary["Relative_PRS_Group"]
    .map(
        {
            "Low": 1,
            "Intermediate": 2,
            "High": 3
        }
    )
)

group_summary = (
    group_summary
    .sort_values("order")
    .drop(columns="order")
)


print("\nRelative PRS groups:")

print(
    group_summary.to_string(
        index=False
    )
)


# ============================================================
# 10. SAVE RELATIVE GROUP OUTPUTS
# ============================================================

risk_group_file = os.path.join(
    RESULTS,
    "T2D_genomewide_PRS_with_risk_groups.tsv"
)

genomewide[
    [
        "IID",
        "T2D_PRS",
        "Relative_PRS_Group"
    ]
].to_csv(
    risk_group_file,
    sep="\t",
    index=False
)


risk_summary_file = os.path.join(
    RESULTS,
    "T2D_risk_group_summary.tsv"
)

group_summary.to_csv(
    risk_summary_file,
    sep="\t",
    index=False
)

print(
    f"\nRisk-group individual file:\n"
    f"{risk_group_file}"
)

print(
    f"\nRisk-group summary:\n"
    f"{risk_summary_file}"
)


# ============================================================
# 11. LOAD 1000 GENOMES POPULATION PANEL
# ============================================================

print("\n" + "=" * 70)
print("POPULATION ANALYSIS")
print("=" * 70)

if not os.path.exists(POP_PANEL):

    raise FileNotFoundError(
        f"Population panel not found:\n{POP_PANEL}"
    )


panel = pd.read_csv(
    POP_PANEL,
    sep=r"\s+",
    engine="python"
)

print(
    f"Population panel individuals: "
    f"{len(panel)}"
)

print(
    "Population panel columns:"
)

print(
    list(panel.columns)
)


# ============================================================
# 12. STANDARDIZE POPULATION PANEL ID
# ============================================================

if "sample" in panel.columns:

    panel = panel.rename(
        columns={
            "sample": "IID"
        }
    )

elif "IID" not in panel.columns:

    raise ValueError(
        "Population panel does not contain "
        "'sample' or 'IID' column."
    )


panel["IID"] = (
    panel["IID"]
    .astype(str)
)


genomewide["IID"] = (
    genomewide["IID"]
    .astype(str)
)


# ============================================================
# 13. CHECK SAMPLE ID MATCHING
# ============================================================

prs_ids = set(
    genomewide["IID"]
)

panel_ids = set(
    panel["IID"]
)

missing_population_labels = (
    prs_ids - panel_ids
)

extra_panel_ids = (
    panel_ids - prs_ids
)

print(
    f"PRS individuals       : {len(prs_ids)}"
)

print(
    f"Panel individuals     : {len(panel_ids)}"
)

print(
    f"Missing panel labels  : "
    f"{len(missing_population_labels)}"
)

print(
    f"Extra panel IDs       : "
    f"{len(extra_panel_ids)}"
)

if missing_population_labels:

    raise ValueError(
        "Some PRS individuals do not have "
        "population labels."
    )


# ============================================================
# 14. MERGE POPULATION LABELS
# ============================================================

population_df = genomewide[
    [
        "IID",
        "T2D_PRS",
        "Relative_PRS_Group"
    ]
].merge(
    panel[
        [
            "IID",
            "pop",
            "super_pop",
            "gender"
        ]
    ],
    on="IID",
    how="left"
)


if population_df["super_pop"].isna().any():

    raise ValueError(
        "Some individuals have missing "
        "superpopulation labels after merging."
    )


# ============================================================
# 15. SAVE POPULATION-LABELED PRS
# ============================================================

population_labeled_file = os.path.join(
    RESULTS,
    "T2D_PRS_with_population_labels.tsv"
)

population_df.to_csv(
    population_labeled_file,
    sep="\t",
    index=False
)

print(
    f"\nPopulation-labeled PRS saved to:\n"
    f"{population_labeled_file}"
)


# ============================================================
# 16. POPULATION DESCRIPTIVE STATISTICS
# ============================================================

superpop_order = [
    "AFR",
    "AMR",
    "EAS",
    "EUR",
    "SAS"
]

population_stats = []

for pop_name in superpop_order:

    values = (
        population_df.loc[
            population_df["super_pop"] == pop_name,
            "T2D_PRS"
        ]
        .dropna()
        .values
    )

    if len(values) == 0:
        continue

    population_stats.append(
        {
            "Superpopulation": pop_name,
            "N": len(values),
            "Mean": np.mean(values),
            "SD": np.std(
                values,
                ddof=1
            ),
            "Median": np.median(values),
            "Min": np.min(values),
            "Q1": np.percentile(
                values,
                25
            ),
            "Q3": np.percentile(
                values,
                75
            ),
            "Max": np.max(values),
            "SE": (
                np.std(
                    values,
                    ddof=1
                )
                / np.sqrt(len(values))
            )
        }
    )


population_stats_df = pd.DataFrame(
    population_stats
)


print("\nPopulation PRS statistics:")

print(
    population_stats_df.to_string(
        index=False
    )
)


# ============================================================
# 17. SAVE POPULATION STATISTICS
# ============================================================

population_stats_file = os.path.join(
    RESULTS,
    "T2D_population_PRS_by_superpopulation.tsv"
)

population_stats_df.to_csv(
    population_stats_file,
    sep="\t",
    index=False
)


# ============================================================
# 18. KRUSKAL-WALLIS TEST
# ============================================================

groups = []

for pop_name in superpop_order:

    values = (
        population_df.loc[
            population_df["super_pop"] == pop_name,
            "T2D_PRS"
        ]
        .dropna()
        .values
    )

    if len(values) > 0:
        groups.append(values)


H_stat, kw_p = kruskal(
    *groups
)


kw_results = pd.DataFrame(
    [
        {
            "Test": "Kruskal-Wallis",
            "H_statistic": H_stat,
            "P_value": kw_p
        }
    ]
)


print("\nKruskal-Wallis test:")

print(
    kw_results.to_string(
        index=False
    )
)


# ============================================================
# 19. SAVE KRUSKAL-WALLIS RESULTS
# ============================================================

kw_file = os.path.join(
    RESULTS,
    "T2D_population_Kruskal_Wallis.tsv"
)

kw_results.to_csv(
    kw_file,
    sep="\t",
    index=False
)


# ============================================================
# 20. DUNN POST-HOC TEST
# ============================================================
#
# Manual Dunn test implementation using pooled ranks.
#
# Holm correction is applied to all pairwise comparisons.
# ============================================================

print("\n" + "=" * 70)
print("DUNN POST-HOC TEST WITH HOLM CORRECTION")
print("=" * 70)


# ------------------------------------------------------------
# Prepare data
# ------------------------------------------------------------

dunn_df = population_df[
    [
        "T2D_PRS",
        "super_pop"
    ]
].dropna().copy()


values = dunn_df[
    "T2D_PRS"
].values

groups_labels = dunn_df[
    "super_pop"
].values


# ------------------------------------------------------------
# Pooled ranks
# ------------------------------------------------------------

ranks = rankdata(
    values,
    method="average"
)

dunn_df["Rank"] = ranks


# ------------------------------------------------------------
# Tie correction
# ------------------------------------------------------------

N_total = len(dunn_df)

_, tie_counts = np.unique(
    ranks,
    return_counts=True
)

tie_sum = np.sum(
    tie_counts ** 3
    - tie_counts
)

tie_correction = (
    1
    - tie_sum
    / (
        N_total ** 3
        - N_total
    )
)


# ------------------------------------------------------------
# Mean rank for each population
# ------------------------------------------------------------

rank_summary = (
    dunn_df
    .groupby("super_pop")["Rank"]
    .agg(
        mean_rank="mean",
        N="count"
    )
)


# ------------------------------------------------------------
# Pairwise comparisons
# ------------------------------------------------------------

from itertools import combinations
from scipy.stats import norm


pairwise_results = []

for pop1, pop2 in combinations(
    superpop_order,
    2
):

    if (
        pop1 not in rank_summary.index
        or pop2 not in rank_summary.index
    ):
        continue

    n1 = rank_summary.loc[
        pop1,
        "N"
    ]

    n2 = rank_summary.loc[
        pop2,
        "N"
    ]

    mean_rank1 = rank_summary.loc[
        pop1,
        "mean_rank"
    ]

    mean_rank2 = rank_summary.loc[
        pop2,
        "mean_rank"
    ]

    # Dunn variance
    variance = (
        (
            N_total
            * (
                N_total + 1
            )
            / 12
        )
        * tie_correction
        * (
            1 / n1
            + 1 / n2
        )
    )

    z = (
        mean_rank1
        - mean_rank2
    ) / np.sqrt(variance)

    p_value = (
        2
        * norm.sf(
            abs(z)
        )
    )

    pairwise_results.append(
        {
            "Group1": pop1,
            "Group2": pop2,
            "Z": z,
            "Raw_P": p_value
        }
    )


dunn_results = pd.DataFrame(
    pairwise_results
)


# ============================================================
# 21. HOLM MULTIPLE-TEST CORRECTION
# ============================================================

reject, p_holm, _, _ = multipletests(
    dunn_results["Raw_P"].values,
    method="holm"
)

dunn_results[
    "Holm_Adjusted_P"
] = p_holm

dunn_results[
    "Significant"
] = reject


# ============================================================
# 22. PRINT DUNN RESULTS
# ============================================================

print(
    dunn_results.to_string(
        index=False
    )
)


# ============================================================
# 23. SAVE DUNN RESULTS
# ============================================================

dunn_file = os.path.join(
    RESULTS,
    "T2D_population_Dunn_Holm.tsv"
)

dunn_results.to_csv(
    dunn_file,
    sep="\t",
    index=False
)


# ============================================================
# 24. SAVE SIGNIFICANT PAIRS
# ============================================================

significant_pairs = dunn_results[
    dunn_results["Significant"]
].copy()


significant_pairs_file = os.path.join(
    RESULTS,
    "T2D_population_significant_pairs.tsv"
)

significant_pairs.to_csv(
    significant_pairs_file,
    sep="\t",
    index=False
)


# ============================================================
# 25. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("T2D PRS ANALYSIS COMPLETE")
print("=" * 70)

print(
    f"\nGenome-wide N: {N}"
)

print(
    f"Genome-wide mean: "
    f"{mean_prs:.6f}"
)

print(
    f"Genome-wide SD: "
    f"{sd_prs:.6f}"
)

print(
    f"Genome-wide median: "
    f"{median_prs:.6f}"
)

print(
    f"20th percentile: "
    f"{p20:.6f}"
)

print(
    f"80th percentile: "
    f"{p80:.6f}"
)

print("\nRelative PRS groups:")

print(
    group_summary.to_string(
        index=False
    )
)

print(
    "\nPopulation analysis:"
)

print(
    population_stats_df.to_string(
        index=False
    )
)

print(
    f"\nKruskal-Wallis H = "
    f"{H_stat:.6f}"
)

print(
    f"Kruskal-Wallis p = "
    f"{kw_p:.6e}"
)

print(
    f"\nSignificant Dunn-Holm pairs: "
    f"{len(significant_pairs)} / "
    f"{len(dunn_results)}"
)

print("\nOutput files:")

print(
    genomewide_file
)

print(
    risk_group_file
)

print(
    risk_summary_file
)

print(
    population_labeled_file
)

print(
    population_stats_file
)

print(
    kw_file
)

print(
    dunn_file
)

print(
    significant_pairs_file
)

print("\nDONE.")
