import os
import pandas as pd

# ============================================================
# CORONARY ARTERY DISEASE — PGS PREPARATION
# PGS000018
# ============================================================

PROJECT = r"C:\PRS_Study"

PGS_FILE = os.path.join(
    PROJECT,
    "data",
    "PGS",
    "PGS000018.txt.gz"
)

OUTPUT_DIR = os.path.join(
    PROJECT,
    "results",
    "CAD",
    "PGS"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

SUMMARY_FILE = os.path.join(
    OUTPUT_DIR,
    "CAD_PGS_chromosome_summary.tsv"
)

print("=" * 80)
print("CORONARY ARTERY DISEASE — PGS000018 PREPARATION")
print("=" * 80)

# ============================================================
# LOAD PGS
# ============================================================

if not os.path.exists(PGS_FILE):
    raise FileNotFoundError(
        f"PGS file not found:\n{PGS_FILE}"
    )

print("\nLoading PGS file...")

pgs = pd.read_csv(
    PGS_FILE,
    sep="\t",
    comment="#",
    compression="gzip"
)

print(
    f"Loaded variants: {len(pgs):,}"
)

# ============================================================
# REQUIRED COLUMNS
# ============================================================

required_columns = [
    "rsID",
    "chr_name",
    "chr_position",
    "effect_allele",
    "other_allele",
    "effect_weight"
]

for column in required_columns:

    if column not in pgs.columns:

        raise ValueError(
            f"Required column missing: {column}"
        )

# ============================================================
# CLEAN TYPES
# ============================================================

pgs["chr_name"] = pd.to_numeric(
    pgs["chr_name"],
    errors="coerce"
)

pgs["chr_position"] = pd.to_numeric(
    pgs["chr_position"],
    errors="coerce"
)

pgs["effect_weight"] = pd.to_numeric(
    pgs["effect_weight"],
    errors="coerce"
)

pgs["effect_allele"] = (
    pgs["effect_allele"]
    .astype(str)
    .str.upper()
    .str.strip()
)

pgs["other_allele"] = (
    pgs["other_allele"]
    .astype(str)
    .str.upper()
    .str.strip()
)

# ============================================================
# INITIAL QC
# ============================================================

print("\n" + "-" * 80)
print("INITIAL QC")
print("-" * 80)

print(
    f"Original variants: {len(pgs):,}"
)

# Missing essential fields

essential = [
    "rsID",
    "chr_name",
    "chr_position",
    "effect_allele",
    "effect_weight"
]

missing_essential = (
    pgs[essential]
    .isna()
    .any(axis=1)
)

print(
    f"Variants with missing essential fields: "
    f"{missing_essential.sum():,}"
)

pgs = pgs[
    ~missing_essential
].copy()

# ============================================================
# CHROMOSOME FILTER
# ============================================================

pgs = pgs[
    pgs["chr_name"].isin(
        range(1, 23)
    )
].copy()

print(
    f"Variants on chromosomes 1–22: "
    f"{len(pgs):,}"
)

# ============================================================
# ALLELE VALIDATION
# ============================================================

valid_alleles = {
    "A",
    "C",
    "G",
    "T"
}

invalid_effect = ~pgs[
    "effect_allele"
].isin(
    valid_alleles
)

invalid_other = ~pgs[
    "other_allele"
].isin(
    valid_alleles
)

invalid_alleles = (
    invalid_effect
    |
    invalid_other
)

print(
    f"Invalid allele records: "
    f"{invalid_alleles.sum():,}"
)

if invalid_alleles.any():

    print(
        "\nRemoving invalid allele records..."
    )

    pgs = pgs[
        ~invalid_alleles
    ].copy()

# ============================================================
# DUPLICATE CHECKS
# ============================================================

duplicate_position = pgs.duplicated(
    subset=[
        "chr_name",
        "chr_position"
    ],
    keep=False
)

duplicate_rsid = pgs.duplicated(
    subset=[
        "rsID"
    ],
    keep=False
)

print(
    f"Duplicate chromosome-position records: "
    f"{duplicate_position.sum():,}"
)

print(
    f"Duplicate rsID records: "
    f"{duplicate_rsid.sum():,}"
)

# ============================================================
# ZERO WEIGHTS
# ============================================================

zero_weights = (
    pgs["effect_weight"] == 0
)

print(
    f"Zero-weight variants retained: "
    f"{zero_weights.sum():,}"
)

# ============================================================
# CHROMOSOME-WISE FILES
# ============================================================

print("\n" + "-" * 80)
print("CREATING CHROMOSOME-SPECIFIC PGS FILES")
print("-" * 80)

summary_rows = []

for chrom in range(1, 23):

    chrom_data = pgs[
        pgs["chr_name"] == chrom
    ].copy()

    chrom_data = chrom_data.sort_values(
        "chr_position"
    )

    pgs_output = os.path.join(
        OUTPUT_DIR,
        f"CAD_chr{chrom}_PGS.tsv"
    )

    bed_output = os.path.join(
        OUTPUT_DIR,
        f"CAD_chr{chrom}_positions.bed"
    )

    # --------------------------------------------------------
    # Save PGS
    # --------------------------------------------------------

    chrom_data.to_csv(
        pgs_output,
        sep="\t",
        index=False
    )

    # --------------------------------------------------------
    # Create position BED
    #
    # PLINK --extract bed1 expects:
    # chromosome, position, position
    #
    # This represents a 1-based single-position interval.
    # --------------------------------------------------------

    bed = chrom_data[
        [
            "chr_name",
            "chr_position"
        ]
    ].copy()

    bed["start"] = bed[
        "chr_position"
    ]

    bed["end"] = bed[
        "chr_position"
    ]

    bed = bed[
        [
            "chr_name",
            "start",
            "end"
        ]
    ]

    bed.to_csv(
        bed_output,
        sep="\t",
        header=False,
        index=False
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary_rows.append(
        {
            "Chromosome": chrom,
            "PGS_variants": len(chrom_data),
            "Duplicate_positions": int(
                chrom_data.duplicated(
                    subset=[
                        "chr_name",
                        "chr_position"
                    ]
                ).sum()
            ),
            "Zero_weights": int(
                (
                    chrom_data[
                        "effect_weight"
                    ] == 0
                ).sum()
            )
        }
    )

    print(
        f"Chr{chrom}: "
        f"{len(chrom_data):,} variants"
    )

# ============================================================
# SUMMARY
# ============================================================

summary = pd.DataFrame(
    summary_rows
)

summary.to_csv(
    SUMMARY_FILE,
    sep="\t",
    index=False
)

# ============================================================
# FINAL QC
# ============================================================

chromosome_total = summary[
    "PGS_variants"
].sum()

print("\n" + "-" * 80)
print("FINAL PREPARATION QC")
print("-" * 80)

print(
    f"Original variants:     {1745179:,}"
)

print(
    f"Prepared variants:     {len(pgs):,}"
)

print(
    f"Chromosome total:      {chromosome_total:,}"
)

print(
    f"Difference:            "
    f"{len(pgs) - chromosome_total:,}"
)

print(
    f"Zero-weight variants:  "
    f"{(pgs['effect_weight'] == 0).sum():,}"
)

# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 80)
print("CAD PGS PREPARATION COMPLETE")
print("=" * 80)

print(
    f"\nChromosome-specific PGS files saved in:\n"
    f"{OUTPUT_DIR}"
)

print(
    f"\nChromosome summary:\n"
    f"{SUMMARY_FILE}"
)

print("\nSTATUS: SUCCESS")
