import os
import pandas as pd

# ============================================================
# PARKINSON'S DISEASE PGS PREPARATION
# ============================================================

PROJECT = r"C:\PRS_Study"

PGS_FILE = os.path.join(
    PROJECT,
    "data",
    "PGS",
    "PGS000903.txt.gz"
)

OUTPUT_DIR = os.path.join(
    PROJECT,
    "results",
    "Parkinsons",
    "PGS"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

print("=" * 80)
print("PARKINSON'S DISEASE PGS PREPARATION")
print("=" * 80)

# ------------------------------------------------------------
# Check input file
# ------------------------------------------------------------

if not os.path.exists(PGS_FILE):

    raise FileNotFoundError(
        f"PGS file not found:\n{PGS_FILE}"
    )

print(f"\nPGS file:")
print(PGS_FILE)

# ------------------------------------------------------------
# Load PGS file
# IMPORTANT:
# PGS Catalog files contain metadata/comment lines beginning
# with '#', so comment='#' is required.
# ------------------------------------------------------------

print("\nLoading PGS file...")

df = pd.read_csv(
    PGS_FILE,
    sep="\t",
    comment="#",
    compression="gzip",
    low_memory=False
)

print(
    f"Loaded {len(df):,} variants."
)

print(
    f"Columns detected:\n{df.columns.tolist()}"
)

# ------------------------------------------------------------
# Clean column names
# ------------------------------------------------------------

df.columns = [
    str(col).strip()
    for col in df.columns
]

# ------------------------------------------------------------
# Required columns
# ------------------------------------------------------------

required_columns = [
    "rsID",
    "chr_name",
    "chr_position",
    "effect_allele",
    "effect_weight"
]

missing_columns = [
    col
    for col in required_columns
    if col not in df.columns
]

if missing_columns:

    raise ValueError(
        f"Missing required columns: "
        f"{missing_columns}"
    )

# ------------------------------------------------------------
# Convert numeric columns
# ------------------------------------------------------------

df["chr_name"] = pd.to_numeric(
    df["chr_name"],
    errors="coerce"
)

df["chr_position"] = pd.to_numeric(
    df["chr_position"],
    errors="coerce"
)

df["effect_weight"] = pd.to_numeric(
    df["effect_weight"],
    errors="coerce"
)

# ------------------------------------------------------------
# Clean effect allele
# ------------------------------------------------------------

df["effect_allele"] = (
    df["effect_allele"]
    .astype(str)
    .str.upper()
    .str.strip()
)

# ------------------------------------------------------------
# Remove missing essential information
# ------------------------------------------------------------

original_count = len(df)

df = df.dropna(
    subset=[
        "chr_name",
        "chr_position",
        "effect_allele",
        "effect_weight"
    ]
).copy()

print(
    f"\nAfter removing missing essential fields: "
    f"{len(df):,}"
)

# ------------------------------------------------------------
# Keep autosomes 1–22
# ------------------------------------------------------------

df = df[
    df["chr_name"].isin(
        range(1, 23)
    )
].copy()

print(
    f"After restricting to chromosomes 1-22: "
    f"{len(df):,}"
)

# ------------------------------------------------------------
# Validate effect alleles
# ------------------------------------------------------------

valid_alleles = {
    "A",
    "C",
    "G",
    "T"
}

invalid_alleles = (
    ~df["effect_allele"].isin(
        valid_alleles
    )
)

invalid_count = int(
    invalid_alleles.sum()
)

print(
    f"Invalid effect alleles: "
    f"{invalid_count}"
)

if invalid_count > 0:

    print(
        "\nInvalid allele records:"
    )

    print(
        df.loc[
            invalid_alleles,
            [
                "rsID",
                "chr_name",
                "chr_position",
                "effect_allele",
                "effect_weight"
            ]
        ].to_string(
            index=False
        )
    )

    raise ValueError(
        "Invalid effect alleles detected."
    )

# ------------------------------------------------------------
# Check duplicate chromosome-position records
# ------------------------------------------------------------

duplicate_positions = df.duplicated(
    subset=[
        "chr_name",
        "chr_position"
    ],
    keep=False
)

duplicate_position_count = int(
    duplicate_positions.sum()
)

print(
    f"Duplicate chromosome-position records: "
    f"{duplicate_position_count}"
)

if duplicate_position_count > 0:

    print(
        "\nDuplicate chromosome-position records:"
    )

    print(
        df.loc[
            duplicate_positions
        ].sort_values(
            [
                "chr_name",
                "chr_position"
            ]
        ).to_string(
            index=False
        )
    )

    raise ValueError(
        "Duplicate chromosome-position records detected."
    )

# ------------------------------------------------------------
# Check duplicate rsIDs
# ------------------------------------------------------------

duplicate_rsids = df["rsID"].duplicated(
    keep=False
)

duplicate_rsid_count = int(
    duplicate_rsids.sum()
)

print(
    f"Duplicate rsIDs: "
    f"{duplicate_rsid_count}"
)

if duplicate_rsid_count > 0:

    print(
        "\nDuplicate rsID records:"
    )

    print(
        df.loc[
            duplicate_rsids
        ].to_string(
            index=False
        )
    )

    raise ValueError(
        "Duplicate rsIDs detected."
    )

# ------------------------------------------------------------
# Sort by chromosome and position
# ------------------------------------------------------------

df = df.sort_values(
    [
        "chr_name",
        "chr_position"
    ]
).reset_index(
    drop=True
)

# ------------------------------------------------------------
# Prepare chromosome-specific files
# ------------------------------------------------------------

summary = []

print(
    "\n" + "-" * 80
)

print(
    "PREPARING CHROMOSOME-SPECIFIC FILES"
)

print(
    "-" * 80
)

for chrom in range(1, 23):

    chr_df = df[
        df["chr_name"] == chrom
    ].copy()

    # --------------------------------------------------------
    # PGS scoring file
    # --------------------------------------------------------

    scoring_file = os.path.join(
        OUTPUT_DIR,
        f"Parkinsons_chr{chrom}_PGS.tsv"
    )

    chr_df[
        [
            "rsID",
            "chr_position",
            "effect_allele",
            "effect_weight"
        ]
    ].to_csv(
        scoring_file,
        sep="\t",
        index=False
    )

    # --------------------------------------------------------
    # Position BED file
    #
    # PLINK --extract bed1 expects:
    #
    # CHR    START    END
    #
    # With bed1, coordinates are 1-based.
    # For a single SNP:
    # START = POS
    # END   = POS
    # --------------------------------------------------------

    bed_file = os.path.join(
        OUTPUT_DIR,
        f"Parkinsons_chr{chrom}_positions.bed"
    )

    bed_df = chr_df[
        [
            "chr_name",
            "chr_position"
        ]
    ].copy()

    bed_df["START"] = (
        bed_df["chr_position"]
    )

    bed_df["END"] = (
        bed_df["chr_position"]
    )

    bed_df[
        [
            "chr_name",
            "START",
            "END"
        ]
    ].to_csv(
        bed_file,
        sep="\t",
        header=False,
        index=False
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    variant_count = len(
        chr_df
    )

    summary.append(
        {
            "Chromosome": chrom,
            "PGS_variants": variant_count,
            "Scoring_file": scoring_file,
            "Position_file": bed_file
        }
    )

    print(
        f"Chr{chrom}: "
        f"{variant_count:,} variants"
    )

# ------------------------------------------------------------
# Create chromosome summary
# ------------------------------------------------------------

summary_df = pd.DataFrame(
    summary
)

summary_file = os.path.join(
    OUTPUT_DIR,
    "Parkinsons_PGS_chromosome_summary.tsv"
)

summary_df.to_csv(
    summary_file,
    sep="\t",
    index=False
)

# ------------------------------------------------------------
# Final checks
# ------------------------------------------------------------

total_prepared = int(
    summary_df["PGS_variants"].sum()
)

print(
    "\n" + "-" * 80
)

print(
    "FINAL CHECK"
)

print(
    "-" * 80
)

print(
    f"Original variants:                 "
    f"{original_count:,}"
)

print(
    f"Prepared variants:                 "
    f"{len(df):,}"
)

print(
    f"Chromosome total:                  "
    f"{total_prepared:,}"
)

print(
    f"Difference:                        "
    f"{len(df) - total_prepared:,}"
)

if total_prepared != len(df):

    raise ValueError(
        "Chromosome totals do not match "
        "prepared dataset."
    )

# ------------------------------------------------------------
# Final summary
# ------------------------------------------------------------

print(
    f"\nChromosome summary saved to:"
)

print(
    summary_file
)

print(
    "\nOutput directory:"
)

print(
    OUTPUT_DIR
)

print(
    "\n" + "=" * 80
)

print(
    "PARKINSON'S PGS PREPARATION COMPLETE"
)

print(
    "=" * 80
)
