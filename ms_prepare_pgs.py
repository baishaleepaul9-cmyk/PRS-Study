import pandas as pd
from pathlib import Path


# ============================================================
# MULTIPLE SCLEROSIS - PREPARE CHROMOSOME-SPECIFIC PGS FILES
# PGS002726
# ============================================================

BASE = Path(r"C:\PRS_Study")

PGS_FILE = (
    BASE
    / "data"
    / "PGS"
    / "PGS002726.txt.gz"
)

RESULTS_DIR = (
    BASE
    / "results"
    / "Multiple_Sclerosis"
)

PGS_DIR = (
    RESULTS_DIR
    / "PGS"
)

PGS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 1. START
# ============================================================

print("=" * 70)
print("MULTIPLE SCLEROSIS - PGS CHROMOSOME PREPARATION")
print("=" * 70)

print("\nInput PGS file:")
print(PGS_FILE)


# ============================================================
# 2. CHECK INPUT
# ============================================================

if not PGS_FILE.exists():

    raise FileNotFoundError(
        f"\nPGS file not found:\n{PGS_FILE}"
    )


# ============================================================
# 3. LOAD PGS FILE
# ============================================================

print("\nReading PGS file...")

df = pd.read_csv(
    PGS_FILE,
    sep="\t",
    comment="#",
    compression="gzip"
)

print(
    f"Loaded {len(df):,} variants."
)

print(
    f"Columns: {list(df.columns)}"
)


# ============================================================
# 4. CHECK REQUIRED COLUMNS
# ============================================================

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
        "Missing required columns: "
        + ", ".join(missing_columns)
    )


# ============================================================
# 5. CLEAN REQUIRED FIELDS
# ============================================================

df["chr_name"] = (
    pd.to_numeric(
        df["chr_name"],
        errors="coerce"
    )
)

df["chr_position"] = (
    pd.to_numeric(
        df["chr_position"],
        errors="coerce"
    )
)

df["effect_weight"] = (
    pd.to_numeric(
        df["effect_weight"],
        errors="coerce"
    )
)

df["effect_allele"] = (
    df["effect_allele"]
    .astype(str)
    .str.upper()
    .str.strip()
)


# ============================================================
# 6. REMOVE INVALID COORDINATES
# ============================================================

before = len(df)

df = df.dropna(
    subset=[
        "chr_name",
        "chr_position",
        "effect_weight",
        "effect_allele"
    ]
).copy()

after = len(df)

print(
    f"\nVariants before coordinate/weight cleaning: "
    f"{before:,}"
)

print(
    f"Variants after cleaning: "
    f"{after:,}"
)


# ============================================================
# 7. KEEP AUTOSOMAL CHROMOSOMES 1-22
# ============================================================

df = df[
    df["chr_name"].isin(
        range(1, 23)
    )
].copy()

print(
    f"Variants on chromosomes 1-22: "
    f"{len(df):,}"
)


# ============================================================
# 8. FORMAT CHROMOSOME AND POSITION
# ============================================================

df["chr_name"] = (
    df["chr_name"]
    .astype(int)
)

df["chr_position"] = (
    df["chr_position"]
    .astype(int)
)


# ============================================================
# 9. CHECK DUPLICATE CHROMOSOME-POSITION RECORDS
# ============================================================
#
# Important:
# The PGS file has 108 duplicate rsID values, but the rsID
# field is ".". Therefore, chromosome + position is the
# meaningful matching coordinate.
# ============================================================

duplicate_positions = (
    df.duplicated(
        subset=[
            "chr_name",
            "chr_position"
        ],
        keep=False
    )
)

duplicate_position_count = (
    duplicate_positions.sum()
)

unique_positions = (
    df[
        [
            "chr_name",
            "chr_position"
        ]
    ]
    .drop_duplicates()
    .shape[0]
)

print(
    "\nDuplicate chromosome-position records:"
)

print(
    duplicate_position_count
)

print(
    f"Unique chromosome-position coordinates: "
    f"{unique_positions:,}"
)


# ============================================================
# 10. CHROMOSOME-SPECIFIC FILES
# ============================================================

chromosome_summary = []


for chrom in range(1, 23):

    print("\n" + "-" * 70)

    chr_df = df[
        df["chr_name"] == chrom
    ].copy()

    print(
        f"Chromosome {chrom}: "
        f"{len(chr_df):,} variants"
    )


    # --------------------------------------------------------
    # Sort by genomic position
    # --------------------------------------------------------

    chr_df = chr_df.sort_values(
        [
            "chr_position"
        ]
    ).reset_index(
        drop=True
    )


    # --------------------------------------------------------
    # Save full chromosome PGS table
    # --------------------------------------------------------

    pgs_output = (
        PGS_DIR
        / f"MS_chr{chrom}_PGS.tsv"
    )

    chr_df[
        [
            "rsID",
            "chr_name",
            "chr_position",
            "effect_allele",
            "effect_weight",
            "variant_description"
        ]
    ].to_csv(
        pgs_output,
        sep="\t",
        index=False
    )


    # --------------------------------------------------------
    # Save chromosome-position file
    #
    # This will be used for coordinate-based extraction from
    # the 1000 Genomes VCF.
    #
    # Format:
    # CHR START END
    #
    # BED coordinates are 0-based start / 1-based end.
    # --------------------------------------------------------

    bed_output = (
        PGS_DIR
        / f"MS_chr{chrom}_positions.bed"
    )

    bed_df = pd.DataFrame({

        "CHR": chr_df["chr_name"],

        "START": (
            chr_df["chr_position"] - 1
        ),

        "END": chr_df["chr_position"]
    })

    bed_df.to_csv(
        bed_output,
        sep="\t",
        header=False,
        index=False
    )


    # --------------------------------------------------------
    # Save simple position list
    # --------------------------------------------------------

    position_output = (
        PGS_DIR
        / f"MS_chr{chrom}_positions.txt"
    )

    chr_df[
        "chr_position"
    ].to_csv(
        position_output,
        index=False,
        header=False
    )


    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    chromosome_summary.append({

        "Chromosome": chrom,

        "PGS_variants": len(chr_df),

        "Unique_positions": (
            chr_df[
                [
                    "chr_name",
                    "chr_position"
                ]
            ]
            .drop_duplicates()
            .shape[0]
        ),

        "Output_PGS": str(
            pgs_output
        ),

        "Output_BED": str(
            bed_output
        ),

        "Output_positions": str(
            position_output
        )
    })


# ============================================================
# 11. SAVE CHROMOSOME SUMMARY
# ============================================================

summary = pd.DataFrame(
    chromosome_summary
)

summary_output = (
    PGS_DIR
    / "MS_PGS_chromosome_summary.tsv"
)

summary.to_csv(
    summary_output,
    sep="\t",
    index=False
)


# ============================================================
# 12. PRINT FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("CHROMOSOME-SPECIFIC PGS SUMMARY")
print("=" * 70)

print(
    summary[
        [
            "Chromosome",
            "PGS_variants",
            "Unique_positions"
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# 13. TOTAL CHECK
# ============================================================

total_prepared = (
    summary[
        "PGS_variants"
    ].sum()
)

print("\n" + "-" * 70)

print(
    f"Original valid PGS variants : "
    f"{len(df):,}"
)

print(
    f"Prepared chromosome variants: "
    f"{total_prepared:,}"
)

print(
    f"Difference                  : "
    f"{len(df) - total_prepared:,}"
)


# ============================================================
# 14. OUTPUT LOCATION
# ============================================================

print("\nPGS files saved in:")

print(PGS_DIR)

print("\nChromosome summary saved to:")

print(summary_output)


# ============================================================
# 15. COMPLETION
# ============================================================

print("\n" + "=" * 70)
print("MULTIPLE SCLEROSIS PGS PREPARATION COMPLETE")
print("=" * 70)

print(
    "\nChromosomes prepared: 1-22"
)

print(
    "Do not start PLINK scoring yet."
)

print(
    "First verify the chromosome counts above."
)

print("=" * 70)
