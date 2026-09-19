import os
import pandas as pd

# ============================================================
# MAJOR DEPRESSIVE DISORDER — PGS PREPARATION
# PGS000907
# ============================================================

BASE = r"C:\PRS_Study"

PGS_FILE = os.path.join(
    BASE,
    "data",
    "PGS",
    "PGS000907.txt.gz"
)

OUTPUT_DIR = os.path.join(
    BASE,
    "results",
    "MDD",
    "PGS"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

# ============================================================
# Check input
# ============================================================

if not os.path.exists(PGS_FILE):

    raise FileNotFoundError(
        f"PGS file not found:\n{PGS_FILE}"
    )

print("=" * 70)
print("MAJOR DEPRESSIVE DISORDER — PGS PREPARATION")
print("PGS000907")
print("=" * 70)

# ============================================================
# Load PGS
# ============================================================

print("\nLoading PGS file...")

pgs = pd.read_csv(
    PGS_FILE,
    sep="\t",
    comment="#"
)

print(
    f"Loaded variants: {len(pgs)}"
)

# ============================================================
# Required columns
# ============================================================

required_columns = [
    "rsID",
    "chr_name",
    "chr_position",
    "effect_allele",
    "effect_weight"
]

for col in required_columns:

    if col not in pgs.columns:

        raise ValueError(
            f"Required column missing: {col}"
        )

# ============================================================
# Keep relevant columns
# ============================================================

pgs = pgs[
    [
        "rsID",
        "chr_name",
        "chr_position",
        "effect_allele",
        "effect_weight"
    ]
].copy()

# ============================================================
# Clean chromosome and position
# ============================================================

pgs["chr_name"] = pd.to_numeric(
    pgs["chr_name"],
    errors="coerce"
)

pgs["chr_position"] = pd.to_numeric(
    pgs["chr_position"],
    errors="coerce"
)

# ============================================================
# Clean allele and weight
# ============================================================

pgs["effect_allele"] = (
    pgs["effect_allele"]
    .astype(str)
    .str.upper()
    .str.strip()
)

pgs["effect_weight"] = pd.to_numeric(
    pgs["effect_weight"],
    errors="coerce"
)

# ============================================================
# Remove invalid records only
# ============================================================

before = len(pgs)

pgs = pgs.dropna(
    subset=[
        "chr_name",
        "chr_position",
        "effect_allele",
        "effect_weight"
    ]
).copy()

pgs["chr_name"] = pgs["chr_name"].astype(int)
pgs["chr_position"] = pgs["chr_position"].astype(int)

after = len(pgs)

print(
    f"After cleaning: {after}"
)

print(
    f"Records removed: {before - after}"
)

# ============================================================
# Keep autosomes only
# ============================================================

pgs = pgs[
    pgs["chr_name"].between(1, 22)
].copy()

print(
    f"After restricting to Chr1-22: {len(pgs)}"
)

# ============================================================
# Check duplicate chromosome-position records
# ============================================================

duplicate_positions = (
    pgs.duplicated(
        subset=[
            "chr_name",
            "chr_position"
        ],
        keep=False
    )
)

duplicate_count = duplicate_positions.sum()

print(
    f"Duplicate chromosome-position records: "
    f"{duplicate_count}"
)

if duplicate_count > 0:

    raise ValueError(
        "Duplicate chromosome-position records detected."
    )

# ============================================================
# Check effect alleles
# ============================================================

valid_alleles = {
    "A",
    "C",
    "G",
    "T"
}

invalid_alleles = ~pgs[
    "effect_allele"
].isin(valid_alleles)

print(
    f"Invalid effect alleles: "
    f"{invalid_alleles.sum()}"
)

if invalid_alleles.any():

    raise ValueError(
        "Invalid effect allele values detected."
    )

# ============================================================
# Check zero weights
# ============================================================

zero_weights = (
    pgs["effect_weight"] == 0
).sum()

print(
    f"Zero-weight variants retained: "
    f"{zero_weights}"
)

# ============================================================
# Create chromosome-specific files
# ============================================================

summary_rows = []

print("\nCreating chromosome-specific PGS files...")

for chrom in range(1, 23):

    chr_pgs = pgs[
        pgs["chr_name"] == chrom
    ].copy()

    chr_pgs = chr_pgs.sort_values(
        "chr_position"
    )

    # --------------------------------------------------------
    # PGS table
    # --------------------------------------------------------

    pgs_output = os.path.join(
        OUTPUT_DIR,
        f"MDD_chr{chrom}_PGS.tsv"
    )

    chr_pgs.to_csv(
        pgs_output,
        sep="\t",
        index=False
    )

    # --------------------------------------------------------
    # Position BED
    #
    # PLINK --extract bed1 expects:
    # CHR  POS  POS
    #
    # BED1 means positions are interpreted as 1-based.
    # --------------------------------------------------------

    bed_output = os.path.join(
        OUTPUT_DIR,
        f"MDD_chr{chrom}_positions.bed"
    )

    bed = pd.DataFrame(
        {
            "CHR": chr_pgs["chr_name"].astype(int),
            "POS1": chr_pgs["chr_position"].astype(int),
            "POS2": chr_pgs["chr_position"].astype(int)
        }
    )

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
            "PGS_variants": len(chr_pgs),
            "Unique_positions": chr_pgs[
                "chr_position"
            ].nunique()
        }
    )

    print(
        f"Chr{chrom}: {len(chr_pgs)} variants"
    )

# ============================================================
# Save chromosome summary
# ============================================================

summary = pd.DataFrame(
    summary_rows
)

summary_output = os.path.join(
    OUTPUT_DIR,
    "MDD_PGS_chromosome_summary.tsv"
)

summary.to_csv(
    summary_output,
    sep="\t",
    index=False
)

# ============================================================
# Final verification
# ============================================================

total_prepared = summary[
    "PGS_variants"
].sum()

print("\n")
print("=" * 70)
print("PREPARATION SUMMARY")
print("=" * 70)

print(
    f"Original variants: {before}"
)

print(
    f"Prepared variants: {total_prepared}"
)

print(
    f"Difference: {before - total_prepared}"
)

print("\nChromosome summary:")
print(summary.to_string(index=False))

if total_prepared != len(pgs):

    raise ValueError(
        "Chromosome totals do not match prepared PGS count."
    )

if total_prepared == 0:

    raise ValueError(
        "No variants were prepared."
    )

print("\nOutput directory:")
print(OUTPUT_DIR)

print("\nStatus: SUCCESS")
print("=" * 70)
