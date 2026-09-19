import pandas as pd
from pathlib import Path

# ============================================================
# T2D PGS PREPARATION
# ============================================================

BASE = Path(r"C:\PRS_Study")

PGS_FILE = (
    BASE /
    "data" /
    "PGS" /
    "PGS000036.txt.gz"
)

OUTPUT_DIR = (
    BASE /
    "results" /
    "T2D" /
    "PGS"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

SUMMARY_FILE = (
    OUTPUT_DIR /
    "T2D_PGS_chromosome_summary.tsv"
)

print("=" * 90)
print("TYPE 2 DIABETES — PGS PREPARATION")
print("=" * 90)

# ============================================================
# 1. LOAD PGS
# ============================================================

print("\nLoading PGS000036...")

df = pd.read_csv(
    PGS_FILE,
    sep="\t",
    comment="#",
    compression="gzip"
)

print(f"Loaded variants: {len(df):,}")

# ============================================================
# 2. KEEP REQUIRED COLUMNS
# ============================================================

required = [
    "rsID",
    "chr_name",
    "chr_position",
    "effect_allele",
    "other_allele",
    "effect_weight"
]

missing_columns = [
    c for c in required
    if c not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

df = df[required].copy()

# ============================================================
# 3. BASIC CLEANING
# ============================================================

df["chr_name"] = (
    df["chr_name"]
    .astype(str)
    .str.replace("chr", "", regex=False)
)

df["chr_position"] = pd.to_numeric(
    df["chr_position"],
    errors="coerce"
)

df["effect_weight"] = pd.to_numeric(
    df["effect_weight"],
    errors="coerce"
)

# ============================================================
# 4. ESSENTIAL FIELD CHECK
# ============================================================

essential_missing = df[
    [
        "rsID",
        "chr_name",
        "chr_position",
        "effect_allele",
        "effect_weight"
    ]
].isna().any(axis=1)

print(
    f"Missing essential fields: "
    f"{essential_missing.sum()}"
)

if essential_missing.sum() > 0:
    df = df.loc[
        ~essential_missing
    ].copy()

# ============================================================
# 5. CHROMOSOME FILTER
# ============================================================

valid_chromosomes = {
    str(i)
    for i in range(1, 23)
}

valid_chr = df["chr_name"].isin(
    valid_chromosomes
)

print(
    f"Variants on chromosomes 1–22: "
    f"{valid_chr.sum():,}"
)

df = df.loc[
    valid_chr
].copy()

# ============================================================
# 6. EFFECT ALLELE VALIDATION
# ============================================================

valid_alleles = {
    "A",
    "C",
    "G",
    "T"
}

invalid_allele = ~df[
    "effect_allele"
].isin(valid_alleles)

print(
    f"Invalid effect allele records: "
    f"{invalid_allele.sum()}"
)

if invalid_allele.sum() > 0:
    df = df.loc[
        ~invalid_allele
    ].copy()

# ============================================================
# 7. DUPLICATE CHECKS
# ============================================================

duplicate_positions = df.duplicated(
    subset=[
        "chr_name",
        "chr_position"
    ]
).sum()

duplicate_rsids = df.duplicated(
    subset=["rsID"]
).sum()

print(
    f"Duplicate chromosome-position records: "
    f"{duplicate_positions}"
)

print(
    f"Duplicate rsID records: "
    f"{duplicate_rsids}"
)

# Do NOT remove duplicate positions automatically.
# They are checked here so the scoring stage can handle
# compatibility and positional duplicates explicitly.

# ============================================================
# 8. ZERO WEIGHT CHECK
# ============================================================

zero_weights = (
    df["effect_weight"] == 0
).sum()

print(
    f"Zero-weight variants retained: "
    f"{zero_weights}"
)

# ============================================================
# 9. SAVE CHROMOSOME-SPECIFIC FILES
# ============================================================

summary = []

print("\nCreating chromosome-specific PGS files...")

for chromosome in range(1, 23):

    chr_str = str(chromosome)

    chr_df = df[
        df["chr_name"] == chr_str
    ].copy()

    chr_df = chr_df.sort_values(
        "chr_position"
    )

    output_file = (
        OUTPUT_DIR /
        f"T2D_chr{chromosome}_PGS.tsv"
    )

    chr_df.to_csv(
        output_file,
        sep="\t",
        index=False
    )

    summary.append({
        "Chromosome": chromosome,
        "PGS_Variants": len(chr_df),
        "Output_File": output_file.name
    })

    print(
        f"Chr{chromosome}: "
        f"{len(chr_df):,} variants"
    )

# ============================================================
# 10. SAVE SUMMARY
# ============================================================

summary_df = pd.DataFrame(
    summary
)

summary_df.to_csv(
    SUMMARY_FILE,
    sep="\t",
    index=False
)

# ============================================================
# 11. FINAL VALIDATION
# ============================================================

total_prepared = (
    summary_df["PGS_Variants"]
    .sum()
)

print("\n" + "=" * 90)
print("T2D PGS PREPARATION COMPLETE")
print("=" * 90)

print(
    f"\nOriginal variants: "
    f"{len(df):,}"
)

print(
    f"Prepared variants: "
    f"{total_prepared:,}"
)

print(
    f"Difference: "
    f"{len(df) - total_prepared:,}"
)

print("\nOutput directory:")
print(OUTPUT_DIR)

print("\nSummary file:")
print(SUMMARY_FILE)
