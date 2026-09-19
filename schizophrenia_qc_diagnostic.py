import pandas as pd
from pathlib import Path

BASE = Path(r"C:\PRS_Study\results\Schizophrenia")

rows = []

for chrom in range(1, 23):

    # Find the existing QC file
    candidates = list(
        BASE.rglob(f"Schizophrenia_chr{chrom}_QC.tsv")
    )

    if not candidates:
        candidates = list(
            BASE.rglob(f"Schizophrenia_chr{chrom}_QC_summary.tsv")
        )

    if not candidates:
        print(f"Chr{chrom}: QC FILE NOT FOUND")
        continue

    file = candidates[0]

    df = pd.read_csv(
        file,
        sep="\t"
    )

    r = df.iloc[0]

    # Handle the actual Schizophrenia column names
    pgs = pd.to_numeric(
        r.get("PGS_variants"),
        errors="coerce"
    )

    coordinate = pd.to_numeric(
        r.get("coordinate_matches"),
        errors="coerce"
    )

    alt = pd.to_numeric(
        r.get("ALT_effect", 0),
        errors="coerce"
    )

    ref = pd.to_numeric(
        r.get("REF_effect", 0),
        errors="coerce"
    )

    mismatch = pd.to_numeric(
        r.get("MISMATCH", 0),
        errors="coerce"
    )

    scored = pd.to_numeric(
        r.get("scored_variants"),
        errors="coerce"
    )

    rows.append({
        "Chromosome": chrom,
        "PGS_variants": pgs,
        "Coordinate_matches": coordinate,
        "ALT_effect": alt,
        "REF_effect": ref,
        "MISMATCH": mismatch,
        "Scored_variants": scored,

        "Coordinate_excess":
            coordinate - pgs,

        "Scored_excess":
            scored - pgs,

        "Coordinate_rate":
            coordinate / pgs * 100,

        "Scoring_rate":
            scored / pgs * 100
    })


# ============================================================
# CREATE TABLE
# ============================================================

result = pd.DataFrame(rows)

result = result.sort_values(
    "Chromosome"
).reset_index(drop=True)


# ============================================================
# DISPLAY CHROMOSOME DETAILS
# ============================================================

print("\n" + "=" * 100)
print("SCHIZOPHRENIA QC DIAGNOSTIC")
print("=" * 100)

print(
    result.to_string(
        index=False
    )
)


# ============================================================
# SHOW CHROMOSOMES WITH EXCESS COUNTS
# ============================================================

print("\n" + "=" * 100)
print("CHROMOSOMES WITH COORDINATE-MATCH EXCESS")
print("=" * 100)

excess_coordinate = result[
    result["Coordinate_excess"] > 0
]

if len(excess_coordinate) == 0:

    print("NONE")

else:

    print(
        excess_coordinate[
            [
                "Chromosome",
                "PGS_variants",
                "Coordinate_matches",
                "Coordinate_excess",
                "Coordinate_rate"
            ]
        ].to_string(
            index=False
        )
    )


# ============================================================
# SHOW CHROMOSOMES WITH SCORING EXCESS
# ============================================================

print("\n" + "=" * 100)
print("CHROMOSOMES WITH SCORING EXCESS")
print("=" * 100)

excess_scoring = result[
    result["Scored_excess"] > 0
]

if len(excess_scoring) == 0:

    print("NONE")

else:

    print(
        excess_scoring[
            [
                "Chromosome",
                "PGS_variants",
                "Scored_variants",
                "Scored_excess",
                "Scoring_rate"
            ]
        ].to_string(
            index=False
        )
    )


# ============================================================
# TOTALS
# ============================================================

print("\n" + "=" * 100)
print("TOTALS")
print("=" * 100)

total_pgs = result[
    "PGS_variants"
].sum()

total_coordinate = result[
    "Coordinate_matches"
].sum()

total_scored = result[
    "Scored_variants"
].sum()


print(
    f"Total PGS variants:        {total_pgs:,.0f}"
)

print(
    f"Total coordinate matches:  {total_coordinate:,.0f}"
)

print(
    f"Total scored variants:     {total_scored:,.0f}"
)

print(
    f"Coordinate difference:     "
    f"{total_coordinate - total_pgs:,.0f}"
)

print(
    f"Scoring difference:        "
    f"{total_scored - total_pgs:,.0f}"
)

print(
    f"Coordinate rate:           "
    f"{total_coordinate / total_pgs * 100:.6f}%"
)

print(
    f"Scoring rate:              "
    f"{total_scored / total_pgs * 100:.6f}%"
)


# ============================================================
# ALLELE TOTALS
# ============================================================

print("\n" + "=" * 100)
print("ALLELE / MISMATCH TOTALS")
print("=" * 100)

print(
    f"ALT effect:    "
    f"{result['ALT_effect'].sum():,.0f}"
)

print(
    f"REF effect:    "
    f"{result['REF_effect'].sum():,.0f}"
)

print(
    f"MISMATCH:      "
    f"{result['MISMATCH'].sum():,.0f}"
)


print("\nDiagnostic complete.")
