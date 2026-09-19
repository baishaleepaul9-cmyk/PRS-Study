import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path(r"C:\Users\baish\OneDrive\文档\PRS Study")
RESULTS = BASE / "results" / "Alzheimer"

rows = []


# ------------------------------------------------------------
# Helper: PRS distribution from existing raw PRS file
# ------------------------------------------------------------
def get_prs_stats(chrom):
    f = RESULTS / f"chr{chrom}" / f"chr{chrom}_raw_prs.tsv"

    if not f.exists():
        return [np.nan] * 5

    df = pd.read_csv(f, sep="\t")

    prs_cols = [c for c in df.columns if c != "#IID"]

    if len(prs_cols) != 1:
        return [np.nan] * 5

    s = pd.to_numeric(df[prs_cols[0]], errors="coerce")

    return [
        int(s.notna().sum()),
        s.mean(),
        s.std(),
        s.min(),
        s.max()
    ]


# ------------------------------------------------------------
# 1. Known QC: Chr1
# ------------------------------------------------------------
known = {
    1: {
        "PGS_variants": 90652,
        "Coordinate_matches": 90651,
        "Unmatched": 1,
        "Eligible_variants": 90651,
        "Duplicate_TARGET_ID": 0,
        "Direct": 90098,
        "Reverse": 418,
        "Strand": 95,
        "Strand_reverse": 40,
        "Incompatible": 0,
        "Score_variants": 90651,
        "Matched_PLINK_IDs": 90651,
        "Missing_PLINK_IDs": 0,
    },

    3: {
        "PGS_variants": 76485,
        "Coordinate_matches": 76485,
        "Unmatched": 0,
        "Eligible_variants": 76485,
        "Duplicate_TARGET_ID": 0,
        "Direct": 76363,
        "Reverse": 122,
        "Strand": 0,
        "Strand_reverse": 0,
        "Incompatible": 0,
        "Score_variants": 76485,
        "Matched_PLINK_IDs": 76485,
        "Missing_PLINK_IDs": 0,
    },

    4: {
        "PGS_variants": 68316,
        "Coordinate_matches": 68316,
        "Unmatched": 0,
        "Eligible_variants": 68316,
        "Duplicate_TARGET_ID": 0,
        "Direct": 68289,
        "Reverse": 27,
        "Strand": 0,
        "Strand_reverse": 0,
        "Incompatible": 0,
        "Score_variants": 68316,
        "Matched_PLINK_IDs": 68316,
        "Missing_PLINK_IDs": 0,
    },
}


# ------------------------------------------------------------
# 2. Existing Chr5–Chr8 progress file
# ------------------------------------------------------------
f58 = RESULTS / "chr5_to_chr22_progress.tsv"

if f58.exists():

    df58 = pd.read_csv(f58, sep="\t")

    for _, r in df58.iterrows():

        chrom = int(r["chromosome"])

        if chrom not in [5, 6, 7, 8]:
            continue

        known[chrom] = {
            "PGS_variants": int(r["PGS_variants"]),
            "Coordinate_matches": int(r["coordinate_matches"]),
            "Unmatched": int(r["unmatched"]),
            "Eligible_variants": int(
                r["coordinate_matches"] - r["incompatible"]
            ),
            "Duplicate_TARGET_ID": int(r["duplicate_TARGET_ID"]),
            "Direct": np.nan,
            "Reverse": np.nan,
            "Strand": np.nan,
            "Strand_reverse": np.nan,
            "Incompatible": int(r["incompatible"]),
            "Score_variants": int(r["score_variants"]),
            "Matched_PLINK_IDs": np.nan,
            "Missing_PLINK_IDs": np.nan,
        }


# ------------------------------------------------------------
# 3. Existing Chr12–Chr22 QC summary
# ------------------------------------------------------------
f922 = RESULTS / "chr9_to_chr22_qc_summary.tsv"

if f922.exists():

    df922 = pd.read_csv(f922, sep="\t")

    for _, r in df922.iterrows():

        chrom = int(r["chromosome"])

        if chrom < 12:
            continue

        known[chrom] = {
            "PGS_variants": int(r["PGS_variants"]),
            "Coordinate_matches": int(r["coordinate_matches"]),
            "Unmatched": (
                int(r["PGS_variants"])
                - int(r["coordinate_matches"])
            ),
            "Eligible_variants": int(r["eligible_variants"]),
            "Duplicate_TARGET_ID": int(r["duplicate_target_ids"]),
            "Direct": int(r["direct"]),
            "Reverse": int(r["reverse"]),
            "Strand": int(r["strand"]),
            "Strand_reverse": int(r["strand_reverse"]),
            "Incompatible": int(r["incompatible"]),
            "Score_variants": int(r["score_variants"]),
            "Matched_PLINK_IDs": int(r["matched_plink_ids"]),
            "Missing_PLINK_IDs": int(r["missing_plink_ids"]),
        }


# ------------------------------------------------------------
# 4. Chr2 from existing harmonized file + score output
# ------------------------------------------------------------
f2 = RESULTS / "chr2" / "chr2_harmonized.tsv"

if f2.exists():

    d2 = pd.read_csv(f2, sep="\t")

    orientation_counts = d2["orientation"].value_counts()

    score_log_variants = 91893
    incompatible = 1

    known[2] = {
        "PGS_variants": len(d2),
        "Coordinate_matches": len(d2),
        "Unmatched": 0,
        "Eligible_variants": len(d2) - incompatible,
        "Duplicate_TARGET_ID": int(
            d2["TARGET_ID"].duplicated().sum()
        ),
        "Direct": int(orientation_counts.get("direct", 0)),
        "Reverse": int(orientation_counts.get("reverse", 0)),
        "Strand": int(orientation_counts.get("strand", 0)),
        "Strand_reverse": int(
            orientation_counts.get("strand_reverse", 0)
        ),
        "Incompatible": incompatible,
        "Score_variants": score_log_variants,
        "Matched_PLINK_IDs": np.nan,
        "Missing_PLINK_IDs": np.nan,
    }


# ------------------------------------------------------------
# 5. Chr9–Chr11
#
# Their saved PGS files contain the variants that reached the
# scoring stage. The original orientation breakdown was not
# preserved, so we deliberately leave those fields NA.
# ------------------------------------------------------------
score_processed = {
    9: 50588,
    10: 58420,
    11: 55886,
}

score_skipped = {
    9: 0,
    10: 216,
    11: 52,
}

for chrom in [9, 10, 11]:

    pgs_file = (
        RESULTS /
        f"chr{chrom}" /
        f"chr{chrom}_pgs.tsv"
    )

    if not pgs_file.exists():
        continue

    d = pd.read_csv(pgs_file, sep="\t")

    pgs_n = len(d)

    known[chrom] = {
        "PGS_variants": pgs_n,
        "Coordinate_matches": pgs_n,
        "Unmatched": 0,
        "Eligible_variants": (
            pgs_n - score_skipped[chrom]
        ),
        "Duplicate_TARGET_ID": np.nan,
        "Direct": np.nan,
        "Reverse": np.nan,
        "Strand": np.nan,
        "Strand_reverse": np.nan,
        "Incompatible": score_skipped[chrom],
        "Score_variants": score_processed[chrom],
        "Matched_PLINK_IDs": np.nan,
        "Missing_PLINK_IDs": np.nan,
    }


# ------------------------------------------------------------
# 6. Build final 22-chromosome table
# ------------------------------------------------------------
for chrom in range(1, 23):

    stats = get_prs_stats(chrom)

    samples, mean_prs, sd_prs, min_prs, max_prs = stats

    if chrom in known:

        q = known[chrom]

        row = {
            "Chromosome": chrom,
            **q,
            "Samples": samples,
            "PRS_mean": mean_prs,
            "PRS_SD": sd_prs,
            "PRS_min": min_prs,
            "PRS_max": max_prs,
            "Status": "SUCCESS"
        }

    else:

        row = {
            "Chromosome": chrom,
            "Status": "QC_DATA_NOT_AVAILABLE"
        }

    rows.append(row)


# ------------------------------------------------------------
# 7. Save
# ------------------------------------------------------------
final = pd.DataFrame(rows)

columns = [
    "Chromosome",
    "PGS_variants",
    "Coordinate_matches",
    "Unmatched",
    "Eligible_variants",
    "Duplicate_TARGET_ID",
    "Direct",
    "Reverse",
    "Strand",
    "Strand_reverse",
    "Incompatible",
    "Score_variants",
    "Matched_PLINK_IDs",
    "Missing_PLINK_IDs",
    "Samples",
    "PRS_mean",
    "PRS_SD",
    "PRS_min",
    "PRS_max",
    "Status"
]

final = final[columns]

out = RESULTS / "Alzheimer_final_QC_table.tsv"

final.to_csv(out, sep="\t", index=False)

print("\n" + "=" * 110)
print("FINAL ALZHEIMER'S DISEASE QC TABLE")
print("=" * 110)

print(final.to_string(index=False))

print("\n" + "-" * 110)

success = (final["Status"] == "SUCCESS").sum()

print(f"QC rows available: {success}/22")

if success < 22:
    missing = final.loc[
        final["Status"] != "SUCCESS",
        "Chromosome"
    ].tolist()

    print("QC data unavailable for:", missing)

print("\nSaved:")
print(out)

print("\nIMPORTANT:")
print("No PLINK command was executed.")
print("No VCF was processed.")
print("No chromosome PRS was recalculated.")
