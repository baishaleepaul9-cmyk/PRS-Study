import pandas as pd
from pathlib import Path

BASE = Path(r"C:\PRS_Study\results\Schizophrenia")

CHROMS = [1, 2, 7]


def read_table(path):
    return pd.read_csv(path, sep="\t", dtype=str)


for chrom in CHROMS:

    folder = BASE / f"Schizophrenia_chr{chrom}"

    print(f"\n{'='*60}")
    print(f"Processing Schizophrenia chromosome {chrom}")
    print(f"{'='*60}")

    # ---------------------------------------------------------
    # 1. PGS file
    # ---------------------------------------------------------
    pgs_file = folder / f"Schizophrenia_chr{chrom}_PGS.tsv"
    pgs = read_table(pgs_file)

    pgs_variants = len(pgs)

    # ---------------------------------------------------------
    # 2. Harmonized PVAR
    # ---------------------------------------------------------
    pvar_file = folder / f"Schizophrenia_chr{chrom}_harmonized.pvar"

    pvar = pd.read_csv(
        pvar_file,
        sep="\t",
        comment="#",
        header=None,
        names=[
            "CHR",
            "POS",
            "ID",
            "REF",
            "ALT",
            "QUAL",
            "FILTER",
            "INFO"
        ],
        dtype=str
    )

    # ---------------------------------------------------------
    # 3. Coordinate matching
    # ---------------------------------------------------------
    pgs["hm_chr"] = pd.to_numeric(pgs["hm_chr"], errors="coerce")
    pgs["hm_pos"] = pd.to_numeric(pgs["hm_pos"], errors="coerce")

    pgs_coord = pgs[
        pgs["hm_chr"].notna() &
        pgs["hm_pos"].notna()
    ].copy()

    pgs_coord["CHR"] = pgs_coord["hm_chr"].astype(int).astype(str)
    pgs_coord["POS"] = pgs_coord["hm_pos"].astype(int).astype(str)

    pvar["CHR"] = pvar["CHR"].astype(str)
    pvar["POS"] = pvar["POS"].astype(str)

    matched = pgs_coord.merge(
        pvar[["CHR", "POS", "ID", "REF", "ALT"]],
        on=["CHR", "POS"],
        how="inner"
    )

    coordinate_matches = len(matched)

    # ---------------------------------------------------------
    # 4. Allele compatibility
    #
    # ALT_effect = effect allele equals ALT
    # REF_effect = effect allele equals REF
    # MISMATCH   = effect allele matches neither
    # ---------------------------------------------------------
    matched["effect_allele"] = matched["effect_allele"].str.upper()
    matched["REF"] = matched["REF"].str.upper()
    matched["ALT"] = matched["ALT"].str.upper()

    alt_effect = (
        matched["effect_allele"] == matched["ALT"]
    )

    ref_effect = (
        matched["effect_allele"] == matched["REF"]
    )

    mismatch = ~(alt_effect | ref_effect)

    alt_count = int(alt_effect.sum())
    ref_count = int(ref_effect.sum())
    mismatch_count = int(mismatch.sum())

    # ---------------------------------------------------------
    # 5. Final PLINK scoring file
    # ---------------------------------------------------------
    score_file = folder / f"Schizophrenia_chr{chrom}_PLINK_score.txt"

    score = pd.read_csv(
        score_file,
        sep=r"\s+",
        header=None,
        names=["ID", "EFFECT_ALLELE", "EFFECT_WEIGHT"],
        dtype=str
    )

    scored_variants = len(score)

    # ---------------------------------------------------------
    # 6. Existing PLINK .sscore
    # ---------------------------------------------------------
    sscore_file = folder / f"Schizophrenia_chr{chrom}_PRS.sscore"

    sscore = pd.read_csv(
        sscore_file,
        sep=r"\s+"
    )

    samples = len(sscore)

    # SCORE1_SUM is the chromosome-level PRS
    prs = pd.to_numeric(
        sscore["SCORE1_SUM"],
        errors="coerce"
    )

    missing_scores = int(prs.isna().sum())

    valid_prs = prs.dropna()

    prs_mean = valid_prs.mean()
    prs_sd = valid_prs.std()
    prs_min = valid_prs.min()
    prs_max = valid_prs.max()

    # ---------------------------------------------------------
    # 7. Build QC row
    # ---------------------------------------------------------
    qc = pd.DataFrame([{
        "chromosome": chrom,
        "PGS_variants": pgs_variants,
        "coordinate_matches": coordinate_matches,
        "ALT_effect": alt_count,
        "REF_effect": ref_count,
        "MISMATCH": mismatch_count,
        "scored_variants": scored_variants,
        "samples": samples,
        "missing_scores": missing_scores,
        "PRS_mean": prs_mean,
        "PRS_SD": prs_sd,
        "PRS_min": prs_min,
        "PRS_max": prs_max
    }])

    output = folder / f"Schizophrenia_chr{chrom}_QC.tsv"

    qc.to_csv(
        output,
        sep="\t",
        index=False
    )

    print(qc.to_string(index=False))
    print(f"\nSaved: {output}")


print("\n" + "="*60)
print("DONE — Chr1, Chr2 and Chr7 QC reconstructed")
print("="*60)
