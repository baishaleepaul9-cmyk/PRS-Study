import pandas as pd
from pathlib import Path

base = Path(r"C:\PRS_Study\results\CAD")

dfs = []

for chr_num in range(1, 23):
    f = base / f"CAD_chr{chr_num}" / f"CAD_chr{chr_num}_PRS.tsv"

    if not f.exists():
        raise FileNotFoundError(f"Missing: {f}")

    df = pd.read_csv(f, sep="\t")

    iid_col = None
    if "IID" in df.columns:
        iid_col = "IID"
    elif "#IID" in df.columns:
        iid_col = "#IID"
    else:
        raise ValueError(
            f"No IID column found in {f}\n"
            f"Columns: {list(df.columns)}"
        )

    prs_cols = [
        c for c in df.columns
        if c not in [iid_col, "FID", "#FID"]
    ]

    if len(prs_cols) != 1:
        raise ValueError(
            f"Could not identify PRS column in {f}\n"
            f"Columns: {list(df.columns)}"
        )

    prs_col = prs_cols[0]

    temp = df[[iid_col, prs_col]].copy()
    temp.columns = ["IID", f"CHR{chr_num}"]

    dfs.append(temp)

    print(
        f"Chr{chr_num}: {len(temp)} samples | "
        f"PRS column: {prs_col}"
    )

print("\nMerging chromosome scores...")

merged = dfs[0]

for df in dfs[1:]:
    merged = merged.merge(df, on="IID", how="inner")

chr_cols = [f"CHR{i}" for i in range(1, 23)]

merged["PRS"] = merged[chr_cols].sum(axis=1)

genomewide = merged[["IID", "PRS"]]

out = base / "CAD_genomewide_PRS.tsv"

genomewide.to_csv(out, sep="\t", index=False)

print("\n" + "=" * 80)
print("CAD GENOME-WIDE PRS")
print("=" * 80)

print(f"Samples: {len(genomewide)}")
print(f"Missing PRS: {genomewide['PRS'].isna().sum()}")
print(f"Mean: {genomewide['PRS'].mean():.8f}")
print(f"SD: {genomewide['PRS'].std():.8f}")
print(f"Median: {genomewide['PRS'].median():.8f}")
print(f"Min: {genomewide['PRS'].min():.8f}")
print(f"Q1: {genomewide['PRS'].quantile(0.25):.8f}")
print(f"Q3: {genomewide['PRS'].quantile(0.75):.8f}")
print(f"Max: {genomewide['PRS'].max():.8f}")

print("\nSaved:")
print(out)
