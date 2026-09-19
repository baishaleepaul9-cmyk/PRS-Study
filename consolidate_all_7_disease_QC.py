import pandas as pd
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

BASE = Path(r"C:\PRS_Study\results")

OUT = BASE / "combined_analysis"
OUT.mkdir(parents=True, exist_ok=True)


# ============================================================
# DISEASE CONFIGURATION
# ============================================================

DISEASES = {

    "Alzheimers": {
        "folder": BASE / "Alzheimer",
        "prefix": "Alzheimer",
        "type": "alzheimer"
    },

    "Schizophrenia": {
        "folder": BASE / "Schizophrenia",
        "prefix": "Schizophrenia",
        "type": "standard"
    },

    "Multiple_Sclerosis": {
        "folder": BASE / "Multiple_Sclerosis",
        "prefix": "MS",
        "type": "standard"
    },

    "MDD": {
        "folder": BASE / "MDD",
        "prefix": "MDD",
        "type": "standard"
    },

    "Parkinsons": {
        "folder": BASE / "Parkinsons",
        "prefix": "Parkinsons",
        "type": "standard"
    },

    "CAD": {
        "folder": BASE / "CAD",
        "prefix": "CAD",
        "type": "standard"
    },

    "T2D": {
        "folder": BASE / "T2D",
        "prefix": "T2D",
        "type": "t2d"
    }
}


# ============================================================
# STANDARD OUTPUT COLUMNS
# ============================================================

STANDARD_COLUMNS = [

    "Disease",
    "Chromosome",

    "PGS_variants",
    "Coordinate_matches",

    "ALT_effect",
    "REF_effect",
    "MISMATCH",

    "Final_scoring_variants",

    "Samples",
    "Missing_scores",

    "PRS_mean",
    "PRS_SD",
    "PRS_min",
    "PRS_max",

    "Retention_percent"
]


# ============================================================
# HELPER: GET VALUE
# ============================================================

def get_value(row, possible_names, default=pd.NA):

    for name in possible_names:

        if name in row.index:

            return row[name]

    return default


# ============================================================
# HELPER: NUMERIC
# ============================================================

def to_number(value):

    try:

        return pd.to_numeric(
            value,
            errors="coerce"
        )

    except Exception:

        return pd.NA


# ============================================================
# HELPER: CREATE STANDARD ROW
# ============================================================

def make_row(
    disease,
    chrom,
    pgs,
    coordinate,
    alt,
    ref,
    mismatch,
    final,
    samples,
    missing,
    mean=pd.NA,
    sd=pd.NA,
    minimum=pd.NA,
    maximum=pd.NA
):

    return {

        "Disease": disease,

        "Chromosome": chrom,

        "PGS_variants": pgs,

        "Coordinate_matches": coordinate,

        "ALT_effect": alt,

        "REF_effect": ref,

        "MISMATCH": mismatch,

        "Final_scoring_variants": final,

        "Samples": samples,

        "Missing_scores": missing,

        "PRS_mean": mean,

        "PRS_SD": sd,

        "PRS_min": minimum,

        "PRS_max": maximum
    }


# ============================================================
# ALZHEIMER
# ============================================================

def load_alzheimer():

    folder = DISEASES[
        "Alzheimers"
    ]["folder"]

    file = (
        folder /
        "Alzheimer_final_QC_table_corrected.tsv"
    )

    if not file.exists():

        raise FileNotFoundError(
            f"\nAlzheimer QC file not found:\n{file}"
        )

    print(
        f"\nUsing Alzheimer QC: {file.name}"
    )

    df = pd.read_csv(
        file,
        sep="\t"
    )

    rows = []

    for _, r in df.iterrows():

        chrom = to_number(
            get_value(
                r,
                ["Chromosome"]
            )
        )

        if pd.isna(chrom):

            continue

        chrom = int(chrom)

        if chrom < 1 or chrom > 22:

            continue


        # ----------------------------------------------------
        # Alzheimer orientation counts
        # ----------------------------------------------------

        direct = to_number(
            get_value(
                r,
                ["Direct"],
                0
            )
        )

        reverse = to_number(
            get_value(
                r,
                ["Reverse"],
                0
            )
        )

        strand = to_number(
            get_value(
                r,
                ["Strand"],
                0
            )
        )

        strand_reverse = to_number(
            get_value(
                r,
                ["Strand_reverse"],
                0
            )
        )

        incompatible = to_number(
            get_value(
                r,
                ["Incompatible"],
                0
            )
        )


        alt = (
            direct +
            strand
        )

        ref = (
            reverse +
            strand_reverse
        )


        # ----------------------------------------------------
        # Row
        # ----------------------------------------------------

        rows.append(
            make_row(

                "Alzheimers",

                chrom,

                get_value(
                    r,
                    ["PGS_variants"]
                ),

                get_value(
                    r,
                    ["Coordinate_matches"]
                ),

                alt,

                ref,

                incompatible,

                get_value(
                    r,
                    ["Score_variants"]
                ),

                get_value(
                    r,
                    ["Samples"]
                ),

                get_value(
                    r,
                    ["Missing_PLINK_IDs"],
                    0
                ),

                get_value(
                    r,
                    ["PRS_mean"]
                ),

                get_value(
                    r,
                    ["PRS_SD"]
                ),

                get_value(
                    r,
                    ["PRS_min"]
                ),

                get_value(
                    r,
                    ["PRS_max"]
                )
            )
        )

    return pd.DataFrame(rows)


# ============================================================
# FIND QC FILE
#
# This searches recursively, so MS files can be anywhere
# inside the Multiple_Sclerosis results directory.
# ============================================================

def find_qc_file(
    folder,
    prefix,
    chrom
):

    possible_names = [

        f"{prefix}_chr{chrom}_QC.tsv",

        f"{prefix}_chr{chrom}_QC_summary.tsv"
    ]


    # --------------------------------------------------------
    # First: exact recursive search
    # --------------------------------------------------------

    for filename in possible_names:

        matches = list(
            folder.rglob(filename)
        )

        if len(matches) > 0:

            return matches[0]


    # --------------------------------------------------------
    # Second: case-insensitive fallback
    # --------------------------------------------------------

    wanted = {

        name.lower()

        for name in possible_names
    }


    for file in folder.rglob("*.tsv"):

        if file.name.lower() in wanted:

            return file


    return None


# ============================================================
# STANDARD DISEASE LOADER
#
# Schizophrenia
# Multiple Sclerosis
# MDD
# Parkinsons
# CAD
# ============================================================

def load_standard(
    disease,
    folder,
    prefix
):

    rows = []

    for chrom in range(1, 23):

        qc_file = find_qc_file(
            folder,
            prefix,
            chrom
        )


        # ----------------------------------------------------
        # Missing
        # ----------------------------------------------------

        if qc_file is None:

            print(
                f"  Chr{chrom}: MISSING"
            )

            continue


        # ----------------------------------------------------
        # Read
        # ----------------------------------------------------

        try:

            df = pd.read_csv(
                qc_file,
                sep="\t"
            )

        except Exception as e:

            raise RuntimeError(

                f"\nCould not read:\n"
                f"{qc_file}\n\n"
                f"{e}"
            )


        if len(df) == 0:

            raise ValueError(

                f"\nQC file is empty:\n"
                f"{qc_file}"
            )


        # ----------------------------------------------------
        # Find chromosome column
        # ----------------------------------------------------

        chrom_col = None

        for candidate in [

            "Chromosome",
            "chromosome"

        ]:

            if candidate in df.columns:

                chrom_col = candidate

                break


        # ----------------------------------------------------
        # Select row
        # ----------------------------------------------------

        if chrom_col is not None:

            df[chrom_col] = pd.to_numeric(

                df[chrom_col],

                errors="coerce"
            )


            matching = df[
                df[chrom_col] == chrom
            ]


            if len(matching) > 0:

                r = matching.iloc[0]

            else:

                r = df.iloc[0]

        else:

            r = df.iloc[0]


        # ====================================================
        # PGS variants
        # ====================================================

        pgs = get_value(

            r,

            [
                "PGS_variants",
                "PGS_Variants"
            ]
        )


        # ====================================================
        # Coordinate matches
        # ====================================================

        coordinate = get_value(

            r,

            [
                "Coordinate_matches",
                "Coordinate_Matches",
                "coordinate_matches"
            ]
        )


        # ====================================================
        # ALT
        # ====================================================

        alt = get_value(

            r,

            [
                "ALT_effect"
            ],

            0
        )


        # ====================================================
        # REF
        # ====================================================

        ref = get_value(

            r,

            [
                "REF_effect"
            ],

            0
        )


        # ====================================================
        # MISMATCH
        # ====================================================

        mismatch = get_value(

            r,

            [
                "MISMATCH"
            ],

            0
        )


        # ====================================================
        # FINAL SCORING VARIANTS
        # ====================================================

        final = get_value(

            r,

            [
                "Final_scoring_variants",
                "Final_Scoring_Variants",
                "scored_variants",
                "Scoring_variants_used"
            ]
        )


        if pd.isna(final):

            raise ValueError(

                f"\n{disease} Chr{chrom}: "
                f"final scoring column not found.\n\n"

                f"File:\n"
                f"{qc_file}\n\n"

                f"Available columns:\n"
                f"{list(df.columns)}"
            )


        # ====================================================
        # SAMPLES
        # ====================================================

        samples = get_value(

            r,

            [
                "Samples",
                "samples"
            ]
        )


        # ====================================================
        # MISSING SCORES
        # ====================================================

        missing = get_value(

            r,

            [
                "Missing_scores",
                "missing_scores",
                "Missing_PRS"
            ],

            0
        )


        # ====================================================
        # PRS MEAN
        # ====================================================

        mean = get_value(

            r,

            [
                "PRS_mean",
                "Mean_PRS"
            ]
        )


        # ====================================================
        # PRS SD
        # ====================================================

        sd = get_value(

            r,

            [
                "PRS_SD",
                "SD_PRS"
            ]
        )


        # ====================================================
        # PRS MIN
        # ====================================================

        minimum = get_value(

            r,

            [
                "PRS_min",
                "Min_PRS"
            ]
        )


        # ====================================================
        # PRS MAX
        # ====================================================

        maximum = get_value(

            r,

            [
                "PRS_max",
                "Max_PRS"
            ]
        )


        # ====================================================
        # APPEND
        # ====================================================

        rows.append(

            make_row(

                disease,

                chrom,

                pgs,

                coordinate,

                alt,

                ref,

                mismatch,

                final,

                samples,

                missing,

                mean,

                sd,

                minimum,

                maximum
            )
        )


        print(

            f"  Chr{chrom}: "
            f"{qc_file.name}"
        )


    return pd.DataFrame(rows)


# ============================================================
# T2D
# ============================================================

def load_t2d():

    folder = DISEASES[
        "T2D"
    ]["folder"]

    rows = []

    for chrom in range(1, 23):

        qc_file = find_qc_file(

            folder,

            "T2D",

            chrom
        )


        if qc_file is None:

            print(
                f"  Chr{chrom}: MISSING"
            )

            continue


        df = pd.read_csv(

            qc_file,

            sep="\t"
        )


        if len(df) == 0:

            raise ValueError(

                f"\nT2D QC file is empty:\n"
                f"{qc_file}"
            )


        r = df.iloc[0]


        rows.append(

            make_row(

                "T2D",

                chrom,

                get_value(

                    r,

                    [
                        "PGS_Variants",
                        "PGS_variants"
                    ]
                ),

                get_value(

                    r,

                    [
                        "Coordinate_Matches",
                        "Coordinate_matches"
                    ]
                ),

                get_value(

                    r,

                    [
                        "ALT_effect"
                    ],

                    0
                ),

                get_value(

                    r,

                    [
                        "REF_effect"
                    ],

                    0
                ),

                get_value(

                    r,

                    [
                        "MISMATCH"
                    ],

                    0
                ),

                get_value(

                    r,

                    [
                        "Final_Scoring_Variants",
                        "Final_scoring_variants"
                    ]
                ),

                get_value(

                    r,

                    [
                        "Samples"
                    ]
                ),

                get_value(

                    r,

                    [
                        "Missing_PRS"
                    ],

                    0
                ),

                # T2D chromosome QC does not contain
                # chromosome-level PRS statistics.

                pd.NA,
                pd.NA,
                pd.NA,
                pd.NA
            )
        )


        print(

            f"  Chr{chrom}: "
            f"{qc_file.name}"
        )


    return pd.DataFrame(rows)


# ============================================================
# LOAD ALL DISEASES
# ============================================================

all_frames = []


# ============================================================
# ALZHEIMER
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "DISEASE: Alzheimers"
)

print(
    "=" * 70
)


alz = load_alzheimer()


print(

    f"Collected chromosomes: "
    f"{len(alz)}/22"
)


all_frames.append(alz)


# ============================================================
# STANDARD DISEASES
# ============================================================

for disease in [

    "Schizophrenia",

    "Multiple_Sclerosis",

    "MDD",

    "Parkinsons",

    "CAD"

]:

    print(
        "\n" + "=" * 70
    )

    print(
        f"DISEASE: {disease}"
    )

    print(
        "=" * 70
    )


    cfg = DISEASES[
        disease
    ]


    df = load_standard(

        disease,

        cfg["folder"],

        cfg["prefix"]
    )


    print(

        f"Collected chromosomes: "
        f"{len(df)}/22"
    )


    all_frames.append(df)


# ============================================================
# T2D
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "DISEASE: T2D"
)

print(
    "=" * 70
)


t2d = load_t2d()


print(

    f"Collected chromosomes: "
    f"{len(t2d)}/22"
)


all_frames.append(t2d)


# ============================================================
# COMBINE
# ============================================================

qc = pd.concat(

    all_frames,

    ignore_index=True
)


# ============================================================
# NUMERIC CONVERSION
# ============================================================

numeric_columns = [

    "Chromosome",

    "PGS_variants",

    "Coordinate_matches",

    "ALT_effect",

    "REF_effect",

    "MISMATCH",

    "Final_scoring_variants",

    "Samples",

    "Missing_scores",

    "PRS_mean",

    "PRS_SD",

    "PRS_min",

    "PRS_max"
]


for col in numeric_columns:

    qc[col] = pd.to_numeric(

        qc[col],

        errors="coerce"
    )


# ============================================================
# RETENTION
# ============================================================

qc[
    "Retention_percent"
] = (

    qc[
        "Final_scoring_variants"
    ]

    /

    qc[
        "PGS_variants"
    ]

    *

    100
)


qc[
    "Retention_percent"
] = qc[
    "Retention_percent"
].round(4)


# ============================================================
# SORT
# ============================================================

qc = qc.sort_values(

    [
        "Disease",
        "Chromosome"
    ]

).reset_index(
    drop=True
)


# ============================================================
# VALIDATION
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "VALIDATION"
)

print(
    "=" * 70
)


print(
    f"Total QC rows: {len(qc)}"
)


print(
    "Expected QC rows: 154"
)


print(
    f"Unique diseases: "
    f"{qc['Disease'].nunique()}"
)


print(
    f"Unique chromosomes: "
    f"{qc['Chromosome'].nunique()}"
)


# ============================================================
# ROWS PER DISEASE
# ============================================================

rows_per_disease = (

    qc.groupby(
        "Disease"
    )[

        "Chromosome"

    ]

    .nunique()
)


print(
    "\nRows per disease:"
)


print(
    rows_per_disease.to_string()
)


# ============================================================
# DUPLICATES
# ============================================================

duplicates = (

    qc.groupby(

        [
            "Disease",
            "Chromosome"
        ]

    )

    .size()

    .reset_index(
        name="count"
    )
)


duplicates = duplicates[

    duplicates["count"] > 1
]


print(
    "\nDuplicate disease/chromosome combinations:"
)


if len(duplicates) == 0:

    print(
        "NONE"
    )

else:

    print(

        duplicates.to_string(
            index=False
        )
    )


# ============================================================
# EXPECTED COMBINATIONS
# ============================================================

expected = {

    (
        disease,
        chrom
    )

    for disease in DISEASES

    for chrom in range(1, 23)
}


observed = {

    (
        row.Disease,
        int(row.Chromosome)
    )

    for row in qc.itertuples()
}


missing = sorted(

    expected - observed
)


print(
    "\nMissing disease/chromosome combinations:"
)


if not missing:

    print(
        "NONE"
    )

else:

    for item in missing:

        print(
            item
        )


# ============================================================
# VALIDATION FLAGS
# ============================================================

validation_failed = False


if len(qc) != 154:

    validation_failed = True


if qc[
    "Disease"
].nunique() != 7:

    validation_failed = True


if qc[
    "Chromosome"
].nunique() != 22:

    validation_failed = True


if len(duplicates) != 0:

    validation_failed = True


if len(missing) != 0:

    validation_failed = True


# ============================================================
# STOP IF FAILED
# ============================================================

if validation_failed:

    print(
        "\n" + "=" * 70
    )

    print(
        "QC VALIDATION FAILED"
    )

    print(
        "=" * 70
    )

    print(
        "The master QC files "
        "will NOT be written."
    )

    raise SystemExit(1)


# ============================================================
# FINAL COLUMN ORDER
# ============================================================

qc = qc[
    STANDARD_COLUMNS
]


# ============================================================
# SAVE MASTER QC
# ============================================================

master_file = (

    OUT /

    "all_7_diseases_chromosome_QC.tsv"
)


qc.to_csv(

    master_file,

    sep="\t",

    index=False
)


# ============================================================
# DISEASE-LEVEL SUMMARY
# ============================================================

summary_rows = []


for disease, df in qc.groupby(
    "Disease"
):

    total_pgs = (

        df[
            "PGS_variants"
        ]

        .sum()
    )


    total_coordinate = (

        df[
            "Coordinate_matches"
        ]

        .sum()
    )


    total_alt = (

        df[
            "ALT_effect"
        ]

        .sum()
    )


    total_ref = (

        df[
            "REF_effect"
        ]

        .sum()
    )


    total_mismatch = (

        df[
            "MISMATCH"
        ]

        .sum()
    )


    total_final = (

        df[
            "Final_scoring_variants"
        ]

        .sum()
    )


    retention = (

        total_final

        /

        total_pgs

        *

        100
    )


    summary_rows.append({

        "Disease":
            disease,

        "Chromosomes":
            df[
                "Chromosome"
            ].nunique(),

        "Total_PGS_variants":
            total_pgs,

        "Total_coordinate_matches":
            total_coordinate,

        "Total_ALT_effect":
            total_alt,

        "Total_REF_effect":
            total_ref,

        "Total_MISMATCH":
            total_mismatch,

        "Total_final_scoring_variants":
            total_final,

        "Overall_retention_percent":
            round(
                retention,
                4
            ),

        "Minimum_samples":
            df[
                "Samples"
            ].min(),

        "Maximum_samples":
            df[
                "Samples"
            ].max(),

        "Total_missing_scores":
            df[
                "Missing_scores"
            ].sum()
    })


summary = pd.DataFrame(
    summary_rows
)


# ============================================================
# SAVE DISEASE SUMMARY
# ============================================================

summary_file = (

    OUT /

    "all_7_diseases_QC_summary.tsv"
)


summary.to_csv(

    summary_file,

    sep="\t",

    index=False
)


# ============================================================
# CHROMOSOME-LEVEL SUMMARY
# ============================================================

chrom_summary = (

    qc.groupby(
        "Chromosome"
    )

    .agg(

        Diseases_present=(
            "Disease",
            "nunique"
        ),

        Total_PGS_variants=(
            "PGS_variants",
            "sum"
        ),

        Total_coordinate_matches=(
            "Coordinate_matches",
            "sum"
        ),

        Total_final_scoring_variants=(
            "Final_scoring_variants",
            "sum"
        ),

        Total_MISMATCH=(
            "MISMATCH",
            "sum"
        )
    )

    .reset_index()
)


# ============================================================
# SAVE CHROMOSOME SUMMARY
# ============================================================

chrom_file = (

    OUT /

    "all_7_diseases_chromosome_summary.tsv"
)


chrom_summary.to_csv(

    chrom_file,

    sep="\t",

    index=False
)


# ============================================================
# SUCCESS
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "SUCCESS — COMPLETE 7-DISEASE QC"
)

print(
    "=" * 70
)


print(
    f"Master QC rows: {len(qc)}"
)


print(
    "Expected: 154"
)


print(
    f"Unique diseases: "
    f"{qc['Disease'].nunique()}"
)


print(
    f"Unique chromosomes: "
    f"{qc['Chromosome'].nunique()}"
)


print(
    "\nRows per disease:"
)


print(
    rows_per_disease.to_string()
)


print(
    "\nDisease-level QC summary:"
)


print(
    summary.to_string(
        index=False
    )
)


print(
    "\nSaved files:"
)


print(
    master_file
)


print(
    summary_file
)


print(
    chrom_file
)


print(
    "\n" + "=" * 70
)

print(
    "DONE"
)

print(
    "=" * 70
)
