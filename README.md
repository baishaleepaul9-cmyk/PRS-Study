# PRS-Study
# Multi-Disease Polygenic Risk Score Analysis Across Populations

## Overview

This repository contains the computational workflow and selected final analysis outputs for a multi-disease polygenic risk score (PRS) study investigating genome-wide PRS distributions across populations represented in the 1000 Genomes Project Phase III.

The study evaluates seven diseases:

1. Alzheimer's disease (AD)
2. Schizophrenia (SCZ)
3. Multiple sclerosis (MS)
4. Major depressive disorder (MDD)
5. Parkinson's disease (PD)
6. Coronary artery disease (CAD)
7. Type 2 diabetes (T2D)

Published disease-specific polygenic score weights were applied to genotype data from 2,504 individuals in the 1000 Genomes Project Phase III. Individuals were classified into five superpopulations:

- AFR — African
- AMR — Ad Mixed American
- EAS — East Asian
- EUR — European
- SAS — South Asian

The study integrates disease-specific PRS calculation, variant harmonization, quality control, population-level analysis, within-disease standardization, and cross-disease population analysis.

> **Important:** This study evaluates PRS distributions in a reference population. It does not estimate clinical disease risk for individual participants. The low-, intermediate-, and high-relative-PRS groups are empirical groupings within each disease and should not be interpreted as clinically validated risk categories.

---

## Study Objectives

The computational framework was designed to:

1. Harmonize published disease-specific polygenic score data with 1000 Genomes genotype data.
2. Perform coordinate- and allele-based variant matching and quality control.
3. Calculate genome-wide PRSs across autosomes 1–22.
4. Consolidate disease-specific genome-wide PRS results.
5. Evaluate PRS distributions across five 1000 Genomes superpopulations.
6. Generate disease-specific relative PRS groups.
7. Standardize PRSs within disease for cross-disease comparison.
8. Quantify population-level differences using non-parametric statistical analysis and effect-size estimates.
9. Evaluate disease × superpopulation interaction patterns.
10. Perform sensitivity analysis of the standardized cross-disease comparisons.

---

## Data

### 1000 Genomes Project

Genotype data were obtained from the 1000 Genomes Project Phase III reference dataset.

The study included:

- 2,504 individuals
- Autosomes 1–22
- Five superpopulations:
  - AFR
  - AMR
  - EAS
  - EUR
  - SAS

Population and superpopulation labels were assigned using the 1000 Genomes population panel and sample/IID matching.

The 1000 Genomes genotype data are not redistributed in this repository.

### Published Polygenic Scores

Disease-specific published polygenic score datasets/weights were used for the seven diseases.

The present study applies published disease-specific score weights to genotype data. It does not develop a new PRS weighting model or derive new GWAS effect estimates.

The original PRS datasets are not redistributed in this repository.

---

# Repository Organization

The repository uses separate branches for disease-specific analyses and a `main` branch for the integrated multi-disease analysis.

## Main Branch

The `main` branch contains the integrated seven-disease analysis, including:

- Disease consolidation
- Cross-disease quality control
- Population consolidation
- Within-disease standardization
- Cross-disease population statistics
- Population effect-size analysis
- Disease × superpopulation interaction analysis
- Standardization sensitivity analysis
- Publication and supplementary figure generation
- Combined final results

## Disease-Specific Branches

The following branches contain disease-specific processing and population analyses:

| Branch | Disease |
|---|---|
| `alzheimer` | Alzheimer's disease |
| `Schizophrenia` | Schizophrenia |
| `MS` | Multiple sclerosis |
| `Mdd` | Major depressive disorder |
| `Parkinson` | Parkinson's disease |
| `CAD` | Coronary artery disease |
| `T2D` | Type 2 diabetes |

Each disease branch contains the corresponding disease-specific analysis scripts and selected final results.

---

# Analytical Workflow

## 1. Variant Harmonization

Published disease-specific score variants were matched to 1000 Genomes variants using genomic coordinates and allele information.

The harmonization workflow included:

- Coordinate matching
- Reference/alternate allele comparison
- Effect-allele identification
- Removal of incompatible variants
- Variant quality-control filtering
- Generation of allele-resolved variant identifiers

The exact harmonization procedure may vary slightly between disease-specific score datasets according to the information provided by the original score source.

---

## 2. Genome-Wide PRS Calculation

For each disease, harmonized variants were processed chromosome-wise across autosomes 1–22.

Published effect weights were retained for PRS calculation.

Chromosome-level scores were subsequently aggregated to obtain genome-wide PRS values for each individual.

PLINK 2 was used for genotype processing and PRS scoring where applicable.

---

## 3. Disease-Specific Population Analysis

Following genome-wide PRS calculation, individual PRS values were merged with 1000 Genomes population labels using sample/IID matching.

Disease-specific population analyses included:

- Population-level descriptive statistics
- PRS distribution analysis
- Relative PRS grouping
- Kruskal–Wallis testing
- Post-hoc pairwise comparisons
- Population-level visualization

The corresponding scripts and selected outputs are available in the respective disease branches.

---

## 4. Relative PRS Groups

Disease-specific PRS distributions were divided into empirical relative groups using the distribution of scores within each disease.

The grouping framework consisted of:

- Low-relative-PRS: lower 20%
- Intermediate-relative-PRS: middle 60%
- High-relative-PRS: upper 20%

These are relative groupings within the analyzed reference population and disease-specific score distribution.

They are not clinical risk categories.

---

## 5. Population-Level Statistical Analysis

For each disease and superpopulation, descriptive statistics were calculated, including:

- Sample size
- Mean
- Standard deviation
- Median
- Quartiles
- Minimum and maximum
- Standard error
- 95% confidence interval

Population differences were evaluated using the Kruskal–Wallis test.

Post-hoc pairwise comparisons were performed using Dunn-style rank comparisons with Holm correction where applicable.

---

## 6. Cross-Disease Standardization

Because PRSs derived for different diseases have different numerical scales, scores were standardized within disease using Z-score transformation:

\[
Z = \frac{PRS-\mu}{\sigma}
\]

where:

- `PRS` = individual disease-specific PRS
- `μ` = mean PRS for that disease
- `σ` = standard deviation of the PRS distribution for that disease

This transformation allows comparison of relative population positions across diseases without interpreting raw PRS scales as directly equivalent.

---

## 7. Population Effect-Size Analysis

Population differences were quantified using the Kruskal–Wallis test together with epsilon-squared (ε²) effect-size estimates.

This provides an estimate of the magnitude of population-associated differences in PRS distributions in addition to statistical significance.

---

## 8. Disease × Superpopulation Interaction

A disease × superpopulation interaction analysis was performed using the standardized PRS values for all seven diseases and five superpopulations.

This analysis evaluates whether population-associated PRS patterns vary across diseases.

The interaction analysis was additionally evaluated using permutation testing with 10,000 permutations.

---

## 9. Standardization Sensitivity Analysis

A rank-based inverse-normal transformation (INT) sensitivity analysis was performed to evaluate whether population-level rank-based findings were sensitive to the transformation used for cross-disease standardization.

Because the Kruskal–Wallis test is rank-based, this analysis assesses whether population comparisons remain invariant under an alternative monotonic rank-based transformation.

---

# Software

The analysis was implemented primarily in Python.

Major software and libraries include:

- Python
- pandas
- NumPy
- SciPy
- scikit-learn
- statsmodels
- Matplotlib
- Seaborn
- PLINK 2

PLINK 2 is required for genotype processing and PRS scoring steps that use PLINK-based workflows.

---

# Reproducibility

The repository contains the analysis scripts used for disease-specific and integrated analyses together with selected final outputs.

Large genotype files and original published PRS datasets are not redistributed.

To reproduce the analysis:

1. Obtain the required 1000 Genomes Project Phase III genotype data from the original source.
2. Obtain the corresponding published disease-specific PRS datasets from their original sources.
3. Install the required Python dependencies.
4. Install PLINK 2.
5. Configure the input and output paths used by the scripts.
6. Run the appropriate disease-specific branch workflow.
7. Consolidate the disease-specific PRS results using the scripts in the `main` branch.
8. Perform the integrated population, standardization, effect-size, interaction, and sensitivity analyses.

The original analysis was performed in a local Windows environment. File paths may therefore need to be adapted when reproducing the workflow on another system.

---

# Results

Selected final outputs are provided within the disease-specific branches and the `main` branch.

Disease-specific outputs include, where applicable:

- Genome-wide PRS summaries
- Final QC summaries
- Population-level PRS statistics
- Kruskal–Wallis results
- Post-hoc pairwise comparisons
- Relative PRS group summaries
- Publication figures

The integrated analysis contains:

- Seven-disease PRS consolidation
- Cross-disease standardized PRS statistics
- Population effect-size estimates
- Disease × superpopulation interaction analysis
- Permutation-based interaction assessment
- Standardization sensitivity analysis
- Combined publication and supplementary figure outputs

Large intermediate genotype files, chromosome-level PLINK files, raw PRS source files, and troubleshooting/diagnostic outputs are excluded.

---

# Interpretation

The analyses in this repository focus on the distribution of polygenic scores across populations.

Differences in PRS distributions should not be interpreted as direct estimates of clinical disease risk between populations.

Polygenic score distributions may be influenced by factors including:

- Allele-frequency differences
- Linkage disequilibrium patterns
- Population composition of the underlying GWAS
- Genetic architecture
- Variant selection
- Effect-size estimation
- Properties of the published score

Standardized PRS values represent relative positions within disease-specific distributions and should not be interpreted as equivalent absolute biological or clinical risk across diseases.

---

# Data Sources

### 1000 Genomes Project

1000 Genomes Project Consortium.  
*A global reference for human genetic variation.*  
Nature. 2015;526:68–74.  
DOI: 10.1038/nature15393

Disease-specific PRS datasets are obtained from their respective original publications/data sources and are not redistributed in this repository.


