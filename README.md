# STM Foundation Model v3.0 — Ewing Sarcoma Risk-Stratification Framework

A six-stage discrete-event Monte Carlo simulation framework for patient-level risk
stratification in Ewing sarcoma. The model calibrates to published *aggregate* trial data
and resolves those cohort-mean results into individual patient trajectories — the conventional
patient-level machine-learning pathway being structurally unavailable in a rare cancer where no
single institution accumulates enough cases to train and validate individual-level predictors.

This repository accompanies the manuscript *"Multi-Timepoint Risk Stratification in Rare Cancers:
A Computational Framework Validated against Published Ewing Sarcoma Trial Data"* (under review,
*Medical Decision Making*).

## What the model does

- **10,000 virtual patients per cohort** (localized and metastatic), under a fixed per-cohort
  random seed (42), so all outputs are exactly reproducible.
- Six stages carry each patient from diagnosis and genotype assignment through treatment, response,
  surveillance, late effects, and — where relapse occurs — a salvage pathway.
- Coupled modules: genotype-conditional biomarker weighting; ctDNA–MRD integration within genetic
  subgroups; long-term adverse-effect models for congestive heart failure, GFR decline (with an
  explicit nephro-cardiac coupling loop), hypertension, and second malignant neoplasms; and a
  relapsed/refractory salvage module split at a two-year cutoff.
- Calibration anchors: COG AEWS0031, Euro-EWING 99, and rEECur, with genetic-subgroup behavior tied
  to published markers and ctDNA–MRD discrimination recovered from the minimal-residual-disease
  literature.
- **Validated at 3.2% mean absolute error across 23 published endpoints** spanning survival,
  subgroup stratification, and treatment-related toxicity.

## Repository contents

| File | Description | MD5 |
|------|-------------|-----|
| `stm_foundation_model_v3_0_reseed_v2.py` | Simulation model (the code) | `d5ddf4399daf8af243bbef605f7e13bf` |
| `stm_v3_figure_data.json` | Primary figure-data file; reproduced byte-for-byte by running the model | `d3f209b6c2d80c3fde7d187f3c20c27f` |
| `data_brs_and_fig2.json` | Biomarker-risk-score and genetic-subgroup (Figure 2) data; a separate authoritative source the model does not regenerate | `e946b6769b1a9e3450a7626bd4b95f6e` |

The two JSON files are the sole authoritative data sources underlying every value in the manuscript
and its figures.

## Requirements

- Python 3
- NumPy

## Reproducing the data

```bash
python3 stm_foundation_model_v3_0_reseed_v2.py
```

Running the model regenerates `stm_v3_figure_data.json`. To confirm you have the exact file used in
the manuscript, verify its checksum:

```bash
md5sum stm_v3_figure_data.json data_brs_and_fig2.json
# stm_v3_figure_data.json   -> d3f209b6c2d80c3fde7d187f3c20c27f
# data_brs_and_fig2.json    -> e946b6769b1a9e3450a7626bd4b95f6e
```

A match confirms an exact, byte-for-byte reproduction. The internal `source_model` metadata string
in the JSON is part of the calibrated baseline and is intentionally left unchanged.

## Data and reproducibility notes

- Every reported value traces to a real computation or a cited published source. No figure, table, or
  manuscript value relies on synthetic or placeholder data.
- The per-cohort reseed (`seed = 42`) guarantees deterministic, reproducible cohorts.
- Figure-generation scripts (which render the published figures from the JSON files) are maintained
  separately and are not required to reproduce the underlying data above.

## Citation

If you use this model or data, please cite the associated manuscript:

> Kress, J. W. *Multi-Timepoint Risk Stratification in Rare Cancers: A Computational Framework
> Validated against Published Ewing Sarcoma Trial Data.* (Under review, *Medical Decision Making*.)

When citing the code/data directly, reference this repository at the tagged release `v3.0`
and the corresponding commit.

## License

Released under the Apache License 2.0. See `LICENSE`.

## Author

James W. Kress, Ph.D. — The KressWorks Foundation, Northville, MI
ORCID: 0000-0002-2511-6822

## Acknowledgments

With gratitude to Dave and Geri Brown and Henriette Eles. This work is dedicated to the memory of
Patience Canice Kress Hensley (1951–2009) who died from recurrent Ewing Sarcoma.
