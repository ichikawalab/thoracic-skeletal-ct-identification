# Thoracic Skeletal CT Identification

Code for linking a chest CT examination to the same person's earlier examination from the shape
of the sternum and the thoracic vertebral bodies, evaluated on the National Lung Screening Trial
(NLST). The method is described in the accompanying paper.

The repository contains source code only: no images, participant-level metadata, features or
results.

## Installation

Requirements: Windows or Linux, [uv](https://docs.astral.sh/uv/) (it installs Python 3.11 and 3.9
itself) and, for segmentation, an NVIDIA GPU with a driver that supports CUDA 12.4.

```bash
git clone https://github.com/ichikawalab/thoracic-skeletal-ct-identification.git
cd thoracic-skeletal-ct-identification
uv sync --locked
uv sync --locked --project environments/env-seg
uv sync --locked --project environments/env-radiomics
uv run --locked --project environments/env-seg python -m common.totalsegmentator_patch
```

The last command repairs a single-component centroid bug in TotalSegmentator 2.18.0; run it again
after reinstalling that package.

## Input

| Input | Source | Required form |
|---|---|---|
| CT images | NLST collection on The Cancer Imaging Archive, downloaded with the NBIA Data Retriever | `<pid>/<study>/<series>/*.dcm`, with the PatientID folders directly below the root |
| Participant table | `prsn` table of the NLST clinical data package | Columns `pid`, `age`, `gender`, `race`, `cigsmok`, `scr_days0` and `scr_days2` in the NLST coding |
| Screening table (optional) | `screen` table of the same package, passed to the selection with `--clinical-screening` | Columns `pid` and `study_yr` (0, 1 or 2), one row per participant and screening year |

Time points are read from the DICOM attribute ClinicalTrialTimePointID (T0, T1 or T2). Baseline
examinations (T0) form the gallery and 2-year examinations (T2) are the queries. The published
analysis used the clinical data package `package-nlst-780.2021-05-28`.

## Pipeline

Run every command from the repository root, and finish each stage before starting the next. The
commands are written for bash; in PowerShell, replace the trailing `\` with a backtick.
`--help` lists the optional arguments of each script.

```bash
DICOM_ROOT=/path/to/NLST          # CT images
RUN=/path/to/run01                # a new directory for each analysis
PRSN_CSV=/path/to/nlst_780_prsn_idc_20210527.csv
```

### 1. Inventory and series selection

Validate every slice of every series, then choose one series per examination.

```bash
uv run --locked python 01_preprocessing/inventory_dicom_series.py \
    --root $DICOM_ROOT \
    --out $RUN/01_inventory/series_inventory.csv

uv run --locked python 01_preprocessing/select_series.py \
    --inventory $RUN/01_inventory/series_inventory.csv \
    --kernel-table 01_preprocessing/kernel_classes.csv \
    --out-dir $RUN/01_selection
```

The selection rules and their numerical tolerances are defined in `common/dicom_series.py` and
`01_preprocessing/select_series.py`.

### 2. NIfTI conversion

Convert each selected series with dcm2niix and verify every slice against the DICOM source.

```bash
uv run --locked python 01_preprocessing/convert_to_nifti.py \
    --root $DICOM_ROOT \
    --selection-dir $RUN/01_selection \
    --out-root $RUN/01_nifti
```

### 3. Segmentation

Segment the sternum and the vertebral bodies with TotalSegmentator.

```bash
uv run --locked --project environments/env-seg python 01_preprocessing/run_segmentation.py \
    --nifti-root $RUN/01_nifti \
    --out-root $RUN/01_segmentation \
    --devices gpu:0
```

The first run downloads the model files into `$RUN/01_segmentation/models`; later runs verify
their hashes offline. The hashes used for the published analysis are listed in
[01_preprocessing/model_weights_nlst.json](01_preprocessing/model_weights_nlst.json).

### 4. Quality control

Decide from the masks alone which examinations and structures are usable.

```bash
uv run --locked python 02_segmentation_qc/qc_segmentation.py \
    --seg-root $RUN/01_segmentation \
    --out $RUN/02_qc/segmentation_qc.csv
```

### 5. Shape features

Extract the 14 PyRadiomics shape features of every usable structure.

```bash
uv run --locked --project environments/env-radiomics python 03_feature_extraction/extract_shape_features.py \
    --seg-root $RUN/01_segmentation \
    --qc $RUN/02_qc/segmentation_qc.csv \
    --params 03_feature_extraction/pyradiomics_shape.yaml \
    --out $RUN/03_features/shape_features.csv
```

### 6. Matching

Rank every query against the whole gallery with the sternum-only, vertebrae-only and integrated
methods.

```bash
uv run --locked python 04_matching/match_examinations.py \
    --features $RUN/03_features/shape_features.csv \
    --qc $RUN/02_qc/segmentation_qc.csv \
    --selection-dir $RUN/01_selection \
    --nifti-root $RUN/01_nifti \
    --seg-root $RUN/01_segmentation \
    --out-dir $RUN/04_matching
```

### 7. Evaluation

Compute the identification and verification tables, the figure and the gallery-size analysis.

```bash
uv run --locked python 05_evaluation/evaluate_matching.py \
    --matching-dir $RUN/04_matching \
    --participants $PRSN_CSV \
    --out-dir $RUN/05_evaluation

uv run --locked python 05_evaluation/gallery_size.py \
    --matching-dir $RUN/04_matching \
    --features $RUN/03_features/shape_features.csv \
    --out-dir $RUN/05_gallery_size
```

## Run records and runtime

Each stage writes its tables into its own directory under `$RUN`, together with a log and a run
record (`<table>.meta.json`: code hash, input hashes, settings, package versions), and refuses to
start until the preceding stage is complete with unchanged inputs. An interrupted inventory,
conversion, segmentation, quality control or feature extraction resumes where it stopped. The
outputs are participant-level; keep them and the run records out of any public repository.

The published analysis (48,044 examinations, one workstation with one GPU) took about 117 h for
segmentation, 100 min for quality control, 39 h for feature extraction and 5 min for matching.
The DICOM collection occupies about 11 TB and the NIfTI volumes about 2.6 TB.

## Tests

```bash
uv run --locked ruff check .
uv run --locked pytest -q
```

The tests use synthetic DICOM series and masks; no patient data are needed.

## Data availability

NLST CT images and the accompanying clinical tables are distributed through
[The Cancer Imaging Archive](https://doi.org/10.7937/TCIA.HMQ8-J677); the full clinical data are
available to approved projects from the National Cancer Institute
([cdas.cancer.gov/nlst](https://cdas.cancer.gov/nlst/)). This repository redistributes neither.

## License and citation

[MIT](LICENSE). Citation metadata are in [`CITATION.cff`](CITATION.cff).
