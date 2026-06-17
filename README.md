# WALINET

WALINET (Water And Lipid Identification NETwork) is a convolutional neural network for removing nuisance water and lipid signals from proton magnetic resonance spectroscopic imaging (¹H-MRSI) data.

The original WALINET method was published in [Weiser et al., 2024](https://doi.org/10.1002/mrm.30402), and the original source code is available at [weiserjpaul/WALINET](https://github.com/weiserjpaul/WALINET).

This repository is based on the original WALINET implementation by Paul Weiser and collaborators. The codebase has been substantially refactored to improve reproducibility, configuration management, training-data generation, and usability across different datasets and acquisition settings.

> **Note**  
> Due to data-sharing restrictions, the MRSI datasets used for training and evaluation cannot be distributed as part of this repository. Users must provide their own MRSI datasets in the expected format to generate training data, train models, and perform inference.

This repository provides workflows for:

- generating WALINET training data from MRSI datasets,
- training WALINET models,
- applying trained models to unseen data,
- reproducing experiments through configuration-driven pipelines.

The code is intended primarily as a research software framework for developing, training, and evaluating WALINET-based nuisance signal removal methods in ¹H-MRSI.


## Table of Contents
- [1. Overview](#1-overview)
- [2. Installation](#2-installation)
- [3. Usage](#3-usage)
    - [3.1 Data Requirements](#31-data-requirements)
    - [3.2 Simulations](#32-simulations)
    - [3.3 Training the Network](#33-training-the-network)
    - [3.4 Inference on Unseen Data](#34-inference-on-unseen-data)

- [4. Novelties](#4-novelties)
- [5. Results](#5-results)
- [6. Data used for training so far](#6-data-used-so-far)

## 1. Installation

The recommended way to use this repository is through the provided Docker container.

### Using Docker

**Prerequisite:** Docker must be installed on your system.

Build the container:

```bash
bash docker/build_docker.sh
```
Launch the container:

```
bash docker/launch_docker.sh
```
This starts a container with the required Python environment and dependencies.

## Repository Structure

## Repository Structure

```text
configs/          YAML configuration files for training-data generation and model training
data/             User-provided MRSI datasets and generated training data
docker/           Docker build and launch scripts
legacy_code/      Older scripts kept for reference
logs/             Runtime logs from long-running scripts
MetabModes/       Metabolite basis/mode files used for spectral simulations
models/           Trained models, checkpoints, configs, and training logs
notebooks/        Analysis and inference notebooks
scripts/          Command-line entry points
src/              Source code for the WALINET package
tests/            Automated tests

B0_correction/    Auxiliary B0-correction code; not required for the core WALINET workflow
```

## 3. Usage

The main workflows are controlled through YAML configuration files in `configs/`.

### 1. Prepare data

Input data must follow this structure:

```text
data/
└── Subject01/
    ├── OriginalData/
    │   └── data.npy
    └── masks/
        ├── brain_mask.npy
        └── lipid_mask.npy
```

```text
data.npy        Complex-valued MRSI data      (x, y, z, t)
brain_mask.npy  Binary brain mask             (x, y, z)
lipid_mask.npy  Binary lipid/scalp mask       (x, y, z)
```

### 2. Generate Training Data

Edit one of the training-data configuration files, for example:

```text
configs/generate_training_data_3T.yaml
configs/generate_training_data_7T.yaml
```

Then run:

```bash
python3 scripts/generate_training_data.py \
  --config configs/generate_training_data_3T.yaml
```

Training data will be generated in:

```text
data/<subject>/TrainData/
```

including files such as:

```text
IsolatedWater_<version>.npy
TrainData_<version>.h5
```
### 3. Train a Model

Edit:

```text
configs/train.yaml
```

Then run:

```bash
python3 scripts/train.py \
  --config configs/train.yaml
```

Trained models are stored in:

```text
models/<model_name>/
```

including files such as:

```text
model_last.pt
model_best.pt
loss.txt
configs/train.yaml
```

### 4. Run Inference

Inference can be performed using the notebook:

```text
notebooks/EvaluateModelB4Fitting.ipynb
```

The notebook loads a trained WALINET model, applies it to an MRSI dataset, and provides visualizations of spectra before and after nuisance signal removal.

A typical inference workflow consists of:

```text
1. Load MRSI data.
2. Load brain and lipid masks.
3. Compute the lipid projection operator.
4. Load a trained WALINET model.
5. Apply the model to remove water and lipid nuisance signals.
6. Visualize spectra before and after correction.
```

Inference requires:

```text
- a trained WALINET model
- complex-valued MRSI data
- brain mask
- lipid/scalp mask
```
## Additional Notebooks

The repository may contain additional notebooks for downstream analysis, such as metabolite quantification or metabolite map visualization.

These notebooks are not part of the core WALINET pipeline and may require additional external software or site-specific processing tools. They are kept mainly for reference and exploratory analysis.


## 6. Data Used So Far

# Data from Vienna (current data)

The following datasets were provided by Bernhard Strasser, acquired on the 7 T scanner at the Medical University of Vienna:

/ceph/mri.meduniwien.ac.at/departments/radiology/mrsbrain/home/bstrasser/Projects/Project9_ImplementRecoInICE/Step2_ISMRMAbstractOnPipeline/LargeData_d3hj/Results/Vol[VolNo]/Vol[VolNo]_64x64x35_DICOM_Ice73/CombinedCSI.mat


- Available volumes: `Vol5`, `Vol6`, `Vol7`, `Vol8`

In addition, Zeinab provided one more subject (measured under the same conditions), corresponding to:

/ceph/mri.meduniwien.ac.at/departments/radiology/mrsbrain/home/zeftekhari/Bernhard_pipeline/Results/Vol9/Vol9_DICOM_NoL2/CombinedCSI.mat

# Data from Vienna that Paul used (~2019)

This data was used by Paul and has different FID length (960 instead of 840) then the rest of the data, and the bandwidth was slightly higher. It should not be used for training anymore.

Paul Train data:
/ceph/mri.meduniwien.ac.at/departments/radiology/mrsbrain/lab/Process_Results/3DMRSIMAP_Volunteers/PaulTrainData

# Data from Brisbane

Vol1:
/ceph/mri.meduniwien.ac.at/departments/radiology/mrsbrain/home/zeftekhari/Brisbane_Data/Results/MRSI-TEST-2/DICOM_NoL2/

Vol2: # Zeinab said that the quality of this one i snot so high. I double checked, this is good enough to use for WALINET training!
 /ceph/mri.meduniwien.ac.at/departments/radiology/mrsbrain/home/zeftekhari/Brisbane_Data/Results/MRSI-TEST-1/DICOM_NoL2/CombinedCSI.mat

Vol3:
/ceph/mri.meduniwien.ac.at/departments/radiology/mrsbrain/home/zeftekhari/Brisbane_Data/Results/MRSI-TEST-3/run-2-DICOM_NoL2/

Vol4:
/ceph/mri.meduniwien.ac.at/departments/radiology/mrsbrain/home/zeftekhari/Brisbane_Data/Results/MRSI-TEST-4/run-1-DICOM_NoL2/

Vol5:
/ceph/mri.meduniwien.ac.at/departments/radiology/mrsbrain/home/zeftekhari/Brisbane_Data/Results/MRSI-TEST-5/DICOM_NoL2/

Vol6:
 /ceph/mri.meduniwien.ac.at/departments/radiology/mrsbrain/home/bstrasser/Projects/Project9_ImplementRecoInICE/Step5_MultiCenterStudy/LargeData_d3hj/Results/Brisbane/MRSI-TEST-7/DICOM_NoL2
 
# Date Versions:

1.0 Water peaks up to 110 times higher than metabos
1.1 Water peaks up to 300 times higher than metabos
2.0 is simulations for B0 corrected data with the same parameters as 1.0 for uncorrected data


