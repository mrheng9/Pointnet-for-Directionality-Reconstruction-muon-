# PointNet++ for directionality reconstruction — Tutorial

## Overview
The Jiangmen Underground Neutrino Observatory **(JUNO)** is a large liquid-scintillator detector designed to determine the neutrino mass ordering and precisely measure oscillation parameters.   
![](https://github.com/user-attachments/assets/e33b6881-fad5-4585-a326-d289810b430b)
Beyond reactor neutrinos, JUNO also observes atmospheric neutrinos whose charged-current interactions produce muons. This project focuses on reconstructing the muon direction from atmospheric neutrino interactions using a PointNet++ regression model on PMT-derived point-cloud features.
 
## Repository Structure
- `train.py` — main training script 
  - `pointnet_regression_ssg.py` — PointNet MSG-based regression model
  - `pointnet_regression_utils.py` — set abstraction layers and MSG ops
- `data_utils/`
  - `PMTLoader.py` — `CustomDataset`, and data stacking helpers
- `README.md` — brief usage notes

## Environment (conda, Windows)

Dependencies:
- python=3.8
- numpy
- pytorch
- matplotlib
- tqdm
- scikit-learn
- pandas
- git

Setup (PowerShell):
```powershell
# Create and activate env
conda create -n pointnet2p python=3.8 -y
conda activate pointnet2p

# Install packages
conda install numpy scikit-learn matplotlib tqdm pandas git -y
conda install pytorch -c pytorch -y
```
## Data Preparation
Implement or point the loader to your dataset through `data_utils/PMTLoader.py`.

Expected conventions:
- Inputs: stacked point features shaped `[N, P, C]`
  - N = samples, P = points per sample, C = input channels (e.g., coordinates + features)
- Targets: regression vectors shaped `[N, D]` (commonly 3D; labels are normalized by max vector norm in training)

Feature origin (PMT waveforms):
- All feature channels C are extracted from PMT charge/time waveforms measured or reconstructed from the detector.
- Typical examples:
  - fht: first-hit time of the PMT pulse
  - slope: rising-edge slope proxy
  - peak / peaktime: pulse peak amplitude and its time
  - timemax: time of maximum sample
  - nperatio5: charge ratio within a short window (e.g., 5 ns) to total charge
  - npe: number of photoelectrons (charge proxy)

Data sources (argument: --data_source):
- det — Features extracted from detector-level simulation. 
- elec — Features after electronics simulation.
- cnn — Features reconstructed by a CNN from waveforms. 
- rawnet — Features reconstructed by a RawNet model. 

## Model Overview
File: `models/pointnet_regression_ssg.py`
- Extracts hierarchical point features using:
  - `PointNetSetAbstractionMsg` (multi-scale grouping)
  - `PointNetSetAbstraction` (global stage)
- Aggregates multi-scale features and regresses to a continuous target.
- Configurable input channels (`in_channel`) to match your data (C above).

Core building blocks in `pointnet_regression_utils.py`:
- `PointNetSetAbstraction`
- `PointNetSetAbstractionMsg`

## Training
[Different ways to run the model](https://github.com/mrheng9/mrheng9/blob/main/tutorial/tutorial.md#execution-methods)  
Choose the one you prefer. Here use `nohup` as a demonstration  
**(note that the --data_source parameter is required)**
```
$ nohup python train.py --data_source cnn --model > cnn.log 2>&1 &
```
Check running display information in nohup.out  

`nohup` is a great way to keep your logs tidy. It ensures that the output from each training process gets its own file, which is perfect for when you're running different jobs on different GPUs at the same time and want to keep things separate.  
Learn how to manage GPU ([GPU Management](https://github.com/mrheng9/mrheng9/blob/main/tutorial/tutorial.md#gpu-management))

- learning_curve.png — Train/Test loss vs. epoch (log y-axis).
- Test Performance.png — Scatter of predicted vs. true θ with y=x reference.
- error_distribution.png — Histogram of (pred − true) angle errors in degrees, with mean/std annotations.
- angel_distribution.png — Opening angle α PDF between true and predicted directions, with 68% quantile marker.