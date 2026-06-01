# Computer Vision UI

## Overview

This project provides a Python-based computer vision UI for running video preprocessing and analysis workflows. The UI is intended to support the extraction and visualisation of crowd-related information that can be used for downstream simulation or analysis.

The program includes tools for preprocessing videos, estimating crowd movement, tracking of small group of pedestrians, and plotting tracked positions from CSV data.

## Features

The UI supports the following computer vision workflows:

- Video preprocessing
- Video cropping
- Training of own models
- Optical flow analysis to estimate the movement speed of large crowds
- Object/person tracking for smaller crowds
- Plotting tracked pedestrian points using CSV data

## Requirements

- Python 3.13.5
- Python virtual environment
- Required packages listed in `requirements.txt`

## Setup Instructions

1. Open a terminal in the CV UI project folder.

2. Create a virtual environment:

```bash
python -m venv .venv
```

3. Activate the virtual environment.

For Windows PowerShell:

```bash
.venv\Scripts\Activate.ps1
```

For Windows Command Prompt:

```bash
.venv\Scripts\activate.bat
```

4. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Run Instructions

After setting up and activating the virtual environment, launch the UI by running:

```bash
python ui.py
```

## Main Program File

The main entry point for the UI is:

```text
ui.py
```

## Usage

1. Launch the UI using `python ui.py`.
2. Select the desired computer vision function.
3. Provide the required input video or CSV file.
4. Run the selected process.
5. Review the generated outputs or visualisations through the preview and in the selected folders for the data to be saved in.

## Notes

For tracking, without a selection of model it will run the base yolo11n.pt model. For the test video provided I have trained a seperate model called best.pt. Different cameras will have different angles and FOV so a dedicated model should be trained for each cameras in the future to drastically improve accuracy.

Additionally there are functions like the counting of higher density crowds called predict_single.py and predict_video.py that has yet to be added into the UI (due to time constraints) but are functioning scripts by themselves.
