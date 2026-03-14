---
name: Plot Loss
description: Parses a training console log file to extract loss values and generates a loss graph as a PNG image.
---
# Training Loss Plotter

This skill provides an automated way to visualize the training loss over time by parsing the terminal output logs from the training process. 

## How to use this skill

1.  **Generate the Plot**:
    -   Run the Python script located in this skill's `scripts` directory, providing it with the path to the training log file and optionally an output path for the PNG image.
    -   Example: `python3 .agent/skills/plot_loss/scripts/plot.py --log-file /tmp/final_gcp_run.log --out-file training_loss.png`

2.  **What it Does**:
    -   The `plot.py` script reads the raw console logs (e.g., from `train.rs` or GCP scripts).
    -   It uses regular expressions to extract `Step` and `Loss` metrics from each log line.
    -   It computes a moving average over the loss and plots both the raw loss and moving average using `matplotlib`.
    -   The final figure is saved to the specified `--out-file` (default is `training_loss.png`).
