# DAug

# Dataset
Please obtain the MIMIC-CXR dataset from this official source: https://physionet.org/content/mimic-cxr-jpg/2.1.0/
Due to data usage agreements, we are currently not able to release our processed dataset.

# Code
- Use the `diffusion_anomaly` to train the diffusion model and generate the heatmaps. The repo is based on existing work.
- Use `daug` folder to train the DAug model for image classification and retrieval, assuming the heatmaps are pre-generated already.