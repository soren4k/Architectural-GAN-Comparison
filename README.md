# Architectural-GAN-Comparison
An empirical comparison of DC-GAN and WGAN-GP for generating architectural images

This repository contains the source code for the empirical comparison of DC-GAN and WGAN-GP for architectural image generation.

## Structure
- `/src`: Contains the Python scripts for the model architectures.
- `/notebooks`: Contains the Jupyter notebooks for running all experiments.
- `archgan_results.csv`: The final, complete log of all experimental results.

## How to Run
1. Ensure all packages from `requirements.txt` are installed (`pip install -r requirements.txt`).
2. Place the dataset (`exteriors_128.zip`) in the root directory.
3. Open the notebooks in `/notebooks` (e.g., in Google Colab) and run the cells sequentially.