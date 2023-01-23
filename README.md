# Detection of infection and antimicrobial resistance of bacteria

This repository contains the official python implementation of the paper - https://doi.org/10.1101/2022.07.07.499154


# Setting up the environment 
```bash
## create new environment
conda create -n qpm_env python=3.6
source activate qpm_env

## Adding new environment to JupyterLab
conda install -c anaconda ipykernel -y
python -m ipykernel install --user --name=qpm_env

conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c nvidia
conda install -c conda-forge matplotlib
conda install -c conda-forge wandb

#install remaining packages through pip
pip install -r requirements.txt
```

# Directory Structure
- Modules - supporting python library
- Notebooks - different experiments
