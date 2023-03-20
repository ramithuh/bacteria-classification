# Detection of infection and antimicrobial resistance of bacteria

This repository contains the official python implementation of the paper - https://doi.org/10.1101/2022.07.07.499154

Test set download link -> [Google Drive](https://drive.google.com/drive/folders/1bMreABC4tNppzYG56SdTbkFrlVfZlS9U?usp=share_link)


## Setting up the environment 
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

## Directory Structure
- Modules - supporting python library
- Notebooks - different experiments

| Classification Task  | Training Notebook    | Saved Model | N bacteria grouped evaluation |
| ----------- |------ |----------- |---------| 
| Gram Stain Classification | [Notebook](https://github.com/ramithuh/bacteria-classification/blob/main/notebooks/Gram_strain/train-gram_strain-resnet.ipynb) |[link](https://github.com/ramithuh/bacteria-classification/blob/a6b1ee8e9e449faf9155e4f588c41c5fad64afe5/results/GramStrain%20-%20Resnet%20181645133407.8305595/latest_model_epoch-7.pth)       | [Notebook](https://github.com/ramithuh/bacteria-classification/blob/main/notebooks/N_group_evaluation/Gram_Strain_with_N.ipynb)        |
| Antibiotic Resistance Prediction   | [Notebook](https://github.com/ramithuh/bacteria-classification/blob/main/notebooks/antibiotic_resistance_prediction/train-a_resistance-resnet.ipynb)  |   [link](https://github.com/ramithuh/bacteria-classification/blob/a6b1ee8e9e449faf9155e4f588c41c5fad64afe5/results/ARP%20-%20Resnet%20181645133220.7868116/latest_model_epoch-8.pth)      | [Notebook](https://github.com/ramithuh/bacteria-classification/blob/main/notebooks/N_group_evaluation/ARP_with_N.ipynb) |
|Species Level Classification|[Notebook](https://github.com/ramithuh/bacteria-classification/blob/main/notebooks/species_level_classification/train-5-species-resnet.ipynb)| [link](https://github.com/ramithuh/bacteria-classification/blob/main/results/Species%20Classification%20-%20Resnet%20181645133998.8009222/latest_model_epoch-7.pth) | [Notebook](https://github.com/ramithuh/bacteria-classification/blob/main/notebooks/N_group_evaluation/species_level_with_N.ipynb)|
|Strain Level Classification| [Notebook](https://github.com/ramithuh/bacteria-classification/blob/main/notebooks/train-21-strains-resnet.ipynb) | [link](https://github.com/ramithuh/bacteria-classification/blob/main/results/Strain%20Classification%20-%20Resnet%20181645133846.9762254/latest_model_epoch-7.pth) | [Notebook](https://github.com/ramithuh/bacteria-classification/blob/main/notebooks/N_group_evaluation/strain_level_with_N.ipynb)|



## Running predicitons on the blind test

After downloading pretrained models and test dataset, you can edit the `path` in the python scripts below and the [dataloader.py](https://github.com/ramithuh/bacteria-classification/blob/75ea65325e4a9ba91f36879e217230d4f73a260c/modules/dataloaders.py#L59) using `data_dir` variable.

Specify the concentration level (N) to be used for prediction in the line below, 
https://github.com/ramithuh/bacteria-classification/blob/75ea65325e4a9ba91f36879e217230d4f73a260c/notebooks/N_group_evaluation/verify_strain.py#L68

Eg. running for N = 3, replace the line with,
`for N in [3]:`

- Antibiotic Resistance - [verify_arp.py](https://github.com/ramithuh/bacteria-classification/blob/main/notebooks/N_group_evaluation/verify_arp.py)
- Gram Stain Classification - [verify_gram.py](https://github.com/ramithuh/bacteria-classification/blob/main/notebooks/N_group_evaluation/verify_gram.py)
- Species Classification - [verify_species.py](https://github.com/ramithuh/bacteria-classification/blob/main/notebooks/N_group_evaluation/verify_species.py)
- Strain Classification - [verify_strain.py](https://github.com/ramithuh/bacteria-classification/blob/main/notebooks/N_group_evaluation/verify_strain.py)




