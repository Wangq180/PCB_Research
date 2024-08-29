# PCB_Research
![image](https://github.com/user-attachments/assets/8d17573a-5b64-4a22-ad62-d699020f917e)
Papers are being published

PCB_Research
Printed Circuit Boards Image Reconstruction from a Data-Centric Technology: A High-Quality Dataset and Comprehensive Evaluation

Overview
This repository contains the code and datasets associated with our research on the reconstruction of Printed Circuit Board (PCB) images using a data-centric approach. Our work focuses on developing robust and generalized methods for PCB image reconstruction by leveraging specialized datasets and evaluating various algorithms using a novel metric tailored for PCB edge information recovery.

Table of Contents
Dataset
Code
Calculation
Graph
Models
Usage
Citation
License
Dataset
Description
The dataset folder contains all the images used in our study, including the original PCB images and those with simulated defects such as masking and blurring. These images were generated to closely mimic real-world PCB defects, allowing for precise and reliable evaluation of image reconstruction algorithms.

Contents
Masked Images: PCB images with parts of the board masked to simulate damage or obstruction.
Blurred Images: PCB images that have been intentionally blurred to replicate the effects of contamination or lens issues.
Tampered Images: PCB images with intentional modifications to simulate tampering or manufacturing defects.
Creation Process
The details of the dataset creation process are thoroughly explained in Section 3 of our paper, where we outline the methodologies used for generating these simulated defects. The dataset will be made fully available upon publication of our paper, including detailed instructions on how to access and cite the dataset.

Code
Description
The code folder contains all the scripts used in our experiments, which are divided into three main categories: Calculation, Graph, and Models. These scripts are essential for replicating our results and performing further analysis on the dataset.

Calculation
The Calculation folder includes scripts for calculating various performance metrics, such as PSNR, MSE, and our proposed Detail Reconstruction Quality (DRQ) metric. These scripts allow researchers to rank the difficulty of the dataset subsets based on these metrics, as discussed in Section 3 of our paper.

Graph
The Graph folder contains scripts for generating visual representations of the performance metrics across different models and datasets. These visualizations are crucial for understanding the variability and effectiveness of different reconstruction techniques.

Models
The Models folder includes implementations of the models used in our study, with links to the original repositories for further reference. These models have been adapted for PCB image reconstruction and can be evaluated using the scripts provided in the Graph folder.

DnCNN: GitHub Repository
U-Net: GitHub Repository
EDSR: GitHub Repository
DocRes: GitHub Repository
SimMIM: GitHub Repository
MAE: GitHub Repository
These models have been fine-tuned and tested on our dataset, and their performance can be replicated and further analyzed using the provided code.

Usage
Dataset Preparation:

Download the dataset from GitHub or access it via Zenodo.
Follow the instructions in the dataset folder to set up the images for training, validation, and testing.
Running Calculations:

Use the scripts in the Calculation folder to evaluate different models on the dataset.
Metrics such as PSNR, MSE, and DRQ will be calculated and stored for analysis.
Generating Graphs:

The scripts in the Graph folder can be used to visualize the performance of various models.
Adjust the parameters as needed to focus on specific subsets or metrics.
Model Training and Evaluation:

The models in the Models folder are ready for training and evaluation on the dataset.
Follow the instructions in each model's README to set up the environment and start training.
Citation
If you use this code or dataset in your research, please cite our paper as follows:

