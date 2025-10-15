## Project 2 - Video Classification
Welcome to repository of the second project of the DTU course [Introduction to Deep Learning in Computer Vision](https://kurser.dtu.dk/course/2025-2026/02516?menulanguage=en) about video classification. 

The classification task at hand is to classify videos of training exercises. The training, validation, and test data is derived from this [Kaggle version of UFC-101](https://www.kaggle.com/datasets/matthewjansen/ucf101-action-recognition). 

We are considering the following models:
- Single-Frame CNN
- Late Fusion
- Early Fusion
- 3D CNN

Additional experiments:
- Information leakage between train/val/test splits
- Dual-stream network 

Each experiment is run using the [DTU HPC](https://www.hpc.dtu.dk/). 

To setup the virtual environment used when submitting a batch job, follow the instructions in [HPC Instructions](https://github.com/SebWae/02516-video-classification/blob/main/docs/hpc_instructions.md). Once the virtual environment (`venv_proj2`) has been created, install the required packages specified in `requirements.txt` by running: 
```
pip install -r requirements.txt
```
If more packages/libraries are needed, install these using `pip` and then update `requirements.txt` by running: 
```
pip freeze > requirements.txt
```
Remember to commit and push the changes 😊 

📅 Hand-in: October 26th, 2025