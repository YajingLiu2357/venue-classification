# Venue Classification Project 
Image classification is essential in fields such as image search. This project aims to categorize venue images into five categories using supervised decision trees, semi-supervised decision trees, and supervised convolutional neural networks (CNNs). The classified venues are: bar, beach, restaurant, subway and bookstore.
- Concordia University
- COMP 6721 Applied Artificial Intelligence, Summer 2024
- Group Name: CTYCNN 

## Group Member
Chaima jaziri (Chaima-Ja)
Taranjeet Kaur (ktaran-jeet)
Yajing Liu (YajingLiu2357)

## Requirements
Code is written in .ipynb files which can be directly run on locally on conda environment or remotely on Jupyter Lab or Google Colab.

dependencies(in torch.yml): name: torch
    - python=3.11
    - pip>=19.0
    - pytorch 
    - torchvision 
    - jupyter
    - scikit-learn
    - scipy
    - pandas
    - pandas-datareader
    - matplotlib
    - pillow
    - tqdm
    - requests
    - h5py
    - pyyaml
    - flask
    - boto3
    - ipykernel
    - pip:
        - bayesian-optimization
        - gym
        - kaggle
        

## How to train the model
Provide the path to the dataset variable to train CNN in CNN.ipynb, and envAAI-DT-SemiSupervised.ipynb, envAAI-DT-finalsupervised.ipynb for training decision tree model.
The hyperparameters for training respective models can also be changed.
To modify the dataset, use envAAI-Data-Preprocessing.ipynb. It has all the preprocessing functionalities- Resizing, duplicate elimination, Color conversion, data augmentation etc.

## How to run the pretrained model
Run the python file Application.ipynb and enter the image path in image_path. This will run the pretrained model on best performance CNN and decision tree classifier.

## Dataset
https://images.cv/

https://web.mit.edu/torralba/www/indoor.html


## GitHub
https://github.com/YajingLiu2357/venue-classification
