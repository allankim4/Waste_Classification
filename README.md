# Waste_Classification

## Author:[Allan Kim Gayahan](https://github.com/allankim4)

## Executive summary:

The goal of this project is to perform Image Classification on two general types of waste, organic and recyclable.
(Images were retreived from kaggle) 

## Contents

- [Introduction](#Introduction)
    - [Problem statement](#Problem-statement)
    - [Dataset](#Dataset)
    - [Hardware](#Hardware)
- [Analysis](#Analysis)
    - [Data Pre-processing](#Data-Pre-processing)
    - [Splitting the Data](#Splitting-the-Data)
    - [CNN Modeling](#CNN-Modeling)
    - [Transfer Learning](#Transfer-Learning)
- [Summary of Files](#Files-summary)


## Introduction

### Dataset
The dataset is included in this repo located at Images/DATASET

These images are classified into two folders for training and test set which also has a subfolder of Organic(O) and Recyclable(R) folders.

### Hardware
Perhaps the biggest challenge for these simple CNN models is the getting the right hardware. Though I think running the notebook/script on google colab may lessen the burden of having a good computer spec specific for deep learning.

I used a dedicated Unix PC for Deep Learning with the following specs and Software:
PC Specification:
CPU: Ryzen 2700X: 8-core, 16-threads
GPU: GTX 1080: 8GB VRAM
RAM: 64GB RAM
SSD: 1TB M.2 SSD

Software Required:
Tensorflow-GPU
(NVIDIA hardware only):
  Cuda-10.0 - please ensure it is indeed CUDA 10.0 and not 10.1. Otherwise, tensorflow-gpu wont work and will use CPU for     training the model
  CuDNN7
  
  
If you are planning to run this notebook on a computer with no GPU, please be advised that it took 3 minutes with 5k+ images to finish a 16-parallel processes per epoch. Also, it would require atleast 32GB ram to compile and preprocess all the images.


## Analysis

### Data Pre-processing

The pre-processing of images were simplified. The Images were broken down to two sets of images, Train and Test set. These respective batches of images were rescaled to encompass the 225 RGB values for each pixel for each color is represented by a number. Each set are read as Training Images and Test Images, respectively


### Splitting the Data

The Images were compiled and then split using sklearn train_test_split module with the inclusion of creating a split of validation data sets.

### CNN Modeling

I created two models, 2-layer and 3-layer CNN. The 2-layer CNN was very acceptable but still a bit confused so I decided to create a deeper one. The result for 3-layer was acceptable as well and a bit better than the 2-layer in some aspects but not totally.

### Transfer Learning

Transfer Learning was utilized to use pretrained models from Keras.Application.

I used the models I was familiar with and which I think is most effective, InceptionV3 and Xception.


## Summary of Files

The technical notebook is named [waste_classification](waste_classification.ipynb) which is recommended for viewing purposes only.

The play-around editable notebook [Image_Classification](image_classification.ipynb) is where users can input images to check how the models will classify it. 

The .py file cnn_fxn contains function for the editable notebook. The .h5 files are the save pre-trained models defined respectively in the .py to allevaite confuion to which model is which.



