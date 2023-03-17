# NOTE
A simple test for the paper [*A Graph Neural Network for superpixel image classification*](https://iopscience.iop.org/article/10.1088/1742-6596/1871/1/012071/pdf). I modified the code for my own study purpose.

Deleted some files.

Add the environment files for a better reproducibility.

The origin code can be found in backup folder.


# gnn-mnist-classification
Image classification using Graph Neural Networks (GNNs) with MNIST dataset

## Description
This repository is the implementation of paper [*A Graph Neural Network for superpixel image classification*](https://iopscience.iop.org/article/10.1088/1742-6596/1871/1/012071/pdf) by Jianwu Long , Zeran yan and Hongfa chen.

The authors of the paper propose to solve the Image Classification Task using Graph Neural Networks (GNNs).

## Creating a graph from an image
In order to use GNNs, each image must be converted into some graph. In this work [SLIC](https://www.epfl.ch/labs/ivrl/research/slic-superpixels/) algorithm is used for this. This algorithm segments a set of superpixels given an image. Each superpixels is considered as a graph node. Adjacent superpixels are connected with edges.

## Dataset
MNIST is used in this repository. Each digit image was converted into 75 superpixels using SLIC algorithm.

Here are some samples for better understanding:
![Superpixels](/all_classes.jpg?raw=true "Superpixels")
![Superpixels](/one_class.jpg?raw=true "Superpixels")

## Results 
One may find the experiment logs here: https://wandb.ai/acherepkov/mnist-gnn-classification/runs/dphbxqga.

I also published a tutorial for Google Colab: https://colab.research.google.com/drive/1d29NDjNMQ6I17rxTr8Wo6S-5nH3MGyXX?usp=sharing.

|                  | MNIST accuracy |
|------------------|-------------------|
| **Authors' results** | 97.11             |
| **My results**       | 74.59             |

Notes:
* The authors of the paper did not release any code and did not provide hyperparameters details like optimizer, learning rate, schedulers, linear layers sizes;
* I am not sure whether my [readout operation](https://github.com/Anton-Cherepkov/gnn-mnist-classification/blob/master/gnn_image_classification/model.py#L60) is correct;

The gap between my and authors' metrics may be due to these two facts.
