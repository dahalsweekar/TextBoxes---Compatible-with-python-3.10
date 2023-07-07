# Script Detection Network (SDN)

## Abstract
*This project introduces a trainable text detection model specifically designed for detecting Nepali handwritten scripts. The network is optimized for fast text detection on custom Nepali handwritten texts. It utilizes the VGG-16 architecture as its backbone and incorporates custom-designed top layers to handle the varying sizes of text found in camera-captured scripts.*

## Introduction
While scene text detection models have seen significant advancements in detecting texts on license plates, road signs, and product packages, there is a growing need for a model that can handle more complex, congested, and numerous instances of text found in scripts. Traditional scene text detection models typically deal with a limited number of texts within an image, where the main challenge is the complexity of the scene. However, our model is tailored to address different challenges specific to script detection.

## Architecture

![arch](https://github.com/dahalsweekar/ocr_service/assets/99968233/5a953f2b-7b41-4caf-87fb-d0b65506eaaa)
<sub>*figure 1: SDN architecture*</sub>

The Script Detection Network adopts the VGG-16 architecture as its backbone. This backbone is responsible for identifying features, which are then fed into fully convolutional networks. The top layers of the network consist of 6 convolutional layers accompanied by their corresponding pooling layers. In total, the top layer comprises 12 layers, carefully designed to accommodate the varying sizes of text present in the scripts.

## Experiments
 ### Datasets
 **Nepali Handwritten Dataset (NHD):** This dataset comprises 958 images and serves as both the training and testing dataset for the   model.

| Metrics        | HND spotting  |         
| ------------- |:-------------:| 
| Recall     | 0.596 | 
| Precision      | 0.907   |   
| F-measure | 0.720    |  

## Implementation details
The model is trained using stochastic gradient descent (SGD) with 300x300 pixel images. It undergoes 6.2k iterations during the training process. The training is performed on a GTX 960m 4GB GPU, taking approximately 120+ hours to complete.

## Running time
Approximately 0.26 seconds per image on GTX 960m GPU.

## Results

![collage](https://github.com/dahalsweekar/ocr_service/assets/99968233/647c8ad0-d4a8-4c07-8654-f0da1504c2a0)

## Getting Started
```
1. git clone https://github.com/dahalsweekar/Script-Detection-Network-SDN
2. cd Script-Detection-Network-SDN/caffe
3. pip install -r requirements.txt
4. make clean
5. make -j8
6. make py
```
## Dataset 
1. Download Dataset: https://github.com/dahalsweekar/Nepali-Handwritten-Dataset-Major-Collection
2. Create train/test lists (train.txt / test.txt) in "./data/text/" with the following form:
```
 path_to_example1.jpg path_to_example1.xml
 path_to_example2.jpg path_to_example2.xml
```
3. Run
 ```
./data/text/create_data.sh
 ```
## Train
 ```
1. run ./services/SDN/train.py
*Pretrained model: https://www.dropbox.com/scl/fo/vv09q0986hh36m1xw9ynr/h?rlkey=l9ic3f0zj2esynugwutwcp1z1&dl=0*
 ```
## Demo
```
1. run ./services/SDN/run.py
```
