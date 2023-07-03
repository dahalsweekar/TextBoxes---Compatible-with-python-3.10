# Script-Detection-Network-SDN
Script Detection Network (SDN) is a text detection model that is designed to detect intricate handwritings. The model is trained on nepali handwritten dataset which is collected from various sources. The repository contains the implementation code, trained model, and collected datasets.

# Getting Started
```
1. git clone https://github.com/dahalsweekar/Script-Detection-Network-SDN
2. cd Script-Detection-Network-SDN/caffe
3. pip install -r requirements.txt
4. make clean
5. make -j8
6. make py
```
# Dataset 
1. Download Dataset: 
2. Create train/test lists (train.txt / test.txt) in "./data/text/" with the following form:
```
 path_to_example1.jpg path_to_example1.xml
 path_to_example2.jpg path_to_example2.xml
```
3. Run
 ```
./data/text/create_data.sh
 ```
# Train
 ```
1.run ./services/SDN/train.py
 ```
