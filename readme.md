# Devanagiri Recognition 🇮🇳

[![](https://img.shields.io/github/license/sourcerer-io/hall-of-fame.svg?colorB=ff0000)](https://github.com/akshaybahadur21/Devanagiri-Recognizer/blob/master/LICENSE.txt)  [![](https://img.shields.io/badge/Akshay-Bahadur-brightgreen.svg?colorB=ff0000)](https://akshaybahadur.com)

This code helps you classify different alphabets of hindi language (Devanagiri) using Convnets

## Code Requirements 🦄
You can install Conda for python which resolves all the dependencies for machine learning.

## Description 🕉️
This code successfully recognizes hindi characters.

### Technique ⚙️

I have used convolutional neural networks.
I am using Tensorflow as the framework and Keras API for providing a high level of abstraction.

## Architecture 🏗️

1) CONV2D 
2) MAXPOOL 
3) CONV2D 
4) MAXPOOL 
5) FC
6) Softmax
7) Classification

### Notes 🗒️

1) You can go for additional conv layers.
2) Add regularization to prevent overfitting.
3) You can add additional images to the training set for increasing the accuracy.


## Python  Implementation 👨‍🔬

1) Dataset- DHCD (Devnagari Character Dataset)
2) Images of size 32 X 32
4) Convolutional Network Support added.

## Results 📊
<img src="https://github.com/akshaybahadur21/Devanagiri-Recognizer/blob/master/hindi.gif">

## Execution 🐉
To run the code, type `python Dev-Rec.py`

```
python Dev-Rec.py
```





