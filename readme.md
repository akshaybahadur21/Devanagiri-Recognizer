# Devanagiri Recognition [![](https://img.shields.io/github/license/sourcerer-io/hall-of-fame.svg?colorB=ff0000)](https://github.com/akshaybahadur21/Devanagiri-Recognizer/blob/master/LICENSE.txt)  [![](https://img.shields.io/badge/Akshay-Bahadur-brightgreen.svg?colorB=ff0000)](https://akshaybahadur.com)
This code helps you classify different alphabets of hindi language (Devanagiri) using Convnets

### Sourcerer
[![](https://sourcerer.io/fame/akshaybahadur21/akshaybahadur21/Devanagiri-Recognizer/images/0)](https://sourcerer.io/fame/akshaybahadur21/akshaybahadur21/Devanagiri-Recognizer/links/0)[![](https://sourcerer.io/fame/akshaybahadur21/akshaybahadur21/Devanagiri-Recognizer/images/1)](https://sourcerer.io/fame/akshaybahadur21/akshaybahadur21/Devanagiri-Recognizer/links/1)[![](https://sourcerer.io/fame/akshaybahadur21/akshaybahadur21/Devanagiri-Recognizer/images/2)](https://sourcerer.io/fame/akshaybahadur21/akshaybahadur21/Devanagiri-Recognizer/links/2)[![](https://sourcerer.io/fame/akshaybahadur21/akshaybahadur21/Devanagiri-Recognizer/images/3)](https://sourcerer.io/fame/akshaybahadur21/akshaybahadur21/Devanagiri-Recognizer/links/3)[![](https://sourcerer.io/fame/akshaybahadur21/akshaybahadur21/Devanagiri-Recognizer/images/4)](https://sourcerer.io/fame/akshaybahadur21/akshaybahadur21/Devanagiri-Recognizer/links/4)[![](https://sourcerer.io/fame/akshaybahadur21/akshaybahadur21/Devanagiri-Recognizer/images/5)](https://sourcerer.io/fame/akshaybahadur21/akshaybahadur21/Devanagiri-Recognizer/links/5)[![](https://sourcerer.io/fame/akshaybahadur21/akshaybahadur21/Devanagiri-Recognizer/images/6)](https://sourcerer.io/fame/akshaybahadur21/akshaybahadur21/Devanagiri-Recognizer/links/6)[![](https://sourcerer.io/fame/akshaybahadur21/akshaybahadur21/Devanagiri-Recognizer/images/7)](https://sourcerer.io/fame/akshaybahadur21/akshaybahadur21/Devanagiri-Recognizer/links/7)

### Code Requirements
You can install Conda for python which resolves all the dependencies for machine learning.

### Description
This code successfully recognizes hindi characters.

### Technique Used

I have used convolutional neural networks.
I am using Tensorflow as the framework and Keras API for providing a high level of abstraction.

### Architecture
It will recognize Hindi alphabets

#### CONV2D --> MAXPOOL --> CONV2D --> MAXPOOL -->FC -->Softmax--> Classification

### Some additional points

1) You can go for additional conv layers.
2) Add regularization to prevent overfitting.
3) You can add additional images to the training set for increasing the accuracy.


### Python  Implementation

1) Dataset- DHCD (Devnagari Character Dataset)
2) Images of size 32 X 32
4) Convolutional Network Support added.

### Train Acuracy ~ 95%
### Test Acuracy ~ 92%

<img src="https://github.com/akshaybahadur21/Devanagiri-Recognizer/blob/master/hindi.gif">

### Execution for writing through webcam
To run the code, type `python Dev-Rec.py`

```
python Dev-Rec.py
```





