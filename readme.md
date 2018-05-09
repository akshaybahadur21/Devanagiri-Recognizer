## Devanagiri Recognition
This code helps you classify different alphabets of hindi language (Devanagiri) using Convnets


### Code Requirements
You can install Conda for python which resolves all the dependencies for machine learning.

### Description
This code successfully recognizes hindi characters.

### Technique Used

I have used convolutional neural networks.
I am using Tensorflow as the framework and Keras API for providing a high level of abstraction.

### Architecture

#### CONV2D --> MAXPOOL --> CONV2D --> MAXPOOL -->FC -->Softmax--> Classification


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





