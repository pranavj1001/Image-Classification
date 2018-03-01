# Image-Classification
A short template to classify images using convolutional neural networks.

Hey there,

This repo contains a python script which can be used to classify images using **Convolutional Neural Networks**. 
There are two convolutional layers. For Pooling I've used Max Pooling.

It uses **keras** library to do much of the behind the scenes work. 
Interested in keras? Find out more info about it from [here](https://keras.io/).

## Before Trying this code

* Make sure you have [python](https://www.python.org/) installed on your machine.
* Make sure you have installed [Theano](http://deeplearning.net/software/theano/), [Tensorflow](https://www.tensorflow.org/) and [Keras](https://keras.io/) on your machine.

**Note**: If you have a dedicated GPU then it is recommended to install tensorflow with gpu enabled.

## Usage

Just paste the images in appropriate folders under 'dataset' folder. Then, just run the script. That's it! 
You just trained your artificial neural network with data and tested it out.

**Note**: 
* ~~This program assumes that you have only two classes to classify from images. Slight changes need to be done if you have multiple classes of images. In future, I'll add a separate python script for this.~~

* I've added the script which can handle multiple classes. So if you have two classes then goto folder 'Two_Classes' else goto folder 'More_than_two_Classes'

* Also, please try to play with the code. Add/remove layers, change units, check the accuracy and then only use the code in production.

## Queries?

email me at pranavj1001@gmail.com

## License

MIT License
