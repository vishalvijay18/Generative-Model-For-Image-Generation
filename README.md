# Generative Models For Image Generation

###Abstract
---
This repository contains code for artificial image generation using generative models, namely, Variational Autoencoders (VAE) and Generative Adversarial Networks (GAN). VAEs are appealing because they are built on top of standard function approximators (neural networks), and can be trained with stochastic gradient descent. In GAN, we train two models: a generative model G that captures the data distribution, and a discriminative model D that estimates the probability that a sample came from the training data rather than G.

###About library and code
---
I have used Keras library for this project. As mentioned on their webpage, it is a high-level neural networks library, written in Python and capable of running on top of either TensorFlow or Theano. It is easy to use and I would highly recommend it.

Code in this repository is heavily influenced from these articles and repositories:
- [Variational Autoencoder using Keras](https://github.com/fchollet/keras/blob/master/examples/variational_autoencoder.py)
- [Keras blog on autoencoders](https://blog.keras.io/building-autoencoders-in-keras.html)
- [DCGAN using Keras](https://github.com/jacobgil/keras-dcgan)
- [OpenAI Generative Models blog](https://openai.com/blog/generative-models/)

###Experiments
---
For this project I used my laptop with i5 processor for training. This restricted the traning set size and iterations I could run to get results in acceptable amount of time. The most interesting datasets were [[Celebrity Face Dataset (CFD)](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)] and [[New York Skyline Dataset (NYSD)](https://www.youtube.com/watch?v=DDo73Njxdqc)]. CFD provides a large set of images for training, but I used only 9000 images. NYSD was created using frames of a video, partitioned into 1200 training images and 125 test images. These images were later resized (32x32 or 64x64) to feed to the network. Our experiment setting is as follows:
