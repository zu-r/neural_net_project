# LeNet5 Project
LeNet (1989 - 1998) is a series of neural nets developed by LeCun et al. to solve the problem of recognizing handwritten digit images

## Basic Implementation

The network receives a 32x32 input image of a handwritten digit. It then passes four convolutional layers reducing the input to 16 maps of 5x5 grids. The next layer (C5) is labelled as a convolutional layer but acts exactly as a fully connected layer. Hence, the next two layers (C5 and F6) are fully connected layers reducing the 16 maps to just 84 neurons. This layer is then compared with predefined constant 7x12 drawings of digits, and the final classification is the predefined drawing most similar to the 84 neuron-wide layer

After 20 epochs, the model outputs a 98.61% accuracy on the training dataset and a 98.38% accuracy on the test dataset.


# Improved LeNet5

In order to allow LeNet to handle unseen data that may contain transformed data, we implemented the following changes

### Modifications and Improvements:

1. **Modified Training Dataset**:  
   * Modified the training data set by taking a certain portion of the images in the original training data set and applied rotations, scalings, translations, and shearings to simulate unseen dataset as described in the affNIST dataset.

2. **Activation Function**:  
   * Replaced tanh activation function with ReLU function for all layers to learn faster and achieving overall better computational efficiency.

3. **Downsampling**:  
   * We used Max pooling in a separate layer instead of strided convolutions to focus on the most dominant features in a region and ensure translational invariance.

4. **Output Layer**:  
   * Replace the RBF layer with a fully connected layer for simplicity.

5. **Loss Function**:  
   * We replaced our original loss function with Cross Entropy Loss for simplicity and computational efficiency. It is also the standard for multi-class classification problems.

6. **Momentum**:  
   * We incorporated momentum into our optimizer to reduce oscillations and reach optimal solution quicker.
  
We were able to achieve approximately 97% accuracy on the original MNIST dataset when training on a combination of transformed images and untransformed images. During the training phase of the model, the training and test accuracies reached approximately 80% on the dataset which consisted of a combination of transformed and untransformed images.

