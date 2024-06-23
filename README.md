# Fruit-Image-Classification

This project focuses on building a robust image classification model to accurately identify and classify various types of fruits in images. I did this project in Python and utilized deep learning libraries such as Tensorflow and Keras. 

# Dataset
Here is where I sourced the data from: https://www.kaggle.com/datasets/sshikamaru/fruit-recognition

The dataset consisted of 33 classes of different fruits, with each class containing images of the fruit in varying orientations against a white background. 

Total number of images: 22495.

Training set size: 16854 images (one fruit or vegetable per image).

Test set size: 5641 images (one fruit or vegetable per image).

Number of classes: 33 (fruits and vegetables).

# Model Architecture

### Layer Details
#### Conv2D Layer:

Filters: 64

Kernel Size: 3x3

Activation: ReLu

Strides: 1

Padding: Valid

Input Shape: (60, 60, 3)

#### MaxPooling2D Layer:

Pool Size: 2x2

Flatten Layer:

Converts the 2D matrix into a 1D vector

#### Dense Layer 1:

Units: 300

Activation: ReLU

#### Dense Layer 2:

Units: 100

Activation: ReLU

#### Dense Layer 3:

Units: 33 (number of classes)



Activation: Softmax

### Training Configuration
Optimizer: SGD (Stochastic Gradient Descent)

Loss Function: Sparse Categorical Crossentropy

Metrics: Accuracy

#### Hyperparameters

Learning Rate: Default (as defined by the SGD optimizer in Keras)

Batch Size: 64

Number of Epochs: 5

Validation Split: 30%

#### Preprocessing

Image Normalization: Input images are normalized by scaling pixel values to the range [0, 1].

# Model Performance
Final Accuracy after 5 epochs of training: 0.9746
Loss achieved the validation set after 5 epochs of training: 0.0957

<img width="645" alt="image" src="https://github.com/calebtran7/Fruit-Image-Classification/assets/121086856/38e046a3-f76e-460c-ba9b-cadf0d1ea983">


<img width="608" alt="image" src="https://github.com/calebtran7/Fruit-Image-Classification/assets/121086856/3e007e19-84b2-4f64-81ca-46f4c133f012">

Confusion Matrix:

<img width="728" alt="image" src="https://github.com/calebtran7/Fruit-Image-Classification/assets/121086856/2d1a2bd5-327c-4610-9539-6a5f3be29911">


## Feeding in custom images (not from dataset):

Ex 1) 

<img width="443" alt="image" src="https://github.com/calebtran7/Fruit-Image-Classification/assets/121086856/18b09450-b67d-445b-8798-9b9d35a88a9d">

Ex 2) 

<img width="443" alt="image" src="https://github.com/calebtran7/Fruit-Image-Classification/assets/121086856/58e65c8d-36d5-4552-af18-ce0e1700b924">

Ex 3) 

<img width="424" alt="image" src="https://github.com/calebtran7/Fruit-Image-Classification/assets/121086856/f6da226d-c1b4-43d3-8e07-82fb656d05ef">

Ex 4) 

<img width="434" alt="image" src="https://github.com/calebtran7/Fruit-Image-Classification/assets/121086856/1d836421-a038-4aae-a2d4-ab53c74ac359">





