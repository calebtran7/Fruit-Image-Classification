# ClassiFruit: Fruit Image Classifier Using CNNs and PyTorch
### Author: Caleb Tran

## Project Overview:
This project focuses on building a robust image classification model to accurately identify and classify various types of fruits in images. I did this project in Python and utilized deep learning libraries such as PyTorch. In the Streamlit interface I created, users can upload an image and receive the top 3 predicted fruit types with confidence scores.

# Technologies Used

PyTorch: Model training and architecture

Torchvision: Dataset handling and transforms

Streamlit: Frontend deployment

Matplotlib & Seaborn: Visualization

Scikit-learn: Evaluation metrics

# Demo

https://github.com/user-attachments/assets/f1b8719a-3c4c-4fcf-888e-bff32e36ce86

# Dataset
Here is where I sourced the data from: https://www.kaggle.com/datasets/sshikamaru/fruit-recognition

The dataset consisted of 33 classes of different fruits, with each class containing images of the fruit in varying orientations against a white background. 

Dataset size: 16854 images (one fruit or vegetable per image).

Data was split into 80% training / 20% testing

Number of classes: 33 (fruits and vegetables).

Example dataset images:

![Peach_0](https://github.com/user-attachments/assets/30024d18-f575-43e6-ab9e-1196c029a02d)

Label: Peach


![Watermelon_1](https://github.com/user-attachments/assets/7d9f8715-b93f-4ab3-85cf-8c5171b7a545)

Label: Watermelon

# Model Overview
The CNN consists of:

- 1 convolutional layer (64 filters)

- Max pooling and dropout

- 3 fully connected layers

- Softmax output for 33 fruit classes

- Regularization via dropout and data augmentation prevents overfitting.

# Training/validation accuracy and loss plots:

<img width="586" alt="image" src="https://github.com/user-attachments/assets/e03397ab-fc1f-4220-b71f-b12a6fc7fee5" />

<img width="581" alt="image" src="https://github.com/user-attachments/assets/d5b7e81a-d050-4357-86d6-7ee3a93a38fc" />


# Confusion matrix:

![image](https://github.com/user-attachments/assets/4048b37a-b49a-4059-a2c2-dca8dbc3d13a)

# What's Next?
- Add top-K bar chart or probability distribution plot

- Enable mobile responsiveness

- Further model enhancement to increase accuracy & robustness

- Adapt interface/features for a specific purpose e.g. agricultural sorting, automated checkout in grocery stores, nutrition apps, accessibility tool, etc.

- Extend the model to accept live video input using webcam integration, and apply real-time fruit detection via object detection frameworks (e.g., YOLO, Detectron2)


