# Image-Processing-and-Classification-of-MNIST-and-CIFAR-10-using-ML-Techniques

Overview
This repository demonstrates the use of machine learning and deep learning techniques for image classification on the CIFAR-10 and MNIST datasets. It covers a range of preprocessing techniques, feature extraction methods, and model architectures, including traditional machine learning models like Support Vector Machines (SVM) and advanced deep learning architectures such as Convolutional Neural Networks (CNN) and Transfer Learning with VGG-16.

The project emphasizes the following:

Data preprocessing and augmentation techniques.
Custom filtering using convolution operations.
Feature extraction and model training.
Comparative analysis of models on multiple datasets.
Project Structure
Dataset Loading and Exploration:

CIFAR-10 and MNIST datasets are loaded using TensorFlow and PyTorch libraries, respectively.
Analysis of dataset structure, including image dimensions, classes, and data splits (training and test sets).
Image Preprocessing:

Normalization: Ensures pixel values are between 0 and 1.
Noise Reduction: Filters like median filtering and Gaussian smoothing are applied to enhance image quality.
Image Resizing: Standardizes image dimensions for uniform model inputs.
Grayscale Conversion: Reduces image complexity, especially for the MNIST dataset.
Custom Image Filtering:

A Python function is implemented to apply custom filters on image data. The function includes options for padding and normalization.
Example filters such as Sobel and Laplacian for edge detection are applied to the datasets.
HOG Feature Extraction and SVM Classification:

Histogram of Oriented Gradients (HOG) is used for feature extraction from the CIFAR-10 dataset.
A Linear SVM model is trained on these features, achieving an accuracy of 80.88% on the CIFAR-10 test set.
Neural Network Architectures:

Feedforward Neural Network (FNN): A simple, fully connected architecture is applied to both MNIST and CIFAR-10 datasets.
Convolutional Neural Network (CNN): A more advanced network is built for both datasets, with significant improvements in accuracy.
Transfer Learning using VGG-16: Pre-trained on ImageNet, the VGG-16 model is fine-tuned for CIFAR-10 and MNIST datasets, providing high classification accuracy.
Preprocessing Techniques
Normalization: Standardizes pixel values, aiding in faster convergence during training.
Noise Reduction: Techniques such as Gaussian smoothing and median filtering remove noise from images.
Resizing and Grayscale Conversion: Uniforms image input sizes and converts RGB images to grayscale where necessary.

Models and Results
1. Support Vector Machine (SVM) with HOG Features:
Applied to CIFAR-10 after HOG feature extraction.
Result: Achieved 80.88% accuracy on the CIFAR-10 test set.

3. Feedforward Neural Network (FNN):
Fully connected neural network applied to both MNIST and CIFAR-10 datasets.
MNIST Result: 99.08% training accuracy and 97.36% test accuracy.
CIFAR-10 Result: Lower performance due to dataset complexity compared to CNN models.

4. Convolutional Neural Network (CNN):
Built with two convolutional layers for MNIST and CIFAR-10.
MNIST Result: 99.35% test accuracy.
CIFAR-10 Result: Comparable performance with significant improvement over simple FNN models.

5. Transfer Learning with VGG-16:
Pre-trained VGG-16 model fine-tuned for both CIFAR-10 and MNIST datasets.
MNIST Result: Achieved 91.8% test accuracy.
CIFAR-10 Result: High accuracy through feature extraction with transfer learning.

Requirements
To run this project, ensure you have the following dependencies installed:

Python 3.7+
TensorFlow
PyTorch
NumPy
Scikit-learn
Matplotlib
tqdm


Install dependencies via:
pip install -r requirements.txt


Usage:
Clone the repository:
git clone https://github.com/your-repo/image-processing-ml.git

Install dependencies:
pip install -r requirements.txt

Run the Jupyter Notebook for each task:
jupyter notebook Image_Processing_ML.ipynb

Conclusion
This project demonstrates the effectiveness of combining image preprocessing techniques with machine learning and deep learning models. By leveraging advanced architectures like CNNs and VGG-16, the classification performance on complex datasets like CIFAR-10 can be significantly improved. Comparative analysis of models across both MNIST and CIFAR-10 datasets shows the superiority of deep learning techniques over traditional models like SVM.

License
This project is licensed under the MIT License.
