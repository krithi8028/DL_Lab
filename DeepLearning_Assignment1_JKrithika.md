
# DEEPLEARNING
## ASSIGNMENT 1
### J.KRITHIKA
### 22011101046

## Problem Statement:
The goal of the experiment is to build a Convolutional Neural Network (CNN) to classify the images from the CIFAR-10 dataset. CNNs are particularly well-suited for image classification tasks due to their ability to automatically learn spatial hierarchies of features from input images.

## Data Description:
The dataset used for this experiment is the CIFAR-10 dataset, which consists of 60,000 32x32 color images in 10 different classes (airplanes, cars, birds, cats, deer, dogs, frogs, horses, ships, and trucks). There are 50,000 training images and 10,000 testing images. Each image is an RGB image of 32x32 pixels, and the goal is to classify each image into one of the 10 categories.

## Experiment Description:
We built a simple CNN model consisting of:

1. **Convolutional Layers**: 
   - The first convolutional layer has 32 filters of size 3x3, followed by a max-pooling layer.
   - The second convolutional layer has 64 filters of size 3x3, followed by another max-pooling layer.
   - The third convolutional layer also has 64 filters.

2. **Dense Layers**:
   - After flattening the output from the convolutional layers, a fully connected (dense) layer with 128 neurons is applied.
   - The output layer has 10 neurons, corresponding to the 10 classes in CIFAR-10, with a softmax activation function to predict class probabilities.

## Data Preprocessing:

The following steps were taken for data preprocessing:
1. **Normalization**: The pixel values of the images, which range from 0 to 255, were normalized to the range [0, 1].
2. **Reshaping**: As the dataset already comes with the correct 32x32x3 shape, no reshaping was necessary.

## Code:

```python 
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10

# Load and preprocess the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize images

# Build the CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')  # 10 classes for CIFAR-10
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(x_train, y_train, epochs=10, 
                    validation_data=(x_test, y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc}")
```

## Output:

![image](https://github.com/user-attachments/assets/2bbfdc7a-a180-4e16-92c8-302a6fe5085c)


### Model Training:
- The model was trained using the Adam optimizer with a sparse categorical crossentropy loss function.
- The training process involved 10 epochs, during which the model learned to minimize the classification error.

### Results:
After training for 10 epochs, the model achieved the following performance on the test dataset:
- **Test Accuracy**: 71.50%
- **Test Loss**: Indicates the error in the model's predictions.

## Interpretation of Results:

- The 71.50% accuracy on the test set shows that the model is able to correctly classify around 71.5% of the images from the CIFAR-10 dataset. This is a good start, especially given the simplicity of the model.
- The performance is reasonably good, but there is certainly room for improvement. Given that CIFAR-10 is a relatively complex dataset with small and detailed images, this performance is expected for a basic CNN architecture.

## Potential Improvements:

To improve the model's accuracy, the following strategies could be explored:
1. **Data Augmentation**: Applying transformations such as rotations, shifts, and flips to the training images could increase the model's generalization ability.
2. **More Complex Architecture**: Increasing the number of layers and filters, or introducing additional layers such as dropout or batch normalization, could improve the model’s ability to learn more complex patterns.
3. **Longer Training**: Training the model for more epochs may allow it to learn better.
4. **Hyperparameter Tuning**: Adjusting learning rates, batch sizes, and optimization methods could lead to better results.
5. **Regularization**: Techniques like L2 regularization or dropout can prevent overfitting and enhance the model's generalization to unseen data.

## Conclusion:

In this experiment, a basic CNN model was built to classify images from the CIFAR-10 dataset, and it achieved a test accuracy of 71.50%. While this is a promising result for a simple architecture, further refinements can be made to enhance the model's performance. Techniques such as data augmentation, more complex architectures, and tuning can push the accuracy closer to state-of-the-art performance on this dataset.
