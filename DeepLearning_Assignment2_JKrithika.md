
# DEEP LEARNING
## ASSIGNMENT 2
**J KRITHIKA**  
**2201101046**

### Problem Description:
1. Implement CNN using any image dataset. Apply variants of convolution operations such as dilation, transpose convolution, etc.
2. Vary the number of convolution layers and different types of pooling with different filter sizes, etc. Record the results, analyze performance in terms of accuracy and the number of parameters. Write your observation.

### Dataset Overview:
- **Name**: CIFAR-10 (Canadian Institute for Advanced Research)
- **Size**: 60,000 images
- **Classes**: 10
- **Images per class**: 6,000
- **Image Size**: 32x32 pixels
- **Color Channels**: 3 (RGB)

#### Dataset Composition:
- **Training Set**: 50,000 images
- **Test Set**: 10,000 images

### SOURCE CODE:

```python
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
import numpy as np

# Load and preprocess the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize pixel values

def create_cnn_model_with_dilation():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu', dilation_rate=2),  # Dilation
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    return model

def create_cnn_model_with_transpose_conv():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same'),  # Transpose convolution
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    return model

def create_cnn_model_basic():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    return model

def compile_and_train(model):
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    history = model.fit(x_train, y_train, epochs=10, batch_size=64, validation_split=0.2)
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    return test_acc, model.count_params()

models = {
    "CNN_with_dilation": create_cnn_model_with_dilation(),
    "CNN_with_transpose_conv": create_cnn_model_with_transpose_conv(),
    "CNN_basic": create_cnn_model_basic()
}
results = {}
for name, model in models.items():
    print(f"Training {name}...")
    acc, params = compile_and_train(model)
    results[name] = {'accuracy': acc, 'parameters': params}
    print(f"{name} - Accuracy: {acc:.4f}, Parameters: {params}")
print(results)
```

### OUTPUT:
```
Training CNN_basic...
CNN_basic - Accuracy: 0.7102, Parameters: 225,034

Training CNN_with_dilation...
CNN_with_dilation - Accuracy: 0.6715, Parameters: 167,690

Training CNN_with_transpose_conv...
CNN_with_transpose_conv - Accuracy: 0.7036, Parameters: 233,290
```

### Result Analysis:
The results provided compare the performance of three different Convolutional Neural Network (CNN) models on an image classification task.

#### Model Comparison:
1. **CNN_basic**  
   - **Accuracy**: 0.7102  
   - **Parameters**: 225,034  
   
2. **CNN_with_dilation**  
   - **Accuracy**: 0.6715  
   - **Parameters**: 167,690  
   
3. **CNN_with_transpose_conv**  
   - **Accuracy**: 0.7036  
   - **Parameters**: 233,290  

### Analysis:
1. **Accuracy Comparison**:  
   - The **CNN_basic** model has the highest accuracy at **0.7102**.  
   - The **CNN_with_transpose_conv** model is slightly behind with an accuracy of **0.7036**.  
   - The **CNN_with_dilation** model has the lowest accuracy at **0.6715**.  

2. **Parameters Comparison**:  
   - The **CNN_with_dilation** model has the fewest parameters (**167,690**), making it the most efficient in terms of memory and computational resources.  
   - The **CNN_with_transpose_conv** model has the most parameters (**233,290**), which could explain its high accuracy, though it is slightly less than the basic model.  
   - The **CNN_basic** model has **225,034** parameters, which is a moderate number compared to the other two.  

### Conclusion:
- The **CNN_basic** model demonstrates the best performance in terms of accuracy (**0.7102**), making it the most effective model for this task among the three, despite not being the most parameter-efficient.  
- The **CNN_with_transpose_conv** model achieves a similar accuracy (**0.7036**) but at the cost of having the highest number of parameters (**233,290**), indicating that it might be overfitting slightly or not fully utilizing the added parameters for better accuracy.  
- The **CNN_with_dilation** model, although the most parameter-efficient (**167,690 parameters**), has the lowest accuracy (**0.6715**), suggesting that while it is more resource-efficient, it compromises on performance.

### Recommendation:
If computational resources and memory are not a constraint, **CNN_basic** offers the best trade-off between accuracy and model complexity. However, if model size is a significant concern, **CNN_with_dilation** might be preferred, though with an expected drop in accuracy.
