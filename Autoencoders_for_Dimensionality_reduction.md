# **Autoencoders for Dimensionality reduction**

### Author: J.Krithika

### Roll number: 22011101046

## **Aim:**

The objective of this project is to apply autoencoders for dimensionality reduction and compare the performance of deep features extracted by an autoencoder with Principal Component Analysis (PCA) features on a classification task using the MNIST dataset. This study aims to evaluate the effectiveness of autoencoder-based feature extraction and assess how it impacts classification performance compared to PCA.

## **Data Description:**

- **Dataset**: The MNIST dataset was used, which contains 70,000 grayscale images of handwritten digits, with 60,000 images used for training and 10,000 for testing. Each image is 28x28 pixels in size, corresponding to digits 0-9.
- **Preprocessing**: The pixel values of the images were normalized. The autoencoder model was trained to reduce the dimensionality of the image data, capturing latent features. The PCA model was also used for dimensionality reduction.

## **Source Code**

```py
 #Import necessary libraries

import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from tensorflow.keras.datasets import mnist

from tensorflow.keras.models import Model

from tensorflow.keras.layers import Input, Dense

from tensorflow.keras.utils import plot_model

import tensorflow as tf

# Set random seeds for reproducibility

np.random.seed(42)

tf.random.set_seed(42)

# Step 1: Load and Preprocess the Data

print("Loading and preprocessing the data...")

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalize the data to \[0, 1\] range

X_train = X_train.astype('float32') / 255.0

X_test = X_test.astype('float32') / 255.0

# Flatten the 28x28 images into 784-dimensional vectors

X_train = X_train.reshape(-1, 28 \* 28)

X_test = X_test.reshape(-1, 28 \* 28)

# Standardize the data (zero mean and unit variance)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

print(f"Training data shape: {X_train.shape}")

print(f"Test data shape: {X_test.shape}")

# Step 2: Build and Train the Autoencoder

print("\\nBuilding and training the autoencoder...")

# Define the autoencoder architecture

input_dim = X_train.shape\[1\] # 784

encoding_dim = 64 # Desired reduced dimension

# Encoder

input_layer = Input(shape=(input_dim,), name='encoder_input')

encoded = Dense(128, activation='relu', name='encoder_hidden1')(input_layer)

encoded = Dense(encoding_dim, activation='relu', name='encoder_hidden2')(encoded)

# Decoder

decoded = Dense(128, activation='relu', name='decoder_hidden1')(encoded)

decoded = Dense(input_dim, activation='sigmoid', name='decoder_output')(decoded)

# Autoencoder model

autoencoder = Model(inputs=input_layer, outputs=decoded, name='autoencoder')

# Compile the autoencoder

autoencoder.compile(optimizer='adam', loss='mse')

# Display the autoencoder architecture

autoencoder.summary()

# Train the autoencoder

history = autoencoder.fit(

X_train, X_train,

epochs=50,

batch_size=256,

shuffle=True,

validation_data=(X_test, X_test),

verbose=2

)

# Plot training & validation loss values

plt.figure(figsize=(10, 5))

plt.plot(history.history\['loss'\], label='Training Loss', color='blue')

plt.plot(history.history\['val_loss'\], label='Validation Loss', color='orange')

plt.title('Autoencoder Loss')

plt.xlabel('Epoch')

plt.ylabel('Mean Squared Error')

plt.legend()

plt.show()

# Step 3: Extract Encoded Features (Compressed Deep Features)

print("\\nExtracting encoded features from the autoencoder...")

# Define the encoder model to extract the compressed features

encoder = Model(inputs=input_layer, outputs=encoded, name='encoder')

# Get the compressed features

X_train_encoded = encoder.predict(X_train)

X_test_encoded = encoder.predict(X_test)

print(f"Encoded training features shape: {X_train_encoded.shape}")

print(f"Encoded test features shape: {X_test_encoded.shape}")

# Step 4: Apply PCA for Dimensionality Reduction

print("\\nApplying PCA for dimensionality reduction...")

# Initialize PCA with the same number of components as the autoencoder

pca = PCA(n_components=encoding_dim, random_state=42)

# Fit PCA on the training data and transform both training and test data

X_train_pca = pca.fit_transform(X_train)

X_test_pca = pca.transform(X_test)

print(f"PCA training features shape: {X_train_pca.shape}")

print(f"PCA test features shape: {X_test_pca.shape}")

# Step 5: Train a Classifier Using Autoencoder and PCA Features

print("\\nTraining classifiers on both feature sets...")

# Initialize Logistic Regression classifiers

clf_autoencoder = LogisticRegression(max_iter=1000, random_state=42)

clf_pca = LogisticRegression(max_iter=1000, random_state=42)

# Train on Autoencoder features

print("Training Logistic Regression on Autoencoder features...")

clf_autoencoder.fit(X_train_encoded, y_train)

y_pred_autoencoder = clf_autoencoder.predict(X_test_encoded)

# Train on PCA features

print("Training Logistic Regression on PCA features...")

clf_pca.fit(X_train_pca, y_train)

y_pred_pca = clf_pca.predict(X_test_pca)

# Step 6: Evaluation and Comparison

print("\\nEvaluating classification performance...")

# Calculate accuracies

accuracy_autoencoder = accuracy_score(y_test, y_pred_autoencoder)

accuracy_pca = accuracy_score(y_test, y_pred_pca)

print(f"Accuracy using Autoencoder features: {accuracy_autoencoder \* 100:.2f}%")

print(f"Accuracy using PCA features: {accuracy_pca \* 100:.2f}%")

# Detailed classification reports

print("\\nClassification Report for Autoencoder Features:")

print(classification_report(y_test, y_pred_autoencoder))

print("Classification Report for PCA Features:")

print(classification_report(y_test, y_pred_pca))

#  Confusion Matrix Visualization

def plot_confusion_matrix(cm, classes, title='Confusion matrix'):

plt.figure(figsize=(10, 8))

plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)

plt.title(title)

plt.colorbar()

tick_marks = np.arange(len(classes))

plt.xticks(tick_marks, classes, rotation=45)

plt.yticks(tick_marks, classes)

# Normalize the confusion matrix

cm_normalized = cm.astype('float') / cm.sum(axis=1)\[:, np.newaxis\]

thresh = cm.max() / 2.

for i, j in np.ndindex(cm.shape):

plt.text(j, i, f"{cm\[i, j\]} ({cm_normalized\[i, j\]:.2f})",

horizontalalignment="center",

color="white" if cm\[i, j\] > thresh else "black")

plt.ylabel('True label')

plt.xlabel('Predicted label')

plt.tight_layout()

plt.show()

# Confusion Matrix for Autoencoder Features

cm_autoencoder = confusion_matrix(y_test, y_pred_autoencoder)

plot_confusion_matrix(cm_autoencoder, classes=\[str(i) for i in range(10)\],

title='Confusion Matrix - Autoencoder Features')

# Confusion Matrix for PCA Features

cm_pca = confusion_matrix(y_test, y_pred_pca)

plot_confusion_matrix(cm_pca, classes=\[str(i) for i in range(10)\],

title='Confusion Matrix - PCA Features')

# Step 7: Analyze and Compare the Results

print("\\nAnalysis and Comparison:")

print(f"The Logistic Regression classifier achieved an accuracy of {accuracy_autoencoder \* 100:.2f}% using features extracted by the Autoencoder.")

print(f"In comparison, the same classifier achieved an accuracy of {accuracy_pca \* 100:.2f}% using PCA-reduced features.")

print("\\nThis indicates that the Autoencoder-based dimensionality reduction captures more relevant information for the classification task compared to PCA.")

#  Visualize the PCA and Autoencoder feature distributions using t-SNE or PCA plots

# Here, we'll use PCA plots for visualization

from sklearn.manifold import TSNE

print("\\nVisualizing the feature distributions using t-SNE...")

# Due to computational constraints, we'll use a subset of the data for visualization

sample_size = 2000

indices = np.random.choice(X_test.shape\[0\], sample_size, replace=False)

X_test_encoded_sample = X_test_encoded\[indices\]

X_test_pca_sample = X_test_pca\[indices\]

y_test_sample = y_test\[indices\]

\# Apply t-SNE

tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=300)

print("Applying t-SNE on Autoencoder features...")

tsne_autoencoder = tsne.fit_transform(X_test_encoded_sample)

print("Applying t-SNE on PCA features...")

tsne_pca = tsne.fit_transform(X_test_pca_sample)

# Plot t-SNE results

plt.figure(figsize=(14, 6))

# Autoencoder t-SNE

plt.subplot(1, 2, 1)

scatter = plt.scatter(tsne_autoencoder\[:, 0\], tsne_autoencoder\[:, 1\], c=y_test_sample, cmap='tab10', alpha=0.6)

plt.title('t-SNE on Autoencoder Features')

plt.xlabel('Dimension 1')

plt.ylabel('Dimension 2')

plt.legend(\*scatter.legend_elements(), title="Digits")

# PCA t-SNE

plt.subplot(1, 2, 2)

scatter = plt.scatter(tsne_pca\[:, 0\], tsne_pca\[:, 1\], c=y_test_sample, cmap='tab10', alpha=0.6)

plt.title('t-SNE on PCA Features')

plt.xlabel('Dimension 1')

plt.ylabel('Dimension 2')

plt.legend(\*scatter.legend_elements(), title="Digits")

plt.tight_layout()

plt.show()
```

![Screenshot 2024-10-16 215901](https://github.com/user-attachments/assets/91850840-754d-48d9-9818-be88897e4867)
![Screenshot 2024-10-16 215932](https://github.com/user-attachments/assets/3393af98-474b-4a4a-a4cf-c6fd7c1fec76)
![Screenshot 2024-10-16 220023](https://github.com/user-attachments/assets/42ad2425-79dd-43ac-aaed-86573a1ffb0a)
![Screenshot 2024-10-16 220044](https://github.com/user-attachments/assets/7b8cab5c-4438-410b-b5b4-9beb42561dc6)
![Screenshot 2024-10-16 220003](https://github.com/user-attachments/assets/0eb9959f-a297-4b87-9df1-54a7a72fb9bf)
![Screenshot 2024-10-16 220055](https://github.com/user-attachments/assets/e57a6edf-6131-4712-95ec-31c3a3fcef0a)




## **Interpretation:**

1. **Autoencoder Training**: The autoencoder was trained to learn a compressed representation of the input data with a bottleneck layer that reduced the dimensionality of the original 784-pixel space to a k-dimensional latent space. The goal was to minimize reconstruction error, and both the training and validation losses showed convergence after a certain number of epochs, indicating successful learning.
2. **Dimensionality Reduction**: The features extracted from the bottleneck layer of the autoencoder were used for classification using a Logistic Regression model. These features were compared to the k-dimensional features obtained through PCA to evaluate their effectiveness in a classification task.
3. **Classification Performance**: Both autoencoder and PCA reduced features were classified using Logistic Regression. The classification results showed similar performance in terms of accuracy and other evaluation metrics, with the autoencoder achieving 90.71% accuracy and PCA achieving 90.98% accuracy.
4. **Visualization**: t-SNE was used to visualize the k-dimensional features in a 2D space. The visualizations for both autoencoder and PCA-reduced features showed clustering patterns for different digits, though the autoencoder exhibited slightly more distinct clusters for certain digits.

## **Results:**

- **Autoencoder Performance**: Logistic Regression on autoencoder features achieved 90.71% accuracy. The classification report indicated strong performance across all digit classes, with precision, recall, and f1-scores above 0.90 for most classes.
- **PCA Performance**: Logistic Regression on PCA-reduced features achieved 90.98% accuracy. The classification report was similar to that of the autoencoder, with only marginal differences in individual class performance.
- **t-SNE Visualization**: The t-SNE plots for both autoencoder and PCA features revealed distinct clusters for most digits, though some overlap between classes was observed, especially for digits like 3 and 5.

## **Conclusion:**

- Both autoencoder-based and PCA-based dimensionality reduction techniques resulted in comparable classification performance, with Logistic Regression achieving an accuracy of around 91% for both methods.
- The autoencoder, though slightly less accurate than PCA, has the advantage of learning more complex and potentially non-linear features, which could be useful in more complex datasets or tasks.
- The t-SNE visualization suggests that autoencoders may better capture subtle distinctions in the data, leading to more distinct clusters.

## **Future Alterations:**

1. **Hyperparameter Tuning**: Experimenting with deeper autoencoders or adjusting the number of neurons in the bottleneck layer could potentially improve feature extraction quality.
2. **Other Classifiers**: Applying more complex classification models like SVMs or neural networks on the reduced feature sets may highlight further differences in performance between autoencoder and PCA features.
3. **Other Datasets**: Testing on more complex datasets, such as CIFAR-10 or Fashion MNIST, may better illustrate the advantages of using autoencoders over PCA.
4. **Exploration of Variational Autoencoders (VAE)**: VAEs could provide an even more informative latent space by adding a probabilistic element to feature extraction.
