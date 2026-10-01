import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
    Rescaling
)

from sklearn.metrics import confusion_matrix


# Main settings used throughout the model
IMAGE_SIZE = 100
BATCH_SIZE = 32
EPOCHS = 30

NUM_CLASSES = 81

DATASET_PATH = "fruits-360"


# Load the image dataset
# Use 70% of the images for training
# Keep the remaining 30% for testing
train_data = tf.keras.utils.image_dataset_from_directory(

    DATASET_PATH,

    validation_split=0.30,

    subset="training",

    seed=42,

    image_size=(100, 100),

    batch_size=32
)


test_data = tf.keras.utils.image_dataset_from_directory(

    DATASET_PATH,

    validation_split=0.30,

    subset="validation",

    seed=42,

    image_size=(100, 100),

    batch_size=32,

    shuffle=False
)


# Save the class names so predictions can be linked back to fruit labels
class_names = train_data.class_names

print("Number of fruit classes:", len(class_names))


# Build the CNN
model = Sequential()


# Scale pixel values from 0-255 to 0-1 so the network trains on a smaller range
model.add(
    Rescaling(
        1.0 / 255,
        input_shape=(100, 100, 3)
    )
)


# Start with 16 filters to learn simple visual patterns such as edges
# A small first layer keeps the early feature extraction simple
# A 2x2 kernel looks at small local regions of each image
# ReLU introduces non-linearity while keeping positive activations
model.add(
    Conv2D(
        16,
        (2, 2),
        activation="relu"
    )
)


# Pooling reduces feature-map size while keeping the strongest local features
model.add(
    MaxPooling2D(
        pool_size=(2, 2)
    )
)


# Increase the number of filters to learn more detailed visual patterns
# More filters allow the network to capture a wider range of features
model.add(
    Conv2D(
        32,
        (2, 2),
        activation="relu"
    )
)


# Reduce the feature-map size again before the next convolution
model.add(
    MaxPooling2D(
        pool_size=(2, 2)
    )
)


# Use a deeper convolution layer to capture more complex fruit features
# The deeper layer uses more filters for richer feature detection
model.add(
    Conv2D(
        64,
        (2, 2),
        activation="relu"
    )
)


# Final pooling step reduces the amount of data passed to the dense layers
model.add(
    MaxPooling2D(
        pool_size=(2, 2)
    )
)


# Dropout removes 30% of activations during training to help reduce overfitting
model.add(
    Dropout(0.3)
)


# Flatten the learned feature maps into a single vector for classification
model.add(
    Flatten()
)


# The dense layer combines the extracted visual features before classification
# 150 neurons provide capacity to combine the learned image features
model.add(
    Dense(
        150,
        activation="relu"
    )
)


# A second dropout layer adds more regularization before the output layer
model.add(
    Dropout(0.4)
)


# The output layer produces one probability for each fruit class
# There are 81 output neurons because the dataset contains 81 fruit classes
model.add(
    Dense(
        81,
        activation="softmax"
    )
)


#
# Compile the CNN
#
# RMSprop adjusts the learning rate during training
# Sparse categorical crossentropy is appropriate because the labels are integer class IDs
# Accuracy shows the proportion of fruit images classified correctly
#

model.compile(

    optimizer="rmsprop",

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]
)


#
# Display the layer structure and number of trainable parameters
#

model.summary()


#
# Train the CNN
#
# The batch size of 32 was set when the dataset was loaded
# Thirty epochs gives the model repeated passes over the training images
#

history = model.fit(

    train_data,

    epochs=30
)



# Evaluate the trained CNN on the held-out test images
test_loss, test_accuracy = model.evaluate(
    test_data
)


print("\nTEST RESULTS")

print(
    "Testing accuracy:",
    test_accuracy
)

print(
    "Testing accuracy percentage:",
    test_accuracy * 100,
    "%"
)



# Generate predicted class labels for the test images
actual = []

predicted = []


for images, labels in test_data:

    predictions = model.predict(
        images,
        verbose=0
    )

    predictions = np.argmax(
        predictions,
        axis=1
    )

    actual.extend(
        labels.numpy()
    )

    predicted.extend(
        predictions
    )


#
# Build a confusion matrix to compare actual and predicted fruit classes
#

matrix = confusion_matrix(
    actual,
    predicted
)


print("\nCONFUSION MATRIX")

print(matrix)


#
# Display the confusion matrix as a heatmap
#

plt.figure(
    figsize=(20, 20)
)

sns.heatmap(
    matrix,
    cmap="Blues"
)

plt.title(
    "Fruit Identification Confusion Matrix"
)

plt.xlabel(
    "Predicted Fruit"
)

plt.ylabel(
    "Actual Fruit"
)

plt.show()


#
# Plot training accuracy across epochs to see how learning progressed
#

plt.plot(
    history.history["accuracy"]
)

plt.title(
    "Training Accuracy"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Accuracy"
)

plt.show()