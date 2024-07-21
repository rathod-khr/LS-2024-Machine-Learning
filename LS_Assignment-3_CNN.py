#As loading files were difficult in google colab, I've run this in pyscripter
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Define dataset path and parameters
dataset_path = 'C:/Users/udayv/Downloads/dataset'
batch_size = 32
img_size = (64, 64)

# Load the dataset
dataset = image_dataset_from_directory(
    dataset_path,
    image_size=img_size,
    batch_size=batch_size,
    label_mode='int'  # Use 'int' for integer labels or 'categorical' for one-hot encoding
)

# Normalize pixel values to be between 0 and 1
def normalize_image(image, label):
    image = image / 255.0  # Normalize pixel values
    return image, label

# Map the normalization function to the dataset
dataset = dataset.map(lambda image, label: normalize_image(image, label))

# Convert dataset to NumPy iterator
data_iterator = dataset.as_numpy_iterator()

# Convert iterator to lists
images_list = []
labels_list = []

for images, labels in data_iterator:
    images_list.append(images)
    labels_list.append(labels)

# Convert lists to numpy arrays
images_array = np.concatenate(images_list, axis=0)
labels_array = np.concatenate(labels_list, axis=0)

# Calculate the number of samples
num_samples = images_array.shape[0]
print(f'Total number of samples: {num_samples}')

# Define split ratios
train_split = 0.7  # 70% for training
val_split = 0.15    # 15% for validation
test_split = 0.15   # 15% for testing

# Calculate the number of samples for each split
train_size = int(train_split * num_samples)
val_size = int(val_split * num_samples)
test_size = num_samples - train_size - val_size  # Remaining samples for test

# Split the data
train_images = images_array[:train_size]
train_labels = labels_array[:train_size]

val_images = images_array[train_size:train_size+val_size]
val_labels = labels_array[train_size:train_size+val_size]

test_images = images_array[train_size+val_size:]
test_labels = labels_array[train_size+val_size:]

print(f'Training data shape: {train_images.shape}, {train_labels.shape}')
print(f'Validation data shape: {val_images.shape}, {val_labels.shape}')
print(f'Test data shape: {test_images.shape}, {test_labels.shape}')

# Build the CNN model
model = Sequential()

# Convolutional Layer
model.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Adding a second convolutional layer
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flattening
model.add(Flatten())

# Fully Connected Layer
model.add(Dense(units=128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=4, activation='softmax'))  # Adjust the number of units based on the number of classes

# Compile the CNN
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_images,
    train_labels,
    validation_data=(val_images, val_labels),
    epochs=25
)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print(f'Test loss: {test_loss}, Test accuracy: {test_accuracy}')
