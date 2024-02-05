from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, LSTM, TimeDistributed, Reshape
from processors.DataPreprocessor import DataPreprocessor
from FileHandler import FileHandler
from PIL import Image
import numpy as np

import tensorflow as tf
print(tf.__version__)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


#le= ["jpg","jpeg"]
#
#fp = FileHandler(dataset_path=r"SKIPPD",foldername="CloudMask",legal_extensions=le,include_subdirectories=False)
#fp = fp.path_file()
#
#
#data_preprocessor = DataPreprocessor(fp)
#X_train, X_val, Y_train, Y_val = data_preprocessor.get_data()
#
#
#with open('test.npy', 'wb') as f:
#    np.save(f,  X_train)
#    np.save(f,  X_val)
#    np.save(f,  Y_train)
#    np.save(f,  Y_val)


with open('test.npy', 'rb') as f:
    X_train = np.load(f)
    Y_train = np.load(f)
    X_val = np.load(f)
    Y_val = np.load(f)


print(f"Training Input Shape: {X_train.shape}")
print(f"Training Output Shape: {Y_train.shape}")
print(f"Validation Input Shape: {X_val.shape}")
print(f"Validation Output Shape: {Y_val.shape}")

# Define model architecture
model = Sequential()
# CNN part for feature extraction
model.add(TimeDistributed(Conv2D(32, (3, 3), activation='relu'), input_shape=(15, 64, 64, 3)))
model.add(TimeDistributed(MaxPooling2D(2, 2)))
model.add(TimeDistributed(Flatten()))
# LSTM part for sequencing
model.add(LSTM(256, return_sequences=True))
# Output layer
model.add(TimeDistributed(Dense(64*64*3, activation='sigmoid')))  # Adjust the activation function and units according to your case
model.add(Reshape((15, 64, 64, 3)))  # Reshape output to match the sequence of images

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')  # Adjust according to your task

# Model summary
model.summary()

# Train the model
model.fit(X_train, Y_train, epochs=20, batch_size=1)

# Replace `X_train` and `Y_train` with your training data and labels respectively
