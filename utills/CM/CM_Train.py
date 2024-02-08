from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, LSTM, TimeDistributed, Reshape
from processors.DataPreprocessor import DataGenerator
from FileHandler import FileHandler
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
import keras

import tensorflow as tf
print(tf.__version__)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


le= ["jpg","jpeg"]

fp = FileHandler(dataset_path=r"SKIPPD",foldername="CloudMask",legal_extensions=le,include_subdirectories=False)
fp = fp.path_file()
folder_paths = fp




# Assuming folder_paths is a list of image file paths
image_paths = folder_paths  # This should be the list of all image paths
train_paths, val_paths = train_test_split(image_paths, test_size=0.2)

train_generator = DataGenerator(train_paths, batch_size=5, sequence_length=15)
val_generator = DataGenerator(val_paths, batch_size=5, sequence_length=15)




# Define model architecture
model = Sequential()
# CNN part for feature extraction


model.add(TimeDistributed(Conv2D(10, (5, 5), activation='linear'), input_shape=(15, 64, 64, 1)))
model.add(TimeDistributed(MaxPooling2D(4, 4)))
model.add(TimeDistributed(Flatten()))
model.add(LSTM(100, return_sequences=True))
model.add(TimeDistributed(Dense(64*64*1, activation='linear')))  
model.add(Reshape((15, 64, 64, 1)))  
model.compile(optimizer='adam', loss=keras.losses.MeanAbsoluteError())  

# Model summary
model.summary()

# Train the model
model.fit(train_generator, validation_data=val_generator, epochs=20)
model.save('location.keras')