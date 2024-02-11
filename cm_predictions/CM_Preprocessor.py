import numpy as np
import matplotlib.pyplot as plt
import keras
from keras import layers
from  FileHandler import FileHandler
from PIL import Image
import tensorflow as tf
import random
from CM_DataGenerator import DataGenerator
import os

le= ["jpg","jpeg"]
fp = FileHandler(dataset_path=r"SKIPPD",foldername="CloudMask",legal_extensions=le,include_subdirectories=False)
fp = fp.path_file()
folder_paths = fp



# Function to load images and convert to numpy arrays
def load_image(infilename):
    img = Image.open(infilename)
    img.load()
    data = np.asarray(img, dtype="float32") / 255
    # Assuming images are grayscale, add a channel dimension
    data = np.expand_dims(data, axis=-1)
    return data

# Load images
dataset = np.array([load_image(img) for img in fp])

# Define sequence length and other parameters
seq_length = 15

def create_shifted_frames(data, sequence_length,threshold=1000):
    X, Y = [], []
    for i in range(data.shape[0] - sequence_length):
        # Extract the input sequence
        sequence_X = data[i:i+sequence_length, :, :, :]
        
        # Extract the corresponding output sequence
        # Assuming the output directly follows the input
        sequence_Y = data[i+1:i+1+sequence_length, :, :, :]
        sequence_sum = np.sum(sequence_X)

        if sequence_sum > threshold:
            X.append(sequence_X)
            Y.append(sequence_Y)
    
    return np.array(X), np.array(Y)

# Creating sequences

# Now train_X, train_Y, val_X, and val_Y are ready for your model
processed_data_path = 'processed_data.npz'  # Path to save/load processed data

# Function to check and load processed data if it exists
def load_processed_data(filepath):
    if os.path.exists(filepath):
        data = np.load(filepath,allow_pickle=True)
        return (data['train_X'], data['train_Y'], data['val_X'], data['val_Y'])
    else:
        return None

# Function to save processed data
def save_processed_data(filepath, train_X, train_Y, val_X, val_Y):
    np.savez(filepath, train_X=train_X, train_Y=train_Y, val_X=val_X, val_Y=val_Y)

# Try to load processed data
loaded_data = load_processed_data(processed_data_path)

if loaded_data is not None:
    train_X, train_Y, val_X, val_Y = loaded_data
    print("Loaded processed data from disk.")
else:
    # Your data processing code goes here (loading images, creating sequences, etc.)
    # After processing, save the data for future use
    X, Y = create_shifted_frames(dataset, seq_length)

    # Shuffle the sequences (both X and Y in unison to maintain alignment)
    indexes = np.arange(X.shape[0])
   
    X = X[indexes]
    Y = Y[indexes]

    # Split into train and validation sets
    train_index = int(0.9 * X.shape[0])
    train_X, train_Y = X[:train_index], Y[:train_index]
    val_X, val_Y = X[train_index:], Y[train_index:]

    save_processed_data(processed_data_path, train_X, train_Y, val_X, val_Y)
    print("Processed and saved data to disk.")




# Function to visualize a random sequence from a dataset
def visualize_random_sequence_from_dataset(dataset, seq_length):
    # Check if the dataset is empty
    if dataset.shape[0] < 1:
        print("Dataset is empty.")
        return
    
    # Select a random index for the sequence
    seq_index = random.randint(0, dataset.shape[0] - 1)
    
    # Extract the sequence
    sequence = dataset[seq_index]
    
    # Assuming the sequence shape is (seq_length, height, width, channels) and channels=1 for grayscale
    fig, axes = plt.subplots(3, 5, figsize=(10, 6))  # Adjust subplot grid if seq_length differs
    
    # Plot each of the sequential images
    for idx, ax in enumerate(axes.flat):
        # Squeeze is used to remove single-dimensional entries from the shape of an array.
        ax.imshow(sequence[idx].squeeze(), cmap="gray")
        ax.set_title(f"Frame {idx + 1}")
        ax.axis("off")
    
    # Display the figure
    plt.tight_layout()
    plt.show()

# To visualize a sequence from your training dataset, call the function like this:
visualize_random_sequence_from_dataset(train_X, 15)
visualize_random_sequence_from_dataset(val_X, 15)

print((None, *train_X.shape[2:]))
# Construct the input layer with no definite frame size.
inp = layers.Input(shape=(None, *train_X.shape[2:]))
filters = 16

x = layers.ConvLSTM2D(
    filters=filters,
    kernel_size=(5, 5),
    padding="same",
    return_sequences=True,
    activation="relu",
)(inp)
x = layers.BatchNormalization()(x)
x = layers.ConvLSTM2D(
    filters=filters,
    kernel_size=(3, 3),
    padding="same",
    return_sequences=True,
    activation="relu",
)(x)
x = layers.BatchNormalization()(x)
x = layers.ConvLSTM2D(
    filters=filters,
    kernel_size=(1, 1),
    padding="same",
    return_sequences=True,
    activation="relu",
)(x)
x = layers.Conv3D(
    filters=1, kernel_size=(3, 3, 3), activation="sigmoid", padding="same"
)(x)

# Next, we will build the complete model and compile it.
model = keras.models.Model(inp, x)
model.compile(
    loss=keras.losses.binary_crossentropy,
    optimizer=keras.optimizers.Adam(),
)

# Define some callbacks to improve training.
early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=5)

# Define modifiable training hyperparameters.
epochs = 10
batch_size = 30





# Create data generators+

train_generator = DataGenerator(train_X, train_Y, batch_size=batch_size)
val_generator = DataGenerator(val_X, val_Y, batch_size=batch_size)

for x_batch, y_batch in train_generator:
    print(x_batch.shape, y_batch.dtype)
    print(y_batch.shape, y_batch.dtype)
    break  # To only check the first batch

# Display model summary to understand model architecture and parameter count
model.summary()

# Fit the model to the training data
model.fit(
    train_generator,
    #steps_per_epoch=180,
    epochs=epochs,
    validation_data=val_generator,
    callbacks=[early_stopping, reduce_lr]
)

model.save('location.keras')