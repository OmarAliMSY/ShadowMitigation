import numpy as np
import matplotlib.pyplot as plt
from  FileHandler import FileHandler
import keras
from keras import layers
from PIL import Image

le= ["jpg","jpeg"]
def load_image(infilename):
    img = Image.open(infilename)
    img.load()
    data = np.asarray(img, dtype="float32")
    # Assuming images are grayscale, add a channel dimension
    data = np.expand_dims(data, axis=-1)
    return data



fp = FileHandler(dataset_path=r"SKIPPD",foldername="CloudMask",legal_extensions=le,include_subdirectories=False)
fp = fp.path_file()
folder_paths = fp

dataset = np.array([load_image(img) for img in fp])



dataset = dataset[:10000]
# Add a channel dimension since the images are grayscale.


# Split into train and validation sets using indexing to optimize memory.
indexes = np.arange(dataset.shape[0])
train_index = indexes[: int(0.9 * dataset.shape[0])]
val_index = indexes[int(0.9 * dataset.shape[0]) :]
train_dataset = dataset[train_index]
val_dataset = dataset[val_index]

# Normalize the data to the 0-1 range.
train_dataset = train_dataset / 255
val_dataset = val_dataset / 255


# We'll define a helper function to shift the frames, where
# `x` is frames 0 to n - 1, and `y` is frames 1 to n.
def create_shifted_frames(data, window_size=15,threshold = 500):
    # Assuming data is loaded correctly with each frame of shape (64, 64, 1)
    x, y = [], []
    for i in range(len(data) - window_size):

        sequence_x = data[i:i+window_size]
        sequence_y = data[i+1:i+window_size+1]

        sequence_sum = np.sum(sequence_x)
        if sequence_sum > threshold:
            x.append(sequence_x)
            y.append(sequence_y)

    return np.array(x), np.array(y)



# Apply the processing function to the datasets.
x_train, y_train = create_shifted_frames(train_dataset)
x_val, y_val = create_shifted_frames(val_dataset)

# Inspect the dataset.
print(f"Training Dataset Shapes:  {x_train.shape} {y_train.shape}")
print(f"Validation Dataset Shapes: {x_val.shape} {y_val.shape}")


## Construct a figure on which we will visualize the images.
#fig, axes = plt.subplots(3, 5, figsize=(10, 8))
#
## Plot each of the sequential images for one random data example.
#data_choice = np.random.choice(range(len(x_train)), size=1)[0]
#for idx, ax in enumerate(axes.flat):
#    ax.imshow(np.squeeze(x_train[data_choice][idx]), cmap="gray")
#    ax.set_title(f"Frame {idx + 1}")
#    ax.axis("off")
#
## Print information and display the figure.
#print(f"Displaying frames for example {data_choice}.")
#plt.show()


## Construct the input layer with no definite frame size.
#inp = layers.Input(shape=(None, *x_train.shape[2:]))
#
## We will construct 3 `ConvLSTM2D` layers with batch normalization,
## followed by a `Conv3D` layer for the spatiotemporal outputs.
#x = layers.ConvLSTM2D(
#    filters=64,
#    kernel_size=(5, 5),
#    padding="same",
#    return_sequences=True,
#    activation="relu",
#)(inp)
#x = layers.BatchNormalization()(x)
#x = layers.ConvLSTM2D(
#    filters=64,
#    kernel_size=(3, 3),
#    padding="same",
#    return_sequences=True,
#    activation="relu",
#)(x)
#x = layers.BatchNormalization()(x)
#x = layers.ConvLSTM2D(
#    filters=64,
#    kernel_size=(1, 1),
#    padding="same",
#    return_sequences=True,
#    activation="relu",
#)(x)
#x = layers.Conv3D(
#    filters=1, kernel_size=(3, 3, 3), activation="sigmoid", padding="same"
#)(x)
#
## Next, we will build the complete model and compile it.
#model = keras.models.Model(inp, x)
#model.compile(
#    loss=keras.losses.MeanSquaredError(),
#    optimizer=keras.optimizers.Adam(),
#)
#
#
## Define some callbacks to improve training.
#early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10)
#reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=5)
#
## Define modifiable training hyperparameters.
#epochs = 25
#batch_size = 5
#
## Fit the model to the training data.
#model.fit(
#    x_train,
#    y_train,
#    batch_size=batch_size,
#    epochs=epochs,
#    validation_data=(x_val, y_val),
#    callbacks=[early_stopping, reduce_lr],
#)

model = keras.models.load_model('CM_Model1.keras')
#model = keras.models.load_model('location.keras')


for i in range(30):
    # Assuming `val_dataset` is correctly shaped as (N, 30, 64, 64, 1) where N is the number of samples
    example = val_dataset
    print(example.shape)
    seq = 15
    # Split the example into initial frames for prediction and the rest as original frames for comparison
    frames = example[:(i+2)*seq, ...]  # Initial 15 frames for generating predictions
    original_frames = example[(i+2)*seq:(i+3)*seq, ...]  # Next 15 frames, the "true" continuation of the sequence
    print("Shape before prediction:", np.expand_dims(frames[-15:], axis=0).shape)

    # Predict a new set of 15 frames.
    for _ in range(15):  # Adjusted to 15 for consistency with your requirement
        # Extract the model's prediction and post-process it.
        new_prediction = model.predict(np.expand_dims(frames[-15:], axis=0))  # Use the most recent 15 frames
        new_prediction = np.squeeze(new_prediction, axis=0)
        predicted_frame = np.expand_dims(new_prediction[-1, ...], axis=0)  # The latest predicted frame

        # Extend the set of prediction frames.
        frames = np.concatenate((frames, predicted_frame), axis=0)

    # Visualization adjustments
    fig, axes = plt.subplots(2, 15, figsize=(20, 4))

    # Plot the original frames (frames 16 to 30 of the sequence).
    for idx, ax in enumerate(axes[0]):
        ax.imshow(np.squeeze(original_frames[idx]), cmap="gray")
        ax.set_title(f"Original {idx + 16}")
        ax.axis("off")

    # Plot the new frames (the predicted continuation).
    new_frames = frames[-15:, ...]  # Select the last 15 frames as the new predicted frames
    for idx, ax in enumerate(axes[1]):
        ax.imshow(np.squeeze(new_frames[idx]), cmap="gray")
        ax.set_title(f"Predicted {idx + 16}")
        ax.axis("off")

    plt.tight_layout()
    plt.show()