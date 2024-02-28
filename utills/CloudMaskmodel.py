import numpy as np
import matplotlib.pyplot as plt
from keras import layers, models
from PIL import Image
import keras



class CloudMaskModel:
    def __init__(self, dataset_path, foldername="CloudMask", legal_extensions=["jpg", "jpeg"], model_path=None):
        self.dataset_path = dataset_path
        self.foldername = foldername
        self.legal_extensions = legal_extensions
        self.model_path = model_path
        self.model = None

    def load_image(self, infilename, dtype="uint8"):
        img = Image.open(infilename).convert('L')
        img.load()
        data = np.asarray(img, dtype=dtype)
        threshold_value = 128
        data = (data >= threshold_value).astype(dtype) 
        #data *= 255
        data = np.expand_dims(data, axis=-1)
        return data

    def load_dataset(self):
        from FileHandler import FileHandler  # Assuming FileHandler is implemented elsewhere
        fp = FileHandler(dataset_path=self.dataset_path, foldername=self.foldername, legal_extensions=self.legal_extensions, include_subdirectories=False)
        folder_paths = fp.path_file()
        dataset = np.array([self.load_image(img) for img in folder_paths])
        dataset = dataset[:20000]
        return dataset

    def prepare_dataset(self, dataset):
        indexes = np.arange(dataset.shape[0])
        train_index = indexes[: int(0.9 * dataset.shape[0])]
        val_index = indexes[int(0.9 * dataset.shape[0]) :]
        train_dataset = dataset[train_index]
        val_dataset = dataset[val_index]
        x_train, y_train = self.create_shifted_frames(train_dataset)
        x_val, y_val = self.create_shifted_frames(val_dataset)
        return x_train, y_train, x_val, y_val
    

    def get_datasets(self,dataset):
        indexes = np.arange(dataset.shape[0])
        train_index = indexes[: int(0.9 * dataset.shape[0])]
        val_index = indexes[int(0.9 * dataset.shape[0]) :]
        train_dataset = dataset[train_index]
        val_dataset = dataset[val_index]
        return train_dataset,val_dataset

    def create_shifted_frames(self, data, window_size=15, threshold=1500):
        x, y = [], []
        for i in range(len(data) - window_size):
            # Original sequences of 15 frames
            sequence_x = data[i:i + window_size]
            sequence_y = data[i + 1:i + window_size + 1]

            # Check if the sequence meets the threshold criteria
            sequence_sum = np.sum(sequence_x)
            if threshold < sequence_sum <= 25000:
                print(sequence_sum)
                # Reduce each sequence from 15 to 5 frames by calculating the mean of every 3 frames
                reduced_sequence_x = sequence_x
                reduced_sequence_y = sequence_y

                x.append(reduced_sequence_x)
                y.append(reduced_sequence_y)

        return np.array(x), np.array(y)

    def reduce_sequence(self, sequence):
        # Assuming sequence shape is (15, height, width, channels)
        reduced_sequence = []
        for i in range(0, sequence.shape[0], 3):
            # Take the mean of every 3 frames, resulting in a new frame
            mean_frame = np.mean(sequence[i:i + 3], axis=0)
            reduced_sequence.append(mean_frame)
        return np.array(reduced_sequence)

    def define_model(self, input_shape):
        print(input_shape)
        inp = layers.Input(shape=input_shape)
        x = layers.ConvLSTM2D(filters=64, kernel_size=(5, 5), padding="same", return_sequences=True, activation="relu")(inp)
        x = layers.BatchNormalization()(x)
        x = layers.ConvLSTM2D(filters=64, kernel_size=(3, 3), padding="same", return_sequences=True, activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.ConvLSTM2D(filters=64, kernel_size=(1, 1), padding="same", return_sequences=True, activation="relu")(x)
        x = layers.Conv3D(filters=1, kernel_size=(3, 3, 3), activation="sigmoid", padding="same")(x)
        self.model = models.Model(inp, x)
        self.model.compile(loss='binary_crossentropy', optimizer='adam')

    def train_model(self, x_train, y_train, x_val, y_val, epochs=10, batch_size=5):
        early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10)
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=5)
        self.model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_val, y_val), callbacks=[early_stopping, reduce_lr])

    def save_model(self, model_path):
        self.model.save(model_path)

    def load_model(self, model_path):
        self.model = models.load_model(model_path)

    def predict_and_visualize(self, val_dataset, num_predictions=30):
        for i in range(num_predictions):
            example = val_dataset[100:]
            seq = 15  # Sequence length for predictions

            # Assuming `example` is shaped correctly as mentioned in your error message, we directly use it
            # Initial frames for generating predictions, ensuring the shape aligns with model expectations
            frames = example[:(i+2)*seq, ...]                       # Initial 15 frames for generating predictions
            original_frames = example[(i+2)*seq:(i+3)*seq, ...]     # Next 15 frames, the "true" continuation of the sequence
            print("Shape before prediction:", np.expand_dims(frames[-15:], axis=0).shape)
            # This adjustment depends on your specific use case. You might need to adapt it to fit your scenario
            print("Shape before prediction:", frames.shape)

            # Predict a new set of frames based on the most recent ones
            for _ in range(seq):
                new_prediction = self.model.predict(np.expand_dims(frames[-15:], axis=0))
                new_prediction = np.squeeze(new_prediction, axis=0)                 # Remove batch dimension
                predicted_frame = np.expand_dims(new_prediction[-1, ...], axis=0)   # Latest predicted frame
                frames = np.concatenate((frames, predicted_frame), axis=0)          # Extend prediction frames

            
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


# Example usage
cm_model = CloudMaskModel(dataset_path=r"SKIPPD")
dataset = cm_model.load_dataset()
x_train, y_train, x_val, y_val = cm_model.prepare_dataset(dataset)
cm_model.define_model(input_shape=(None, 64, 64, 1))  # Assuming all images are 64x64 pixels
print(y_val.shape)
cm_model.train_model(x_train, y_train, x_val, y_val)
#cm_model.load_model(model_path="240216_2.keras")
cm_model.save_model(model_path="240217_3.keras")
#cm_model.load_model('NewCM_.keras')
_,val_dataset = cm_model.get_datasets(dataset=dataset)
cm_model.predict_and_visualize(num_predictions=30,val_dataset=val_dataset)
