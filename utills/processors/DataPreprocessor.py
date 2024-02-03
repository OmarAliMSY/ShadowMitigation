import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split

class DataPreprocessor:
    """
    Class for preprocessing image data: loading, creating sequences, and splitting into training/validation sets.
    """
    
    def __init__(self, folder_path, sequence_length=15):
        self.folder_path = folder_path
        self.sequence_length = sequence_length
        self.input_sequences = None
        self.output_sequences = None
        self.X_train = None
        self.X_val = None
        self.Y_train = None
        self.Y_val = None
        self._create_sequences()
        self._split_data()

    def _create_sequences(self):
        images = sorted([os.path.join(self.folder_path, fname) for fname in os.listdir(self.folder_path) if fname.endswith('.jpg') or fname.endswith('.png')])
        input_sequences = []
        output_sequences = []
        for i in range(len(images) - self.sequence_length * 2 + 1):
            input_seq = images[i:i+self.sequence_length]
            output_seq = images[i+self.sequence_length:i+self.sequence_length*2]
            input_sequences.append(np.array([img_to_array(load_img(img, color_mode='rgb')) / 255.0 for img in input_seq]))
            output_sequences.append(np.array([img_to_array(load_img(img, color_mode='rgb')) / 255.0 for img in output_seq]))
        self.input_sequences = np.array(input_sequences)
        self.output_sequences = np.array(output_sequences)

    def _split_data(self):
        self.X_train, self.X_val, self.Y_train, self.Y_val = train_test_split(self.input_sequences, self.output_sequences, test_size=0.2, random_state=42)

    def get_data(self):
        return self.X_train, self.X_val, self.Y_train, self.Y_val

