import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from progressbar import progressbar


def load_image( infilename ) :
    img = Image.open( infilename )
    img.load()
    data = np.asarray( img, dtype="float16" )
    return data


class DataPreprocessor:
    """
    Class for preprocessing image data: loading, creating sequences, and splitting into training/validation sets.
    """
     
    def __init__(self, folder_path, sequence_length=15):
        self.folder_paths = folder_path if isinstance(folder_path, list) else [folder_path]
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
        input_sequences = []
        output_sequences = []

        
        images = self.folder_paths
        for i in progressbar(range(len(images) - self.sequence_length * 2 + 1)):
            input_seq = images[i:i+self.sequence_length]
            output_seq = images[i+self.sequence_length:i+self.sequence_length*2]
            input_sequences.append(np.stack([load_image(img) for img in input_seq]))
            output_sequences.append(np.stack([load_image(img) for img in output_seq]))
        
        
        self.input_sequences = np.array(input_sequences)
        self.output_sequences = np.array(output_sequences)

    def _split_data(self):
        self.X_train, self.X_val, self.Y_train, self.Y_val = train_test_split(self.input_sequences, self.output_sequences, test_size=0.2, random_state=42)

    def get_data(self):
        return self.X_train, self.X_val, self.Y_train, self.Y_val
