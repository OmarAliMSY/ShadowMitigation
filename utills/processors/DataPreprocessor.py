from keras.utils import Sequence
import numpy as np
import os
from PIL import Image
from sklearn.model_selection import train_test_split

def load_image(infilename):
    infilename = str(infilename) 
    img = Image.open(infilename)
    img.load()
    
    data = np.asarray(img, dtype="float16")
    data/=255
    return data

class DataGenerator(Sequence):
    """
    Keras Sequence based generator to load and preprocess image sequences on-the-fly.
    """

    def __init__(self, image_paths, sequence_length=15, batch_size=32):
        self.image_paths = image_paths
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.indexes = np.arange(len(self.image_paths) - self.sequence_length * 2 + 1)
        

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor((len(self.image_paths) - self.sequence_length * 2 + 1) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [k for k in indexes]
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.image_paths) - self.sequence_length * 2 + 1)
        
    def __data_generation(self, list_IDs_temp):
     X = []
     y = []

     for i in list_IDs_temp:
         # Input sequence
         input_seq = self.image_paths[i:i+self.sequence_length]
         # Output sequence
         output_seq = self.image_paths[i+self.sequence_length:i+self.sequence_length*2]

         # Stack the input sequence and add a channel dimension
         X.append(np.stack([load_image(img)[..., np.newaxis] for img in input_seq], axis=0))
         # Stack the output sequence and add a channel dimension
         y.append(np.stack([load_image(img)[..., np.newaxis] for img in output_seq], axis=0))

     return np.array(X), np.array(y)
