from keras.utils import Sequence
import numpy as np
import tensorflow as tf

class DataGenerator(Sequence):
    def __init__(self, X_data, Y_data, batch_size):
        self.X_data, self.Y_data = X_data, Y_data
        self.batch_size = batch_size
        
    def __len__(self):
        return np.ceil(len(self.X_data) / self.batch_size).astype(int)
    
    def __getitem__(self, idx):
        batch_x = self.X_data[idx * self.batch_size:(idx + 1) * self.batch_size].astype('float16')
        batch_y = self.Y_data[idx * self.batch_size:(idx + 1) * self.batch_size]
        return tf.convert_to_tensor(batch_x), tf.convert_to_tensor(batch_y)

