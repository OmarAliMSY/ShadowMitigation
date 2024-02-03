from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, LSTM, TimeDistributed, Reshape
from processors.DataPreprocessor import DataPreprocessor




folder_path = r'C:\Users\o.abdulmalik\Documents\Shadow-Mitigation\SKIPPD\05'  # Update this path to your images directory
data_preprocessor = DataPreprocessor(folder_path)
X_train, X_val, Y_train, Y_val = data_preprocessor.get_data()

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
model.fit(X_train, Y_train, epochs=10, batch_size=32)

# Replace `X_train` and `Y_train` with your training data and labels respectively
