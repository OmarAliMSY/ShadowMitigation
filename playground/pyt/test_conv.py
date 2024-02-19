

import io
import imageio

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# The input video frames are grayscale, thus single channel
model = Seq2Seq(num_channels=1, num_kernels=64, 
                kernel_size=(3, 3), padding=(1, 1), activation="relu", 
                frame_size=(64, 64), num_layers=3).to(device)

# optim = Adam(model.parameters(), lr=1e-4)

# # Binary Cross Entropy, target pixel values either 0 or 1
# criterion = nn.BCELoss(reduction='sum')