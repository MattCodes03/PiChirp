import torch.nn as nn
import torch.nn.functional as F


class ConvLSTMModel(nn.Module):
    def __init__(self):
        super(ConvLSTMModel, self).__init__()
        
        # Layer 1: 1D Convolution
        self.conv1 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.dropout1 = nn.Dropout(0.3) 
        
        # Layer 2: 1D Convolution
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=16, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm1d(16)
        
        # Layer 3: 1D Convolution
        self.conv3 = nn.Conv1d(in_channels=16, out_channels=8, kernel_size=5, stride=2, padding=2)
        self.bn3 = nn.BatchNorm1d(8)
        self.dropout3 = nn.Dropout(0.3)

        # Flatten the output of the last convolutional layer
        self.flatten = nn.Flatten()

         # LSTM Layer 1: Classify features
        # Input size will be the flattened output of the conv layers
        self.lstm1 = nn.LSTM(input_size=320, hidden_size=64, num_layers=1, batch_first=True, bidirectional=True)
        self.dropout_lstm1 = nn.Dropout(0.3)  # Dropout after the first LSTM
        
        # LSTM Layer 2: Map to a 4D vector
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=5, num_layers=1, batch_first=True, bidirectional=True)

        # Dense Layer: Maps the LSTM output to the 5 classes, input size of 10 as LSTM 2 is bidirectional (logit can then be softmaxed to determine probability of each class)
        self.fc = nn.Linear(10, 5)

    def forward(self, x):
        x = x.squeeze(1)
        
        # Forward pass through the convolutional layers with batch normalization and ReLU
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout1(x)  # Apply dropout after the first convolution
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout3(x)  # Apply dropout after the third convolution
        
        # Flatten the output for LSTM input
        x = self.flatten(x)
        
        # Prepare for LSTM layers (Reshape for batch_first=True)
        x = x.unsqueeze(1)  # Adding a dummy sequence dimension
        
        # Pass through LSTM layers
        x, (hn, cn) = self.lstm1(x)  # Output from LSTM Layer 1
        x = self.dropout_lstm1(x)  # Apply dropout after the first LSTM layer
        
        x, (hn, cn) = self.lstm2(x)  # Output from LSTM Layer 2

        # Output from the last timestep of LSTM 2
        x = x[:, -1, :]
        x = self.fc(x)
        
        return x
