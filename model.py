import torch
import torch.nn as nn


class DeepSenseModel(nn.Module):
    GRU_INPUT = 64  # Increased feature size after reduction

    def __init__(self, input_dims=144, num_axis=3, num_classes=6):
        super(DeepSenseModel, self).__init__()

        # Convolutional layers for accelerometer
        self.accel_conv1 = nn.Conv1d(num_axis, 16, kernel_size=3, stride=1, padding=1)
        self.accel_bn1 = nn.BatchNorm1d(16)
        self.accel_conv2 = nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1)
        self.accel_bn2 = nn.BatchNorm1d(32)
        self.accel_pool = nn.MaxPool1d(2)

        # Convolutional layers for gyroscope
        self.gyro_conv1 = nn.Conv1d(num_axis, 16, kernel_size=3, stride=1, padding=1)
        self.gyro_bn1 = nn.BatchNorm1d(16)
        self.gyro_conv2 = nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1)
        self.gyro_bn2 = nn.BatchNorm1d(32)
        self.gyro_pool = nn.MaxPool1d(2)

        # Merge convolutional layers
        self.merge_conv = nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1)
        self.merge_bn = nn.BatchNorm1d(64)
        self.merge_pool = nn.MaxPool1d(2)

        # Feature reduction
        self.feature_reduce = nn.Linear(64, self.GRU_INPUT)

        # GRU layer
        self.gru = nn.GRU(self.GRU_INPUT, 32, num_layers=3, batch_first=True)

        # Fully connected layer
        self.fc = nn.Linear(32, num_classes)

        # Dropout layers
        self.dropout_conv = nn.Dropout(p=0.5)
        self.dropout_gru = nn.Dropout(p=0.5)

    def forward(self, accel, gyro):
        # Accelerometer convolution
        accel = torch.relu(self.accel_bn1(self.accel_conv1(accel)))
        accel = self.dropout_conv(accel)
        accel = torch.relu(self.accel_bn2(self.accel_conv2(accel)))
        accel = self.accel_pool(accel)

        # Gyroscope convolution
        gyro = torch.relu(self.gyro_bn1(self.gyro_conv1(gyro)))
        gyro = self.dropout_conv(gyro)
        gyro = torch.relu(self.gyro_bn2(self.gyro_conv2(gyro)))
        gyro = self.gyro_pool(gyro)

        # Merge features
        x = torch.cat((accel, gyro), dim=1)  # Concatenate along channel dimension
        x = torch.relu(self.merge_bn(self.merge_conv(x)))
        x = self.merge_pool(x)

        # Flatten and feature reduction
        x = x.permute(0, 2, 1)  # Shape: (batch, seq_len, channels)
        x = self.feature_reduce(x)

        # GRU
        x, _ = self.gru(x)
        x = x[:, -1, :]  # Use the last time step

        # Fully connected layer
        x = self.dropout_gru(x)
        x = self.fc(x)
        return x
