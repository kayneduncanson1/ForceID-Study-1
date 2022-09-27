import torch
import torch.nn as nn
import math


class FCNN(nn.Module):

    def __init__(self, in_features, fc1_out, fc2_out):

        super(FCNN, self).__init__()

        self.in_features = in_features
        self.fc1_out = fc1_out
        self.fc2_out = fc2_out

        self.fc1 = nn.Sequential(nn.Linear(self.in_features, self.fc1_out), nn.BatchNorm1d(num_features=self.fc1_out),
                                 nn.ELU())
        self.fc2 = nn.Linear(self.fc1_out, self.fc2_out)

    def forward(self, inputs):

        # Flatten input into feature vector:
        out = inputs.reshape(inputs.size(0), -1)

        out = self.fc1(out)
        out = self.fc2(out)

        return out


class CNN(nn.Module):

    def __init__(self, nc0, nc1, nc2, nc3, nc4, fc1_out, fc2_out):

        super(CNN, self).__init__()

        self.nc0 = nc0
        self.nc1 = nc1
        self.nc2 = nc2
        self.nc3 = nc3
        self.nc4 = nc4

        self.fc1_out = fc1_out
        self.fc2_out = fc2_out

        # Set up to go from len 600 to len 25 over four conv layers via local average pooling:
        self.conv1 = nn.Sequential(nn.Conv1d(self.nc0, self.nc1, kernel_size=3, padding=1),
                                   nn.BatchNorm1d(num_features=self.nc1), nn.ELU(),
                                   nn.AvgPool1d(kernel_size=2, stride=2, padding=0))
        self.conv2 = nn.Sequential(nn.Conv1d(self.nc1, self.nc2, kernel_size=3, padding=1),
                                   nn.BatchNorm1d(num_features=self.nc2), nn.ELU(),
                                   nn.AvgPool1d(kernel_size=2, stride=2, padding=0))
        self.conv3 = nn.Sequential(nn.Conv1d(self.nc2, self.nc3, kernel_size=3, padding=1),
                                   nn.BatchNorm1d(num_features=self.nc3), nn.ELU(),
                                   nn.AvgPool1d(kernel_size=2, stride=2, padding=0))
        self.conv4 = nn.Sequential(nn.Conv1d(self.nc3, self.nc4, kernel_size=3, padding=1),
                                   nn.BatchNorm1d(num_features=self.nc4), nn.ELU(),
                                   nn.AvgPool1d(kernel_size=3, stride=3, padding=0))

        # 25 is the len and self.nc4 is the no. of channels after conv4:
        self.fc1 = nn.Sequential(nn.Linear(25 * self.nc4, self.fc1_out), nn.BatchNorm1d(num_features=self.fc1_out),
                                 nn.ELU())
        self.fc2 = nn.Linear(self.fc1_out, self.fc2_out)

    def forward(self, inputs):

        out = self.conv1(inputs)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)

        # Flatten to feature vector for fully-connected layers:
        out = out.reshape(inputs.size(0), -1)

        out = self.fc1(out)
        out = self.fc2(out)

        return out


class CLSTMNN(nn.Module):

    def __init__(self, nc0, nc1, nc2, nc3, nc4, lstm_layers, fc1_out, fc2_out):

        super(CLSTMNN, self).__init__()

        self.nc0 = nc0
        self.nc1 = nc1
        self.nc2 = nc2
        self.nc3 = nc3
        self.nc4 = nc4

        self.lstm_layers = lstm_layers

        self.fc1_out = fc1_out
        self.fc2_out = fc2_out

        # Set up to go from len 600 to len 25 over four conv layers via local average pooling:
        self.conv1 = nn.Sequential(nn.Conv1d(self.nc0, self.nc1, kernel_size=3, padding=1),
                                   nn.BatchNorm1d(num_features=self.nc1), nn.ELU(),
                                   nn.AvgPool1d(kernel_size=2, stride=2, padding=0))
        self.conv2 = nn.Sequential(nn.Conv1d(self.nc1, self.nc2, kernel_size=3, padding=1),
                                   nn.BatchNorm1d(num_features=self.nc2), nn.ELU(),
                                   nn.AvgPool1d(kernel_size=2, stride=2, padding=0))
        self.conv3 = nn.Sequential(nn.Conv1d(self.nc2, self.nc3, kernel_size=3, padding=1),
                                   nn.BatchNorm1d(num_features=self.nc3), nn.ELU(),
                                   nn.AvgPool1d(kernel_size=2, stride=2, padding=0))
        self.conv4 = nn.Sequential(nn.Conv1d(self.nc3, self.nc4, kernel_size=3, padding=1),
                                   nn.BatchNorm1d(num_features=self.nc4), nn.ELU(),
                                   nn.AvgPool1d(kernel_size=3, stride=3, padding=0))

        self.lstm = nn.LSTM(self.nc4, self.nc4, self.lstm_layers, bidirectional=False)

        # 25 is the len and self.nc4 is the no. of channels after lstm:
        self.fc1 = nn.Sequential(nn.Linear(25 * self.nc4, self.fc1_out), nn.BatchNorm1d(num_features=self.fc1_out),
                                 nn.ELU())
        self.fc2 = nn.Linear(self.fc1_out, self.fc2_out)

    def forward(self, inputs):

        out = self.conv1(inputs)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)

        # Change from shape (N, C, L) to (L, N, C) for LSTM layer:
        out = out.view(-1, inputs.size(0), self.nc4)

        h0, c0 = self.init_hidden(inputs)
        out, (hn, cn) = self.lstm(out, (h0, c0))

        # Flatten to feature vector for fully-connected layers:
        out = out.reshape(inputs.size(0), -1)

        out = self.fc1(out)
        out = self.fc2(out)

        return out

    def init_hidden(self, inputs):

        h0 = torch.zeros(self.lstm_layers, inputs.size(0), self.nc4)
        c0 = torch.zeros(self.lstm_layers, inputs.size(0), self.nc4)

        return [t.cuda() for t in (h0, c0)]


class CBiLSTMNN(nn.Module):

    def __init__(self, nc0, nc1, nc2, nc3, nc4, lstm_layers, fc1_out, fc2_out):

        super(CBiLSTMNN, self).__init__()

        self.nc0 = nc0
        self.nc1 = nc1
        self.nc2 = nc2
        self.nc3 = nc3
        self.nc4 = nc4

        self.lstm_layers = lstm_layers

        self.fc1_out = fc1_out
        self.fc2_out = fc2_out

        # Set up to go from len 600 to len 25 over four conv layers via local average pooling:
        self.conv1 = nn.Sequential(nn.Conv1d(self.nc0, self.nc1, kernel_size=3, padding=1),
                                   nn.BatchNorm1d(num_features=self.nc1), nn.ELU(),
                                   nn.AvgPool1d(kernel_size=2, stride=2, padding=0))
        self.conv2 = nn.Sequential(nn.Conv1d(self.nc1, self.nc2, kernel_size=3, padding=1),
                                   nn.BatchNorm1d(num_features=self.nc2), nn.ELU(),
                                   nn.AvgPool1d(kernel_size=2, stride=2, padding=0))
        self.conv3 = nn.Sequential(nn.Conv1d(self.nc2, self.nc3, kernel_size=3, padding=1),
                                   nn.BatchNorm1d(num_features=self.nc3), nn.ELU(),
                                   nn.AvgPool1d(kernel_size=2, stride=2, padding=0))
        self.conv4 = nn.Sequential(nn.Conv1d(self.nc3, self.nc4, kernel_size=3, padding=1),
                                   nn.BatchNorm1d(num_features=self.nc4), nn.ELU(),
                                   nn.AvgPool1d(kernel_size=3, stride=3, padding=0))

        self.lstm = nn.LSTM(self.nc4, self.nc4, self.lstm_layers, bidirectional=True)

        # 25 is the len and self.nc4 is the no. of channels after lstm (x2 because the LSTM is bi-directional):
        self.fc1 = nn.Sequential(nn.Linear(25 * self.nc4 * 2, self.fc1_out), nn.BatchNorm1d(num_features=self.fc1_out),
                                 nn.ELU())
        self.fc2 = nn.Linear(self.fc1_out, self.fc2_out)

    def forward(self, inputs):

        out = self.conv1(inputs)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)

        # Change from shape (N, C, L) to (L, N, C) for LSTM layer:
        out = out.view(-1, inputs.size(0), self.nc4)

        h0, c0 = self.init_hidden(inputs)
        out, (hn, cn) = self.lstm(out, (h0, c0))

        # Flatten to feature vector for fully-connected layers:
        out = out.reshape(inputs.size(0), -1)

        out = self.fc1(out)
        out = self.fc2(out)

        return out

    def init_hidden(self, inputs):

        h0 = torch.zeros(2 * self.lstm_layers, inputs.size(0), self.nc4)
        c0 = torch.zeros(2 * self.lstm_layers, inputs.size(0), self.nc4)

        return [t.cuda() for t in (h0, c0)]


# Positional encoding for CTNN:
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=600):

        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):

        return x + self.pe[:x.size(0), :]


class CTNN(nn.Module):

    def __init__(self, nc0, nc1, nc2, nc3, nc4, tf_layers, fc1_out, fc2_out):

        super(CTNN, self).__init__()

        self.nc0 = nc0
        self.nc1 = nc1
        self.nc2 = nc2
        self.nc3 = nc3
        self.nc4 = nc4

        self.tf_layers = tf_layers

        self.fc1_out = fc1_out
        self.fc2_out = fc2_out

        # Set up to go from len 600 to len 25 over four conv layers via local average pooling:
        self.conv1 = nn.Sequential(nn.Conv1d(self.nc0, self.nc1, kernel_size=3, padding=1),
                                   nn.BatchNorm1d(num_features=self.nc1), nn.ELU(),
                                   nn.AvgPool1d(kernel_size=2, stride=2, padding=0))
        self.conv2 = nn.Sequential(nn.Conv1d(self.nc1, self.nc2, kernel_size=3, padding=1),
                                   nn.BatchNorm1d(num_features=self.nc2), nn.ELU(),
                                   nn.AvgPool1d(kernel_size=2, stride=2, padding=0))
        self.conv3 = nn.Sequential(nn.Conv1d(self.nc2, self.nc3, kernel_size=3, padding=1),
                                   nn.BatchNorm1d(num_features=self.nc3), nn.ELU(),
                                   nn.AvgPool1d(kernel_size=2, stride=2, padding=0))
        self.conv4 = nn.Sequential(nn.Conv1d(self.nc3, self.nc4, kernel_size=3, padding=1),
                                   nn.BatchNorm1d(num_features=self.nc4), nn.ELU(),
                                   nn.AvgPool1d(kernel_size=3, stride=3, padding=0))

        self.pos_encoder = PositionalEncoding(self.nc4)
        # 25 is the len and self.nc4 is the no. of channels after conv4:
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.nc4, nhead=4, activation='gelu',
                                                        dim_feedforward=25 * self.nc4)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=self.tf_layers)

        self.fc1 = nn.Linear(25 * self.nc4, self.fc1_out)
        self.bn_elu1 = nn.Sequential(nn.BatchNorm1d(num_features=self.fc1_out), nn.ELU())
        self.fc2 = nn.Linear(self.fc1_out, self.fc2_out)

    def forward(self, inputs):

        out = self.conv1(inputs)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)

        # Change from shape (N, C, L) to (L, N, C) for transformer encoder layer:
        out = out.view(-1, inputs.size(0), self.nc4)

        out = self.pos_encoder(out)
        out = self.transformer_encoder(out)

        # Flatten to feature vector for fully-connected layers:
        out = out.reshape(inputs.size(0), -1)

        out = self.fc1(out)
        out = self.bn_elu1(out)
        out = self.fc2(out)

        return out
