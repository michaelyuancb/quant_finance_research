from torch import nn


class Base_DNN(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.):
        super(Base_DNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.drop = nn.Dropout(dropout_rate)
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid(),
            nn.BatchNorm1d(num_features=hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
            nn.BatchNorm1d(num_features=hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        x = self.drop(x)
        x = self.fc(x)
        return x