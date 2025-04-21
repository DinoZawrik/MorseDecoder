import torch
import torch.nn as nn
import torch.nn.functional as F

class CRNNModel_4Layer(nn.Module):
    def __init__(self, n_features, num_classes, rnn_hidden_size=256,
                 num_rnn_layers=2, cnn_dropout=0.15, rnn_dropout=0.15):
        super().__init__()
        self.n_features = n_features
        self.num_classes = num_classes
        self.rnn_hidden_size = rnn_hidden_size
        self.time_reduction_factor = 2 * 2 * 2 * 2

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Dropout2d(p=cnn_dropout),

            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)), 
            nn.Dropout2d(p=cnn_dropout),

            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Dropout2d(p=cnn_dropout), 

            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
        )

        with torch.no_grad():
            dummy_input = torch.randn(1, 1, self.n_features, 512)
            cnn_out = self.cnn(dummy_input)
            self.cnn_output_features = cnn_out.shape[1] * cnn_out.shape[2] # C * F'
            print(f"Модель {self.__class__.__name__}: Вход в RNN = {self.cnn_output_features} признаков (C={cnn_out.shape[1]}, F'={cnn_out.shape[2]})")
            print(f"Модель {self.__class__.__name__}: Фактор сокращения времени CNN = {self.time_reduction_factor}x")


        actual_rnn_dropout = rnn_dropout if num_rnn_layers > 1 else 0.0
        self.rnn = nn.LSTM(
            input_size=self.cnn_output_features, hidden_size=self.rnn_hidden_size,
            num_layers=num_rnn_layers, bidirectional=True, batch_first=False,
            dropout=actual_rnn_dropout
        )

        self.layer_norm = nn.LayerNorm(2 * self.rnn_hidden_size)
        self.rnn_dropout_layer = nn.Dropout(rnn_dropout)
        self.fc = nn.Linear(2 * self.rnn_hidden_size, num_classes)

    def _calculate_output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        output_lengths = torch.div(input_lengths, self.time_reduction_factor, rounding_mode='floor')
        return torch.clamp(output_lengths, min=1).long()

    def forward(self, x: torch.Tensor, input_lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:

        x = x.unsqueeze(1)
        x = self.cnn(x)
        output_lengths = self._calculate_output_lengths(input_lengths.cpu())

        B, C, F_prime, T_prime = x.shape
        x = x.permute(3, 0, 1, 2).contiguous() 
        x = x.view(T_prime, B, C * F_prime)

        x, _ = self.rnn(x)

        x = self.layer_norm(x)
        x = self.rnn_dropout_layer(x)

        x = self.fc(x)
        output_lengths = torch.clamp(output_lengths, max=T_prime)
        if (output_lengths == 0).any():
             print("Warning: Zero output lengths detected after clamping!")
             output_lengths = torch.clamp(output_lengths, min=1)


        # LogSoftmax по измерению классов
        log_probs = F.log_softmax(x, dim=2)

        return log_probs, output_lengths.to(x.device) 