import torch
import torch.nn as nn

class ImprovedRNNModel(nn.Module):
    """
    Improved RNN model with dropout, batch normalization, and bidirectional capabilities
    to reduce overfitting and improve performance.
    """
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, dropout_rate=0.3, bidirectional=True):
        super(ImprovedRNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout_rate = dropout_rate
        
        # Input dropout
        self.input_dropout = nn.Dropout(dropout_rate)
        
        # RNN layer
        self.rnn = nn.GRU(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout_rate if num_layers > 1 else 0
        )
        
        # Batch normalization for the RNN output
        self.layer_norm = nn.LayerNorm(hidden_size * 2 if bidirectional else hidden_size)
        
        # Fully connected layers with dropout
        fc_size = hidden_size * 2 if bidirectional else hidden_size
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(fc_size, fc_size // 2)
        self.bn1 = nn.BatchNorm1d(fc_size // 2)
        self.relu = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(fc_size // 2, output_size)

    def forward(self, x):
        # Apply input dropout
        x = self.input_dropout(x)
        
        # Pass through RNN layer
        x, _ = self.rnn(x)
        
        # Get the last time step (use all the final outputs for bidirectional)
        if self.bidirectional:
            # For bidirectional, concatenate the last outputs from both directions
            x = x[:, -1, :]
        else:
            x = x[:, -1, :]
        
        # Apply layer normalization
        x = self.layer_norm(x)
        
        # Pass through fully connected layers with dropout
        x = self.dropout1(x)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        
        return x


class LSTMModel(nn.Module):
    """
    LSTM-based model with dropout and batch normalization for ASL recognition.
    """
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, dropout_rate=0.3, bidirectional=True):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout_rate = dropout_rate
        
        # Input dropout
        self.input_dropout = nn.Dropout(dropout_rate)
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout_rate if num_layers > 1 else 0
        )
        
        # Attention layer
        self.attention = nn.Linear(hidden_size * 2 if bidirectional else hidden_size, 1)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size * 2 if bidirectional else hidden_size)
        
        # Fully connected layers with dropout
        fc_size = hidden_size * 2 if bidirectional else hidden_size
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(fc_size, fc_size // 2)
        self.bn1 = nn.BatchNorm1d(fc_size // 2)
        self.relu = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(fc_size // 2, output_size)

    def forward(self, x):
        # Apply input dropout
        x = self.input_dropout(x)
        
        # Pass through LSTM layer
        outputs, _ = self.lstm(x)  # output shape: [batch_size, seq_len, hidden_size * num_directions]
        
        # Apply attention mechanism
        attention_weights = torch.softmax(self.attention(outputs), dim=1)
        context = torch.sum(attention_weights * outputs, dim=1)
        
        # Apply layer normalization
        x = self.layer_norm(context)
        
        # Pass through fully connected layers with dropout
        x = self.dropout1(x)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        
        return x


class TransformerModel(nn.Module):
    """
    Transformer-based model for ASL recognition with self-attention mechanism.
    """
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, dropout_rate=0.3, nhead=4):
        super(TransformerModel, self).__init__()
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        
        # Input embedding
        self.input_projection = nn.Linear(input_size, hidden_size)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_size, dropout_rate)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, 
            nhead=nhead, 
            dim_feedforward=hidden_size*2, 
            dropout=dropout_rate,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        # Fully connected layers
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.bn1 = nn.BatchNorm1d(hidden_size // 2)
        self.relu = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_size // 2, output_size)

    def forward(self, x):
        # Project input to hidden size
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Pass through transformer encoder
        x = self.transformer_encoder(x)
        
        # Use the representation of the [CLS] token (first token)
        # or average over the sequence dimension
        x = torch.mean(x, dim=1)  # [batch_size, hidden_size]
        
        # Apply layer normalization
        x = self.layer_norm(x)
        
        # Pass through fully connected layers with dropout
        x = self.dropout1(x)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        
        return x


class PositionalEncoding(nn.Module):
    """
    Positional encoding for Transformer model.
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Add positional encoding to the input
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
