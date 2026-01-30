"""LSTM model for character-level language modeling."""

import torch
import torch.nn as nn
from typing import Tuple


class CharLSTM(nn.Module):
    """Character-level LSTM for text generation.
    
    Architecture:
    - Embedding layer (vocab_size → embed_dim)
    - Multi-layer LSTM
    - Output projection (hidden_dim → vocab_size)
    
    Suitable for federated learning with reasonable parameter count.
    """
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 64,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.2,
        num_classes: int = None,  # Alias for vocab_size (for compatibility)
    ):
        """Initialize LSTM model.
        
        Args:
            vocab_size: Size of character vocabulary
            embed_dim: Embedding dimension
            hidden_dim: LSTM hidden dimension
            num_layers: Number of LSTM layers
            dropout: Dropout probability
            num_classes: Alias for vocab_size (ignored if vocab_size is set)
        """
        super().__init__()
        
        # Use num_classes as fallback for vocab_size
        if vocab_size is None or vocab_size == 0:
            vocab_size = num_classes
        
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Output projection
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with reasonable defaults."""
        # Embedding initialization
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        
        # LSTM initialization
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
                # Set forget gate bias to 1.0 for better gradient flow
                param.data[self.hidden_dim:2*self.hidden_dim].fill_(1.0)
        
        # Output layer
        nn.init.zeros_(self.fc.bias)
        nn.init.uniform_(self.fc.weight, -0.1, 0.1)
    
    def forward(
        self,
        x: torch.Tensor,
        hidden: Tuple[torch.Tensor, torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length)
            hidden: Optional initial hidden state
        
        Returns:
            Output logits of shape (batch_size, sequence_length, vocab_size)
        """
        # Embedding: (batch, seq) -> (batch, seq, embed_dim)
        embedded = self.dropout(self.embedding(x))
        
        # LSTM: (batch, seq, embed_dim) -> (batch, seq, hidden_dim)
        if hidden is None:
            lstm_out, hidden = self.lstm(embedded)
        else:
            lstm_out, hidden = self.lstm(embedded, hidden)
        
        # Dropout
        lstm_out = self.dropout(lstm_out)
        
        # Project to vocabulary: (batch, seq, hidden_dim) -> (batch, seq, vocab_size)
        logits = self.fc(lstm_out)
        
        return logits
    
    def init_hidden(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize hidden state.
        
        Args:
            batch_size: Batch size
            device: Device to create tensors on
        
        Returns:
            Tuple of (h_0, c_0) each of shape (num_layers, batch_size, hidden_dim)
        """
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        return (h_0, c_0)
