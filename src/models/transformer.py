"""Transformer model for character-level language modeling."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer."""
    
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
        
        Returns:
            Output with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class CharTransformer(nn.Module):
    """Lightweight Transformer decoder for character-level language modeling.
    
    A micro-GPT style architecture suitable for federated learning:
    - Character-level embedding + positional encoding
    - Multiple transformer decoder layers
    - Causal attention mask for autoregressive generation
    - ~5-10M parameters (FL-friendly)
    """
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 4,
        ff_dim: int = 512,
        dropout: float = 0.1,
        max_seq_len: int = 128,
        num_classes: int = None,  # Alias for vocab_size (for compatibility)
    ):
        """Initialize Transformer model.
        
        Args:
            vocab_size: Size of character vocabulary
            embed_dim: Embedding/model dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            ff_dim: Feed-forward hidden dimension
            dropout: Dropout probability
            max_seq_len: Maximum sequence length
            num_classes: Alias for vocab_size (ignored if vocab_size is set)
        """
        super().__init__()
        
        # Use num_classes as fallback for vocab_size
        if vocab_size is None or vocab_size == 0:
            vocab_size = num_classes
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        
        # Token embedding
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(embed_dim, max_seq_len, dropout)
        
        # Transformer decoder layers
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # Pre-norm for better training
        )
        self.transformer = nn.TransformerEncoder(
            decoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        
        # Layer norm
        self.ln_f = nn.LayerNorm(embed_dim)
        
        # Output projection
        self.fc_out = nn.Linear(embed_dim, vocab_size)
        
        # Initialize weights
        self._init_weights()
        
        # Cache for causal mask
        self._causal_mask_cache = {}
    
    def _init_weights(self):
        """Initialize weights with GPT-style initialization."""
        # Embeddings
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        
        # Output projection
        nn.init.normal_(self.fc_out.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.fc_out.bias)
        
        # Apply special initialization to transformer layers
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def _get_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Get or create causal attention mask.
        
        Args:
            seq_len: Sequence length
            device: Device for mask tensor
        
        Returns:
            Causal mask of shape (seq_len, seq_len)
        """
        cache_key = (seq_len, device)
        if cache_key not in self._causal_mask_cache:
            # Create causal mask (upper triangular with -inf)
            mask = torch.triu(
                torch.ones(seq_len, seq_len, device=device) * float('-inf'),
                diagonal=1
            )
            self._causal_mask_cache[cache_key] = mask
        return self._causal_mask_cache[cache_key]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length)
        
        Returns:
            Output logits of shape (batch_size, sequence_length, vocab_size)
        """
        batch_size, seq_len = x.shape
        device = x.device
        
        # Token embedding: (batch, seq) -> (batch, seq, embed_dim)
        embedded = self.embedding(x) * math.sqrt(self.embed_dim)
        
        # Add positional encoding
        embedded = self.pos_encoding(embedded)
        
        # Get causal mask
        causal_mask = self._get_causal_mask(seq_len, device)
        
        # Transformer forward
        hidden = self.transformer(embedded, mask=causal_mask, is_causal=True)
        
        # Final layer norm
        hidden = self.ln_f(hidden)
        
        # Project to vocabulary
        logits = self.fc_out(hidden)
        
        return logits
    
    @torch.no_grad()
    def generate(
        self,
        prompt: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """Generate text autoregressively.
        
        Args:
            prompt: Starting token ids of shape (batch_size, prompt_len)
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_k: If set, only sample from top-k tokens
        
        Returns:
            Generated token ids of shape (batch_size, prompt_len + max_new_tokens)
        """
        self.eval()
        
        generated = prompt
        
        for _ in range(max_new_tokens):
            # Truncate to max sequence length
            x = generated[:, -self.max_seq_len:]
            
            # Get logits
            logits = self(x)
            
            # Take last position
            logits = logits[:, -1, :] / temperature
            
            # Apply top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append
            generated = torch.cat([generated, next_token], dim=1)
        
        return generated
