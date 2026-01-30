"""NLP dataset loading utilities for federated learning.

Handles text-based datasets like Shakespeare for character-level language modeling.
"""

from typing import Dict, Optional, Tuple, Any, List
import torch
from torch.utils.data import DataLoader, Dataset
from omegaconf import DictConfig


class CharacterVocab:
    """Character-level vocabulary for text tokenization."""
    
    def __init__(self, chars: Optional[str] = None):
        """Initialize vocabulary.
        
        Args:
            chars: String of unique characters to include. If None, uses default charset.
        """
        if chars is None:
            # Default charset for Shakespeare
            chars = "\n !\"$&'(),-.0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]abcdefghijklmnopqrstuvwxyz"
        
        self.chars = sorted(set(chars))
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}
        self.vocab_size = len(self.chars)
    
    def encode(self, text: str) -> List[int]:
        """Encode text to list of indices."""
        return [self.char_to_idx.get(ch, 0) for ch in text]
    
    def decode(self, indices: List[int]) -> str:
        """Decode list of indices to text."""
        return ''.join([self.idx_to_char.get(i, '') for i in indices])


class CharLMDataset(Dataset):
    """Dataset for character-level language modeling."""
    
    def __init__(
        self,
        text: str,
        vocab: CharacterVocab,
        sequence_length: int = 80,
    ):
        """Initialize dataset.
        
        Args:
            text: Full text string
            vocab: Character vocabulary
            sequence_length: Length of input sequences
        """
        self.vocab = vocab
        self.sequence_length = sequence_length
        
        # Encode the text
        self.encoded = torch.tensor(vocab.encode(text), dtype=torch.long)
        
        # Calculate number of sequences
        self.num_sequences = max(0, len(self.encoded) - sequence_length)
    
    def __len__(self) -> int:
        return self.num_sequences
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sequence pair for language modeling.
        
        Returns:
            Dict with 'input_ids' and 'target_ids'
        """
        input_ids = self.encoded[idx:idx + self.sequence_length]
        target_ids = self.encoded[idx + 1:idx + self.sequence_length + 1]
        
        return {
            "input_ids": input_ids,
            "target_ids": target_ids,
        }


def build_vocab_from_dataset(dataset_cfg: DictConfig) -> CharacterVocab:
    """Build vocabulary from dataset configuration.
    
    Args:
        dataset_cfg: Dataset configuration
    
    Returns:
        CharacterVocab instance
    """
    # Use predefined charset for Shakespeare
    # This covers most printable ASCII characters in the corpus
    shakespeare_chars = "\n !\"$&'(),-.0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]abcdefghijklmnopqrstuvwxyz"
    return CharacterVocab(shakespeare_chars)


def load_text_data(
    partition_id: int,
    num_partitions: int,
    dataset_cfg: DictConfig,
    partitioner_cfg: DictConfig,
    batch_size: int = 32,
    test_fraction: float = 0.2,
) -> Tuple[DataLoader, DataLoader]:
    """Load partitioned text data for a client.
    
    For Shakespeare-style datasets where the entire corpus is one text blob,
    we partition by splitting the text into chunks.
    
    Args:
        partition_id: ID of the partition to load
        num_partitions: Total number of partitions
        dataset_cfg: Dataset configuration
        partitioner_cfg: Partitioner configuration
        batch_size: Batch size for data loaders
        test_fraction: Fraction of data to use for testing
    
    Returns:
        Tuple of (train_loader, test_loader)
    """
    from datasets import load_dataset
    
    # Load the full training dataset
    dataset = load_dataset(
        dataset_cfg.dataset_name, 
        split="train",
        trust_remote_code=True,
    )
    
    # Extract text - Shakespeare has one row with all text
    text_key = dataset_cfg.get("text_key", "text")
    full_text = dataset[0][text_key]
    
    # Partition the text by splitting into chunks
    # Each client gets a contiguous chunk of the text
    chunk_size = len(full_text) // num_partitions
    start_idx = partition_id * chunk_size
    end_idx = start_idx + chunk_size if partition_id < num_partitions - 1 else len(full_text)
    
    client_text = full_text[start_idx:end_idx]
    
    # Split client's text into train/val
    split_point = int(len(client_text) * (1 - test_fraction))
    train_text = client_text[:split_point]
    val_text = client_text[split_point:]
    
    # Build vocabulary
    vocab = build_vocab_from_dataset(dataset_cfg)
    
    # Get sequence length from config
    sequence_length = dataset_cfg.get("sequence_length", 80)
    
    # Create datasets
    train_dataset = CharLMDataset(train_text, vocab, sequence_length)
    test_dataset = CharLMDataset(val_text, vocab, sequence_length)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    
    return train_loader, test_loader


def get_centralized_text_testset(
    dataset_cfg: DictConfig,
    batch_size: int = 32,
) -> DataLoader:
    """Get centralized test set for server-side evaluation.
    
    Args:
        dataset_cfg: Dataset configuration
        batch_size: Batch size
    
    Returns:
        DataLoader for test set
    """
    from datasets import load_dataset
    
    # Load test split
    dataset = load_dataset(
        dataset_cfg.dataset_name, 
        split="test",
        trust_remote_code=True,
    )
    
    # Extract text
    text_key = dataset_cfg.get("text_key", "text")
    full_text = dataset[0][text_key]
    
    # Build vocabulary
    vocab = build_vocab_from_dataset(dataset_cfg)
    
    # Get sequence length from config
    sequence_length = dataset_cfg.get("sequence_length", 80)
    
    # Create dataset
    test_dataset = CharLMDataset(full_text, vocab, sequence_length)
    
    return DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
