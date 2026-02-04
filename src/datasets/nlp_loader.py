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


class PrebuiltSequenceDataset(Dataset):
    """Dataset for pre-built sequences (used for IID partitioning)."""
    
    def __init__(self, sequences: List[Tuple[List[int], List[int]]]):
        """Initialize dataset from pre-built sequences.
        
        Args:
            sequences: List of (input_ids, target_ids) tuples
        """
        self.sequences = sequences
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sequence pair."""
        input_ids, target_ids = self.sequences[idx]
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "target_ids": torch.tensor(target_ids, dtype=torch.long),
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
    
    Supports both IID and non-IID partitioning:
    - IID: Sequences are shuffled globally, then distributed uniformly
    - Non-IID (contiguous): Each client gets a contiguous chunk of text
    
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
    import random
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
    
    # Build vocabulary
    vocab = build_vocab_from_dataset(dataset_cfg)
    
    # Get sequence length from config
    sequence_length = dataset_cfg.get("sequence_length", 80)
    
    # Check partitioner type
    partitioner_name = partitioner_cfg.get("name", "iid").lower()
    
    if partitioner_name == "iid":
        # TRUE IID: Build sequences with stride, shuffle, then partition
        # Encode the full text
        encoded = vocab.encode(full_text)
        
        # OPTIMIZATION: Use stride to reduce sequence count (LEAF-like scale)
        # stride=10 on 1.1M chars → ~110k total sequences
        # max_sequences_per_client=5000 caps each client for fast training
        stride = dataset_cfg.get("stride", 10)
        max_sequences_per_client = dataset_cfg.get("max_sequences_per_client", 5000)
        
        # Build sequences with stride (not every position)
        all_sequences = []
        for i in range(0, len(encoded) - sequence_length, stride):
            input_seq = encoded[i:i + sequence_length]
            target_seq = encoded[i + 1:i + sequence_length + 1]
            all_sequences.append((input_seq, target_seq))
        
        # Shuffle with fixed seed for reproducibility
        seed = partitioner_cfg.get("seed", 42)
        random.Random(seed).shuffle(all_sequences)
        
        # Partition uniformly across clients with cap
        sequences_per_client = len(all_sequences) // num_partitions
        sequences_per_client = min(sequences_per_client, max_sequences_per_client)
        
        start_idx = partition_id * sequences_per_client
        end_idx = min(start_idx + sequences_per_client, len(all_sequences))
        
        client_sequences = all_sequences[start_idx:end_idx]
        
        # Split into train/val
        split_point = int(len(client_sequences) * (1 - test_fraction))
        train_sequences = client_sequences[:split_point]
        val_sequences = client_sequences[split_point:]
        
        # Create datasets from pre-built sequences
        train_dataset = PrebuiltSequenceDataset(train_sequences)
        test_dataset = PrebuiltSequenceDataset(val_sequences)
    else:
        # NON-IID (contiguous chunks): Original behavior
        # Each client gets a contiguous chunk of the text
        chunk_size = len(full_text) // num_partitions
        start_idx = partition_id * chunk_size
        end_idx = start_idx + chunk_size if partition_id < num_partitions - 1 else len(full_text)
        
        client_text = full_text[start_idx:end_idx]
        
        # Split client's text into train/val
        split_point = int(len(client_text) * (1 - test_fraction))
        train_text = client_text[:split_point]
        val_text = client_text[split_point:]
        
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
