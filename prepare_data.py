import os
import torch
import numpy as np
import torch.nn as nn 
import sentencepiece as spm
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader

# Load dataset from hf
dataset = load_dataset("thainq107/iwslt2015-en-vi")

class MachineTranslationDataset(Dataset):
    def __init__(self, source_ids, target_ids):
        self.source_ids = source_ids
        self.target_ids = target_ids
    
    def __len__(self):
        return len(self.source_ids)
    
    def __getitem__(self, idx):
        x = torch.tensor(self.source_ids[idx], dtype=torch.long)
        y = torch.tensor(self.target_ids[idx], dtype=torch.long)
        # x = self.source_ids[idx]
        # y = self.target_ids[idx]
        return x, y
    
def prepare_data(vocab_size: int, max_seq_length: int):
    os.makedirs('data', exist_ok=True)
    with open('./data/translate_en_vi.txt', 'w', encoding='utf-8') as f:
        for split in ['train', 'test', 'validation']:
            for sample in dataset[split]:
                f.write(sample['en'] + '\n')
                f.write(sample['vi'] + '\n')
    
    # Train sentencepiece tokenizer
    spm.SentencePieceTrainer.Train(
        input="./data/translate_en_vi.txt",
        model_prefix="tokenizer",
        vocab_size=vocab_size,
        character_coverage=1.0,
        model_type="bpe"
    )

    # Load trained tokenizer
    sp = spm.SentencePieceProcessor()
    sp.load("tokenizer.model")

    for split in ['train', 'test', 'validation']:
        source_ids = []
        target_ids = []
        for sample in dataset[split]:
            source = sp.encode(sample['en'])
            target = sp.encode(sample['vi'])
            if len(source) <= max_seq_length and len(target) <= max_seq_length:
                source_ids.append(source)
                target_ids.append(target)
        np.save(f"./data/{split}_source.npy", np.array(source_ids, dtype=object))
        np.save(f"./data/{split}_target.npy", np.array(target_ids, dtype=object))
    
    # Save vocab size for later
    stats = {
        "vocab_size": vocab_size,
        "train_samples": len(dataset['train']),
        "val_samples": len(dataset['validation']),
        "test_samples": len(dataset['test'])
    }
    return stats

if __name__=="__main__":
    stats = prepare_data(vocab_size=16000, max_seq_length=128)
    print(stats)
  
    # Build dataset
    train_src = np.load('./data/train_source.npy', allow_pickle=True)
    train_target = np.load('./data/train_source.npy', allow_pickle=True)

    test_src = np.load('./data/test_source.npy', allow_pickle=True)
    test_target = np.load('./data/test_source.npy', allow_pickle=True)

    val_src = np.load('./data/validation_source.npy', allow_pickle=True)
    val_target = np.load('./data/validation_source.npy', allow_pickle=True)

    train_dataset = MachineTranslationDataset(source_ids=train_src, target_ids=train_target)
    test_dataset = MachineTranslationDataset(source_ids=test_src, target_ids=test_target)
    val_dataset = MachineTranslationDataset(source_ids=val_src, target_ids=val_target)

    # Build dataloader
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=32, shuffle=False)