import argparse
import os
import torch
import torch.nn as nn
import numpy as np
from torch import optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from transformer import Transformer
from prepare_data import prepare_data, MachineTranslationDataset

def collate_fn(batch):
    """Custom collate function to pad sequences dynamically"""
    src_batch, tgt_batch = zip(*batch)
    src_batch = pad_sequence(src_batch, batch_first=True, padding_value=0)
    tgt_batch = pad_sequence(tgt_batch, batch_first=True, padding_value=0)
    return src_batch, tgt_batch

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    for src, tgt in tqdm(dataloader, desc="Training"):
        src, tgt = src.to(device), tgt.to(device)
        
        optimizer.zero_grad()
        output = model(src, tgt[:, :-1])
        loss = criterion(output.contiguous().view(-1, model.fc.out_features), 
                        tgt[:, 1:].contiguous().view(-1))
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device):
    """Evaluate the model"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for src, tgt in tqdm(dataloader, desc="Evaluating"):
            src, tgt = src.to(device), tgt.to(device)
            
            output = model(src, tgt[:, :-1])
            loss = criterion(output.contiguous().view(-1, model.fc.out_features), 
                           tgt[:, 1:].contiguous().view(-1))
            
            total_loss += loss.item()
    
    return total_loss / len(dataloader)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    
    # Model parameters
    parser.add_argument('--vocab_size', type=int, default=16000, help='Vocabulary size')
    parser.add_argument('--d_model', type=int, default=512, help='Model dimension')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=6, help='Number of encoder/decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='Feed-forward dimension')
    parser.add_argument('--max_seq_length', type=int, default=128, help='Maximum sequence length')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--prepare_data', action='store_true', help='Prepare data before training')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', 
                       help='Device to use')
    
    args = parser.parse_args()
    
    print(f"Using device: {args.device}")
    
    # Step 1: Prepare data if needed
    if args.prepare_data or not os.path.exists('./data/train_source.npy'):
        print("Preparing data...")
        stats = prepare_data(vocab_size=args.vocab_size, max_seq_length=args.max_seq_length)
        print(f"Data preparation completed: {stats}")
    else:
        print("Using existing data...")
    
    # Step 2: Load prepared data
    print("Loading datasets...")
    train_src = np.load('./data/train_source.npy', allow_pickle=True)
    train_target = np.load('./data/train_target.npy', allow_pickle=True)
    
    test_src = np.load('./data/test_source.npy', allow_pickle=True)
    test_target = np.load('./data/test_target.npy', allow_pickle=True)
    
    val_src = np.load('./data/validation_source.npy', allow_pickle=True)
    val_target = np.load('./data/validation_target.npy', allow_pickle=True)
    
    print(f"Train samples: {len(train_src)}")
    print(f"Test samples: {len(test_src)}")
    print(f"Validation samples: {len(val_src)}")
    
    # Step 3: Create datasets and dataloaders
    train_dataset = MachineTranslationDataset(source_ids=train_src, target_ids=train_target)
    test_dataset = MachineTranslationDataset(source_ids=test_src, target_ids=test_target)
    val_dataset = MachineTranslationDataset(source_ids=val_src, target_ids=val_target)
    
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, 
                                 shuffle=True, collate_fn=collate_fn)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, 
                                shuffle=False, collate_fn=collate_fn)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, 
                               shuffle=False, collate_fn=collate_fn)
    
    # Step 4: Initialize model
    print("Initializing model...")
    transformer = Transformer(
        src_vocab_size=args.vocab_size,
        tgt_vocab_size=args.vocab_size,
        d_model=args.d_model,
        d_ff=args.d_ff,
        max_seq_length=args.max_seq_length,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout
    ).to(args.device)
    
    # Print model parameters
    total_params = sum(p.numel() for p in transformer.parameters())
    trainable_params = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Step 5: Setup training
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(transformer.parameters(), lr=args.lr)
    
    # Step 6: Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"{'='*50}")
        
        # Train
        train_loss = train_epoch(transformer, train_dataloader, criterion, optimizer, args.device)
        print(f"Train Loss: {train_loss:.4f}")
        
        # Validate
        val_loss = evaluate(transformer, val_dataloader, criterion, args.device)
        print(f"Validation Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': transformer.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'model_args': {
                    'vocab_size': args.vocab_size,
                    'd_model': args.d_model,
                    'num_heads': args.num_heads,
                    'num_layers': args.num_layers,
                    'd_ff': args.d_ff,
                    'max_seq_length': args.max_seq_length,
                    'dropout': args.dropout
                }
            }, 'best_model.pt')
            print(f"Best model saved with validation loss: {val_loss:.4f}")
    
    # Step 7: Final evaluation on test set
    print(f"\n{'='*50}")
    print("Final Evaluation on Test Set")
    print(f"{'='*50}")
    
    # Load best model
    checkpoint = torch.load('best_model.pt')
    transformer.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss = evaluate(transformer, test_dataloader, criterion, args.device)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Perplexity: {torch.exp(torch.tensor(test_loss)):.4f}")
    
    print("\n✓ Training completed!")
    print("\nTo evaluate the model, run:")
    print("  python evaluate.py --model_path best_model.pt --dataset test")
    print("\nTo translate sentences, run:")
    print("  python generate.py")
    print("  python generate.py --prompt 'Hello, how are you?'")
    print("  python generate.py --prompt 'Hello, how are you?' --beam_size 10")
    print("  python generate.py --prompt 'Hello, how are you?' --greedy")
