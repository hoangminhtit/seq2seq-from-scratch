import argparse
import torch
import torch.nn as nn
import numpy as np
import sentencepiece as spm
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from transformer import Transformer
from prepare_data import MachineTranslationDataset

def collate_fn(batch):
    """Custom collate function to pad sequences dynamically"""
    src_batch, tgt_batch = zip(*batch)
    src_batch = pad_sequence(src_batch, batch_first=True, padding_value=0)
    tgt_batch = pad_sequence(tgt_batch, batch_first=True, padding_value=0)
    return src_batch, tgt_batch

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

def translate_sentence(model, sentence, sp, device, max_length=128):
    """Translate a single sentence"""
    model.eval()
    
    # Encode source sentence
    src_ids = sp.encode(sentence)
    src = torch.tensor([src_ids], dtype=torch.long).to(device)
    
    # Start with BOS token (assuming id=1)
    tgt_ids = [1]  # BOS token
    
    with torch.no_grad():
        for _ in range(max_length):
            tgt = torch.tensor([tgt_ids], dtype=torch.long).to(device)
            
            # Forward pass
            output = model(src, tgt)
            
            # Get next token
            next_token = output[0, -1, :].argmax().item()
            
            # Stop if EOS token (assuming id=2)
            if next_token == 2:
                break
            
            tgt_ids.append(next_token)
    
    # Decode target sentence
    translation = sp.decode(tgt_ids[1:])  # Skip BOS token
    return translation

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate or translate with trained model')
    
    # Model parameters
    parser.add_argument('--vocab_size', type=int, default=16000, help='Vocabulary size')
    parser.add_argument('--d_model', type=int, default=512, help='Model dimension')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=6, help='Number of encoder/decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='Feed-forward dimension')
    parser.add_argument('--max_seq_length', type=int, default=128, help='Maximum sequence length')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    
    # Evaluation parameters
    parser.add_argument('--model_path', type=str, default='best_model.pt', help='Path to trained model')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', 
                       help='Device to use')
    parser.add_argument('--dataset', type=str, default='test', choices=['train', 'validation', 'test'],
                       help='Dataset to evaluate on')
    
    # Translation parameters
    parser.add_argument('--translate', action='store_true', help='Enter translation mode')
    parser.add_argument('--sentence', type=str, default=None, help='Sentence to translate')
    
    args = parser.parse_args()
    
    print(f"Using device: {args.device}")
    
    # Load tokenizer
    print("Loading tokenizer...")
    sp = spm.SentencePieceProcessor()
    sp.load("tokenizer.model")
    
    # Initialize model
    print("Initializing model...")
    model = Transformer(
        src_vocab_size=args.vocab_size,
        tgt_vocab_size=args.vocab_size,
        d_model=args.d_model,
        d_ff=args.d_ff,
        max_seq_length=args.max_seq_length,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout
    ).to(args.device)
    
    # Load trained model
    print(f"Loading model from {args.model_path}...")
    checkpoint = torch.load(args.model_path, map_location=args.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if 'epoch' in checkpoint:
        print(f"Model trained for {checkpoint['epoch']+1} epochs")
    if 'val_loss' in checkpoint:
        print(f"Best validation loss: {checkpoint['val_loss']:.4f}")
    
    # Translation mode
    if args.translate:
        print("\n" + "="*50)
        print("Translation Mode")
        print("="*50)
        
        if args.sentence:
            # Translate single sentence
            print(f"\nSource: {args.sentence}")
            translation = translate_sentence(model, args.sentence, sp, args.device, args.max_seq_length)
            print(f"Translation: {translation}")
        else:
            # Interactive translation
            print("Enter sentences to translate (type 'quit' or 'exit' to stop):\n")
            while True:
                sentence = input("English: ").strip()
                if sentence.lower() in ['quit', 'exit', 'q']:
                    break
                if sentence:
                    translation = translate_sentence(model, sentence, sp, args.device, args.max_seq_length)
                    print(f"Vietnamese: {translation}\n")
    else:
        # Evaluation mode
        print("\n" + "="*50)
        print(f"Evaluating on {args.dataset} set")
        print("="*50)
        
        # Load dataset
        print(f"Loading {args.dataset} dataset...")
        src = np.load(f'./data/{args.dataset}_source.npy', allow_pickle=True)
        target = np.load(f'./data/{args.dataset}_target.npy', allow_pickle=True)
        
        print(f"Dataset size: {len(src)} samples")
        
        # Create dataloader
        dataset = MachineTranslationDataset(source_ids=src, target_ids=target)
        dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size, 
                               shuffle=False, collate_fn=collate_fn)
        
        # Evaluate
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        loss = evaluate(model, dataloader, criterion, args.device)
        
        print(f"\n{args.dataset.capitalize()} Loss: {loss:.4f}")
        print(f"Perplexity: {torch.exp(torch.tensor(loss)):.4f}")
        
        # Show some sample translations
        print("\n" + "="*50)
        print("Sample Translations")
        print("="*50)
        
        num_samples = min(5, len(src))
        for i in range(num_samples):
            src_text = sp.decode(src[i].tolist())
            tgt_text = sp.decode(target[i].tolist())
            pred_text = translate_sentence(model, src_text, sp, args.device, args.max_seq_length)
            
            print(f"\nSample {i+1}:")
            print(f"  Source:      {src_text}")
            print(f"  Reference:   {tgt_text}")
            print(f"  Translation: {pred_text}")
    
    print("\n✓ Evaluation completed!")
