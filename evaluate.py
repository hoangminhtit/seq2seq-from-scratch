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

def translate_sentence(model, sentence, sp, device, max_length=128, use_beam_search=False, beam_size=5):
    """Translate a single sentence using greedy or beam search"""
    model.eval()
    
    # Encode source sentence
    src_ids = sp.encode(sentence)
    src = torch.tensor([src_ids], dtype=torch.long).to(device)
    
    if use_beam_search:
        return beam_search_translate(model, src, sp, device, max_length, beam_size)
    else:
        return greedy_translate(model, src, sp, device, max_length)

def greedy_translate(model, src, sp, device, max_length=128):
    """Greedy decoding translation"""
    # Start with BOS token
    bos_id = sp.bos_id() if sp.bos_id() >= 0 else 1
    eos_id = sp.eos_id() if sp.eos_id() >= 0 else 2
    
    tgt_ids = [bos_id]
    
    with torch.no_grad():
        for _ in range(max_length):
            tgt = torch.tensor([tgt_ids], dtype=torch.long).to(device)
            
            # Forward pass
            output = model(src, tgt)
            
            # Get next token
            next_token = output[0, -1, :].argmax().item()
            
            # Stop if EOS token
            if next_token == eos_id:
                break
            
            tgt_ids.append(next_token)
    
    # Decode target sentence (skip BOS)
    output_ids = [t for t in tgt_ids[1:] if t not in [0, bos_id, eos_id]]
    translation = sp.decode(output_ids)
    return translation

def beam_search_translate(model, src, sp, device, max_length=128, beam_size=5):
    """Beam search translation"""
    bos_id = sp.bos_id() if sp.bos_id() >= 0 else 1
    eos_id = sp.eos_id() if sp.eos_id() >= 0 else 2
    
    # Initialize beams
    beams = [(torch.tensor([[bos_id]], dtype=torch.long).to(device), 0.0)]
    completed = []
    
    with torch.no_grad():
        for _ in range(max_length):
            candidates = []
            
            for seq, score in beams:
                # Check if ended
                if seq[0, -1].item() == eos_id:
                    completed.append((seq, score))
                    continue
                
                # Forward pass
                output = model(src, seq)
                logits = output[0, -1, :]
                log_probs = torch.log_softmax(logits, dim=-1)
                
                # Get top-k
                top_log_probs, top_indices = torch.topk(log_probs, beam_size)
                
                for log_prob, idx in zip(top_log_probs, top_indices):
                    new_seq = torch.cat([seq, idx.unsqueeze(0).unsqueeze(0)], dim=1)
                    new_score = score + log_prob.item()
                    candidates.append((new_seq, new_score))
            
            if not candidates:
                break
            
            # Select top beams (normalize by length)
            candidates.sort(key=lambda x: x[1] / x[0].size(1), reverse=True)
            beams = candidates[:beam_size]
            
            # All ended?
            if all(seq[0, -1].item() == eos_id for seq, _ in beams):
                completed.extend(beams)
                break
        
        completed.extend(beams)
    
    if not completed:
        return ""
    
    # Get best
    completed.sort(key=lambda x: x[1] / x[0].size(1), reverse=True)
    best_seq = completed[0][0][0].tolist()
    
    # Clean output
    output_ids = [t for t in best_seq[1:] if t not in [0, bos_id, eos_id]]
    translation = sp.decode(output_ids)
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
    parser.add_argument('--beam_search', action='store_true', help='Use beam search for translation')
    parser.add_argument('--beam_size', type=int, default=5, help='Beam size for beam search')
    
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
        print(f"Decoding: {'Beam Search (size=' + str(args.beam_size) + ')' if args.beam_search else 'Greedy'}")
        
        if args.sentence:
            # Translate single sentence
            print(f"\nSource: {args.sentence}")
            translation = translate_sentence(
                model, args.sentence, sp, args.device, 
                args.max_seq_length, args.beam_search, args.beam_size
            )
            print(f"Translation: {translation}")
        else:
            # Interactive translation
            print("Enter sentences to translate (type 'quit' or 'exit' to stop):\n")
            while True:
                sentence = input("English: ").strip()
                if sentence.lower() in ['quit', 'exit', 'q']:
                    break
                if sentence:
                    translation = translate_sentence(
                        model, sentence, sp, args.device, 
                        args.max_seq_length, args.beam_search, args.beam_size
                    )
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
            pred_text = translate_sentence(
                model, src_text, sp, args.device, 
                args.max_seq_length, args.beam_search, args.beam_size
            )
            
            print(f"\nSample {i+1}:")
            print(f"  Source:      {src_text}")
            print(f"  Reference:   {tgt_text}")
            print(f"  Translation: {pred_text}")
    
    print("\n✓ Evaluation completed!")
