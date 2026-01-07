import os
import torch
import sentencepiece as spm
import argparse

from transformer import Transformer

def generate_translation(model_path, tokenizer_path, prompt, max_new_tokens=50, device='cpu', use_beam_search=True, beam_size=5):
    """
    Generate translation from English to Vietnamese using trained Transformer Seq2Seq
    """
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    model_args = checkpoint.get('model_args', {})

    # Load tokenizer SentencePiece
    sp = spm.SentencePieceProcessor()
    if os.path.isdir(tokenizer_path):
        sp.load(os.path.join(tokenizer_path, "tokenizer.model"))
    else:
        sp.load(tokenizer_path)

    # Get vocab size from model_args or use default
    vocab_size = model_args.get('vocab_size', 16000)

    # Create model
    model = Transformer(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        d_model=model_args.get('d_model', 512),
        num_heads=model_args.get('num_heads', 8),
        num_layers=model_args.get('num_layers', 6),
        d_ff=model_args.get('d_ff', 2048),
        max_seq_length=model_args.get('max_seq_length', 128),
        dropout=0.0
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Encode input prompt
    src_ids = sp.encode(prompt, out_type=int)
    src_tensor = torch.tensor(src_ids, dtype=torch.long, device=device).unsqueeze(0)

    # Generate translation
    bos_id = sp.bos_id() if sp.bos_id() >= 0 else 1
    eos_id = sp.eos_id() if sp.eos_id() >= 0 else 2

    if use_beam_search:
        translation = beam_search_decode(model, src_tensor, sp, device, max_new_tokens, beam_size, bos_id, eos_id)
    else:
        translation = greedy_decode(model, src_tensor, sp, device, max_new_tokens, bos_id, eos_id)
    
    return translation

def greedy_decode(model, src_tensor, sp, device, max_new_tokens, bos_id, eos_id):
    """Greedy decoding strategy"""
    with torch.no_grad():
        src_mask = (src_tensor != 0).unsqueeze(1).unsqueeze(2)
        src_emb = model.dropout(model.positional_encoding(model.encoder_embedding(src_tensor)))
        for enc_layer in model.encoder_layers:
            src_emb = enc_layer(src_emb, src_mask)
        enc_output = src_emb

        ys = torch.full((1, 1), bos_id, dtype=torch.long, device=device)
        for _ in range(max_new_tokens):
            tgt_mask = torch.tril(torch.ones((1, 1, ys.size(1), ys.size(1)), device=device)).bool()
            tgt_emb = model.dropout(model.positional_encoding(model.decoder_embedding(ys)))
            dec_output = tgt_emb
            for dec_layer in model.decoder_layers:
                dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)
            out = model.fc(dec_output[:, -1])
            next_token = out.argmax(-1).unsqueeze(1)
            ys = torch.cat([ys, next_token], dim=1)
            if next_token.item() == eos_id:
                break

        output_ids = ys[0, 1:].tolist()  # remove BOS
        # Remove EOS if present
        if output_ids and output_ids[-1] == eos_id:
            output_ids = output_ids[:-1]

    return sp.decode(output_ids)

def beam_search_decode(model, src_tensor, sp, device, max_new_tokens, beam_size, bos_id, eos_id):
    """Beam search decoding strategy"""
    with torch.no_grad():
        # Encode source
        src_mask = (src_tensor != 0).unsqueeze(1).unsqueeze(2)
        src_emb = model.dropout(model.positional_encoding(model.encoder_embedding(src_tensor)))
        for enc_layer in model.encoder_layers:
            src_emb = enc_layer(src_emb, src_mask)
        enc_output = src_emb

        # Initialize beams: (sequence, score)
        beams = [(torch.tensor([[bos_id]], dtype=torch.long, device=device), 0.0)]
        completed = []

        for step in range(max_new_tokens):
            candidates = []
            
            for seq, score in beams:
                # Skip if already ended
                if seq[0, -1].item() == eos_id:
                    completed.append((seq, score))
                    continue

                # Decode
                tgt_mask = torch.tril(torch.ones((1, 1, seq.size(1), seq.size(1)), device=device)).bool()
                tgt_emb = model.dropout(model.positional_encoding(model.decoder_embedding(seq)))
                dec_output = tgt_emb
                for dec_layer in model.decoder_layers:
                    dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)
                out = model.fc(dec_output[:, -1])
                log_probs = torch.log_softmax(out, dim=-1)

                # Get top-k candidates
                top_log_probs, top_indices = torch.topk(log_probs[0], beam_size)
                
                for log_prob, idx in zip(top_log_probs, top_indices):
                    new_seq = torch.cat([seq, idx.unsqueeze(0).unsqueeze(0)], dim=1)
                    new_score = score + log_prob.item()
                    candidates.append((new_seq, new_score))

            if not candidates:
                break

            # Select top beam_size candidates (normalize by length)
            candidates.sort(key=lambda x: x[1] / x[0].size(1), reverse=True)
            beams = candidates[:beam_size]

            # Stop if all beams ended
            if all(seq[0, -1].item() == eos_id for seq, _ in beams):
                completed.extend(beams)
                break

        # Add remaining beams
        completed.extend(beams)
        
        if not completed:
            return ""

        # Get best sequence
        completed.sort(key=lambda x: x[1] / x[0].size(1), reverse=True)
        best_seq = completed[0][0][0].tolist()
        
        # Remove special tokens
        output_ids = [token for token in best_seq[1:] if token not in [bos_id, eos_id, 0]]
        
    return sp.decode(output_ids)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate translation with trained Transformer Seq2Seq model")
    parser.add_argument("--model_path", type=str, default="best_model.pt", help="Path to trained model")
    parser.add_argument("--tokenizer_path", type=str, default="tokenizer.model", help="Path to trained tokenizer")
    parser.add_argument("--prompt", type=str, default=None, help="English sentence to translate")
    parser.add_argument("--max_tokens", type=int, default=100, help="Maximum number of tokens to generate")
    parser.add_argument("--beam_size", type=int, default=5, help="Beam size for beam search")
    parser.add_argument("--greedy", action="store_true", help="Use greedy decoding instead of beam search")
    parser.add_argument("--no_cuda", action="store_true", help="Force CPU inference")

    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu'
    
    print(f"Using device: {device}")
    print(f"Decoding method: {'Greedy' if args.greedy else 'Beam Search'}")
    if not args.greedy:
        print(f"Beam size: {args.beam_size}")
    print("-" * 50)
    
    if args.prompt:
        # Single translation
        translation = generate_translation(
            model_path=args.model_path,
            tokenizer_path=args.tokenizer_path,
            prompt=args.prompt,
            max_new_tokens=args.max_tokens,
            device=device,
            use_beam_search=not args.greedy,
            beam_size=args.beam_size
        )
        print(f"\nEnglish: {args.prompt}")
        print(f"Vietnamese: {translation}")
    else:
        # Interactive mode
        print("\nInteractive Translation Mode (English to Vietnamese)")
        print("Type 'quit', 'exit' or 'q' to stop\n")
        
        while True:
            try:
                prompt = input("English: ").strip()
                if prompt.lower() in ['quit', 'exit', 'q']:
                    break
                if prompt:
                    translation = generate_translation(
                        model_path=args.model_path,
                        tokenizer_path=args.tokenizer_path,
                        prompt=prompt,
                        max_new_tokens=args.max_tokens,
                        device=device,
                        use_beam_search=not args.greedy,
                        beam_size=args.beam_size
                    )
                    print(f"Vietnamese: {translation}\n")
            except KeyboardInterrupt:
                print("\n\nExiting...")
                break