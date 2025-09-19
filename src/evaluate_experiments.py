import os
import json
import torch
import pandas as pd
import sentencepiece as spm
from model import Encoder, Decoder, Seq2Seq
from evaluate import evaluate_model
import argparse

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_experiment_model(exp_dir):
    """Load model from experiment directory"""
    checkpoint = torch.load(os.path.join(exp_dir, 'best.pt'), map_location=DEVICE)
    config = checkpoint['config']
    
    # Load tokenizers
    vocab_dir = os.path.join("data", "processed", "vocab")
    sp_urdu = spm.SentencePieceProcessor(model_file=os.path.join(vocab_dir, "urdu_bpe.model"))
    sp_roman = spm.SentencePieceProcessor(model_file=os.path.join(vocab_dir, "roman_bpe.model"))
    urdu_vocab_size = sp_urdu.get_piece_size()
    roman_vocab_size = sp_roman.get_piece_size()
    
    # Create model
    encoder = Encoder(urdu_vocab_size, config['emb'], config['hid'], 
                     n_layers=config['enc_layers'], dropout=config['dropout'])
    decoder = Decoder(roman_vocab_size, config['emb'], config['hid'], 
                     n_layers=config['dec_layers'], dropout=config['dropout'])
    model = Seq2Seq(encoder, decoder, DEVICE).to(DEVICE)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    
    return model, config, sp_urdu, sp_roman

def evaluate_all_experiments():
    """Evaluate all completed experiments and create comparison table"""
    
    runs_dir = "runs"
    if not os.path.exists(runs_dir):
        print("❌ No runs directory found. Run experiments first.")
        return
    
    results = []
    
    for exp_name in os.listdir(runs_dir):
        exp_dir = os.path.join(runs_dir, exp_name)
        if not os.path.isdir(exp_dir):
            continue
            
        best_pt = os.path.join(exp_dir, 'best.pt')
        results_json = os.path.join(exp_dir, 'results.json')
        
        if not os.path.exists(best_pt) or not os.path.exists(results_json):
            print(f"⚠️ Skipping {exp_name}: missing best.pt or results.json")
            continue
        
        print(f"🔍 Evaluating experiment: {exp_name}")
        
        try:
            # Load experiment results
            with open(results_json, 'r') as f:
                exp_results = json.load(f)
            
            # Load model and evaluate
            model, config, sp_urdu, sp_roman = load_experiment_model(exp_dir)
            
            # Temporarily save model for evaluation
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), "models/best_model.pth")
            
            # Run evaluation (this will print results)
            print(f"📊 Results for {exp_name}:")
            evaluate_model()
            
            # Parse results from evaluation output (you might need to modify this)
            # For now, we'll use the training results
            results.append({
                'experiment': exp_name,
                'best_val_loss': exp_results.get('best_val_loss', 'N/A'),
                'best_epoch': exp_results.get('best_epoch', 'N/A'),
                'total_epochs': exp_results.get('total_epochs', 'N/A'),
                'emb_dim': config.get('emb', 'N/A'),
                'hid_dim': config.get('hid', 'N/A'),
                'dropout': config.get('dropout', 'N/A'),
                'lr': config.get('lr', 'N/A'),
                'batch_size': config.get('batch', 'N/A'),
                'scheduler': config.get('scheduler', 'N/A')
            })
            
        except Exception as e:
            print(f"❌ Error evaluating {exp_name}: {e}")
            continue
    
    # Create comparison table
    if results:
        df = pd.DataFrame(results)
        df = df.sort_values('best_val_loss')
        
        print(f"\n📊 Experiment Comparison Table:")
        print("=" * 100)
        print(df.to_string(index=False))
        
        # Save to CSV
        df.to_csv('experiment_comparison.csv', index=False)
        print(f"\n💾 Results saved to: experiment_comparison.csv")
        
        # Find best experiment
        best_exp = df.iloc[0]
        print(f"\n🏆 Best Experiment: {best_exp['experiment']}")
        print(f"   Validation Loss: {best_exp['best_val_loss']:.4f}")
        print(f"   Best Epoch: {best_exp['best_epoch']}")
        print(f"   Config: emb={best_exp['emb_dim']}, hid={best_exp['hid_dim']}, dropout={best_exp['dropout']}")
    else:
        print("❌ No valid experiments found to evaluate.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, help='Evaluate specific experiment')
    args = parser.parse_args()
    
    if args.exp:
        # Evaluate specific experiment
        exp_dir = os.path.join("runs", args.exp)
        if not os.path.exists(exp_dir):
            print(f"❌ Experiment {args.exp} not found in runs/")
            return
        
        print(f"🔍 Evaluating experiment: {args.exp}")
        model, config, sp_urdu, sp_roman = load_experiment_model(exp_dir)
        
        # Save model temporarily for evaluation
        os.makedirs("models", exist_ok=True)
        torch.save(model.state_dict(), "models/best_model.pth")
        
        # Run evaluation
        evaluate_model()
    else:
        # Evaluate all experiments
        evaluate_all_experiments()

if __name__ == "__main__":
    main()
