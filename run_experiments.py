#!/usr/bin/env python3
"""
Run multiple experiments for Urdu to Roman Urdu translation
Based on assignment requirements and your friend's approach
"""

import subprocess
import os
import json
from datetime import datetime

def run_experiment(exp_name, **kwargs):
    """Run a single experiment with given parameters"""
    cmd = ["python", "src/train_experiments.py", "--exp", exp_name]
    
    for key, value in kwargs.items():
        cmd.extend([f"--{key}", str(value)])
    
    print(f"\n🚀 Running experiment: {exp_name}")
    print(f"📋 Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
        if result.returncode == 0:
            print(f"✅ Experiment {exp_name} completed successfully")
            return True
        else:
            print(f"❌ Experiment {exp_name} failed:")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"❌ Error running experiment {exp_name}: {e}")
        return False

def main():
    """Run all experiments as per assignment requirements"""
    
    # Create experiments log
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"experiments_log_{timestamp}.txt"
    
    experiments = [
        # Experiment 1: Base configuration (your friend's approach)
        {
            "name": "base_friend_approach",
            "emb": 256,
            "hid": 512,
            "enc_layers": 2,
            "dec_layers": 4,
            "dropout": 0.3,
            "batch": 64,
            "lr": 5e-4,
            "epochs": 20,
            "patience": 5,
            "scheduler": "plateau",
            "tf_start": 0.5,
            "tf_end": 0.5,
            "tf_decay_epochs": 20,
            "label_smoothing": 0.0,
            "weight_decay": 0.0
        },
        
        # Experiment 2: Improved configuration (our optimizations)
        {
            "name": "improved_config",
            "emb": 512,
            "hid": 512,
            "enc_layers": 2,
            "dec_layers": 4,
            "dropout": 0.5,
            "batch": 64,
            "lr": 1e-3,
            "epochs": 60,
            "patience": 12,
            "scheduler": "onecycle",
            "tf_start": 0.6,
            "tf_end": 0.25,
            "tf_decay_epochs": 20,
            "label_smoothing": 0.1,
            "weight_decay": 1e-5,
            "min_delta": 0.01
        },
        
        # Experiment 3: Larger embedding dimension
        {
            "name": "large_embedding",
            "emb": 512,
            "hid": 256,
            "enc_layers": 2,
            "dec_layers": 4,
            "dropout": 0.5,
            "batch": 64,
            "lr": 1e-3,
            "epochs": 60,
            "patience": 12,
            "scheduler": "onecycle",
            "tf_start": 0.6,
            "tf_end": 0.25,
            "tf_decay_epochs": 20,
            "label_smoothing": 0.1,
            "weight_decay": 1e-5
        },
        
        # Experiment 4: Different hidden sizes
        {
            "name": "hid_256",
            "emb": 256,
            "hid": 256,
            "enc_layers": 2,
            "dec_layers": 4,
            "dropout": 0.3,
            "batch": 64,
            "lr": 1e-3,
            "epochs": 60,
            "patience": 12,
            "scheduler": "onecycle",
            "tf_start": 0.6,
            "tf_end": 0.25,
            "tf_decay_epochs": 20,
            "label_smoothing": 0.1,
            "weight_decay": 1e-5
        },
        
        # Experiment 5: Different dropout rates
        {
            "name": "dropout_01",
            "emb": 512,
            "hid": 512,
            "enc_layers": 2,
            "dec_layers": 4,
            "dropout": 0.1,
            "batch": 64,
            "lr": 1e-3,
            "epochs": 60,
            "patience": 12,
            "scheduler": "onecycle",
            "tf_start": 0.6,
            "tf_end": 0.25,
            "tf_decay_epochs": 20,
            "label_smoothing": 0.1,
            "weight_decay": 1e-5
        },
        
        # Experiment 6: Different learning rates
        {
            "name": "lr_1e4",
            "emb": 512,
            "hid": 512,
            "enc_layers": 2,
            "dec_layers": 4,
            "dropout": 0.5,
            "batch": 64,
            "lr": 1e-4,
            "epochs": 60,
            "patience": 12,
            "scheduler": "onecycle",
            "tf_start": 0.6,
            "tf_end": 0.25,
            "tf_decay_epochs": 20,
            "label_smoothing": 0.1,
            "weight_decay": 1e-5
        },
        
        # Experiment 7: Different batch sizes
        {
            "name": "batch_128",
            "emb": 512,
            "hid": 512,
            "enc_layers": 2,
            "dec_layers": 4,
            "dropout": 0.5,
            "batch": 128,
            "lr": 1e-3,
            "epochs": 60,
            "patience": 12,
            "scheduler": "onecycle",
            "tf_start": 0.6,
            "tf_end": 0.25,
            "tf_decay_epochs": 20,
            "label_smoothing": 0.1,
            "weight_decay": 1e-5
        }
    ]
    
    print(f"🧪 Starting {len(experiments)} experiments...")
    print(f"📝 Log file: {log_file}")
    
    successful_experiments = []
    failed_experiments = []
    
    with open(log_file, 'w') as f:
        f.write(f"Experiment Log - {timestamp}\n")
        f.write("=" * 50 + "\n\n")
        
        for i, exp in enumerate(experiments, 1):
            f.write(f"Experiment {i}: {exp['name']}\n")
            f.write(f"Parameters: {json.dumps(exp, indent=2)}\n")
            f.write("-" * 30 + "\n")
            
            success = run_experiment(**exp)
            
            if success:
                successful_experiments.append(exp['name'])
                f.write("Status: SUCCESS\n\n")
            else:
                failed_experiments.append(exp['name'])
                f.write("Status: FAILED\n\n")
    
    # Summary
    print(f"\n📊 Experiment Summary:")
    print(f"✅ Successful: {len(successful_experiments)}")
    print(f"❌ Failed: {len(failed_experiments)}")
    
    if successful_experiments:
        print(f"\n✅ Successful experiments:")
        for exp in successful_experiments:
            print(f"  - {exp}")
    
    if failed_experiments:
        print(f"\n❌ Failed experiments:")
        for exp in failed_experiments:
            print(f"  - {exp}")
    
    print(f"\n📁 All results saved in 'runs/' directory")
    print(f"📝 Detailed log: {log_file}")

if __name__ == "__main__":
    main()
