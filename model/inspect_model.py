"""
file: model/inspect_model.py
author: Samuel Kudla <xkudlas00@stud.fit.vutbr.cz>
date: 12.05.2026
"""
import torch
import os
from Undercomplete import Undercomplete

def inspect_pth_file(file_path):
    if not os.path.exists(file_path):
        print(f"Error: file '{file_path}' does not exist.")
        return

    print(f"{'='*50}")
    print(f"INSPECTION OF MODEL: {file_path}")
    print(f"{'='*50}\n")

    try:        
        model = torch.load(file_path, weights_only=False)
        print(f"Name (model.name): {getattr(model, 'name', 'N/A')}")
        print(f"Type of object: {type(model)}")
        
        print("ENCODER:")
        print(model.encoder)
        print("\nDECODER:")
        print(model.decoder)

        state_dict = model.state_dict()
        for layer_name, weights in state_dict.items():
            print(f"Layer: {layer_name:30} | Shape: {str(list(weights.shape)):15}")
        if hasattr(model, 'epochs_trained'):
            print(f"\nEpochs trained: {model.epochs_trained}")
        print(f"Loss function: {model.criterion}")
        print(f"Optimizer: {type(model.optimizer).__name__}")
        print(f"Learning Rate: {model.optimizer.param_groups[0]['lr']}")
        print(f"Threshold: {getattr(model, 'threshold', 'N/A')}")
        if hasattr(model, 'early_stop'):
            print(f"Early Stopping settings: {model.early_stop}")

    except Exception as e:
        print(f"Error occurred while loading the model: {e}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2 or '-h' in sys.argv:
        print("Usage: python3 inspect_model.py <model_path> (.pth file)")
    else:
        inspect_pth_file(sys.argv[1])