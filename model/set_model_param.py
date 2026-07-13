"""
file: model/set_model_param.py
author: Samuel Kudla <xkudlas00@stud.fit.vutbr.cz>
date: 12.05.2026
"""
import sys
import torch
from Undercomplete import Undercomplete

def set_model_param(model_file, param_name, param_value):
    
    try:
        param_value = float(param_value)
    except ValueError:
        pass  # Keep as string if it cannot be converted to float
    
    model = torch.load(model_file, map_location=torch.device('cpu'), weights_only=False)
    if not isinstance(model, Undercomplete):
        print(f"Error: The loaded model from {model_file} is not an instance of Undercomplete.")
        return
    
    model.__setattr__(param_name, param_value)
    model.save()
    print(f"Parameter '{param_name}' set to {param_value} in model '{model_file}'.")

if __name__ == "__main__":
    if len(sys.argv) < 4 or sys.argv[1] == '-h':
        print("Usage: python set_model_param.py <model_file.pth> <model_param> <value>")
        print("Example: python set_model_param.py embedding_test/AirDroid/182-140-98-56-14.pth threshold 0.002")
        print("Note: changing the model name will result in saving a new entity.")
        sys.exit(1)
    
    set_model_param(sys.argv[1], sys.argv[2], sys.argv[3])