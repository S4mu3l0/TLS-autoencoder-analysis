"""
file: model/Undercomplete.py
author: Samuel Kudla <xkudlas00@stud.fit.vutbr.cz>
date: 12.05.2026
"""
import torch.nn as nn
import json
import torch 
import torch.optim as optim 
from pathlib import Path
import sys
import time
from ignite.engine import Engine, Events
from ignite.handlers import EarlyStopping

root_path = (Path(__file__).resolve().parent.parent)
sys.path.append(str(root_path))
embeddings_path = root_path / 'embeddings'
sys.path.append(str(embeddings_path))
thresholds_path = root_path / 'model/thresholds.json'

def get_model_file(config):
    return config['train_model_name'] if config['train_model_name'].endswith('.pth') else (
        config['train_model_name']+'.pth')

def write_config(config_dict):
    with open(str(root_path / 'config.json'), "w") as f:
        json.dump(config_dict, f, indent=4)
    
def load_config():
    with open(str(root_path / 'config.json'), "r") as f:
        config = json.load(f)
    return config

def get_model_topology_and_layers(model):
    layers = []
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            if not layers:
                layers.append(str(m.in_features))
            layers.append(str(m.out_features))
            
    min_val = min(int(x) for x in layers)
    latent_index = layers.index(str(min_val))
    num_layers_to_latent = len(layers[:latent_index + 1])
    
    topology = "-".join(layers)
    return topology, num_layers_to_latent

class Undercomplete(nn.Module):
    def __init__(self, model_dimensions: list, model_name: str = None):
        super(Undercomplete, self).__init__()                
        self.encoder = nn.Sequential()
        self.decoder = nn.Sequential()
        self.epochs_trained = 0
        
        self.name = model_name if model_name else get_model_file(load_config())
                        
        self.encoder.append(nn.Linear(model_dimensions[0], model_dimensions[1]))
        
        # build encoder
        for i in range(1, len(model_dimensions) - 1):
            self.encoder.append(nn.Tanh())
            self.encoder.append(nn.Linear(model_dimensions[i], model_dimensions[i+1]))   
        
        # revert sizes for decoder building
        model_dimensions = model_dimensions[::-1]
        
        # build decoder
        for i in range(len(model_dimensions) - 2):
            self.decoder.append(nn.Linear(model_dimensions[i], model_dimensions[i+1]))
            self.decoder.append(nn.Tanh())
            
        self.decoder.append(nn.Linear(model_dimensions[-2], model_dimensions[-1]))
        
        # loss function for autoencoder is Mean Squared Error, returns mean value
        self.criterion = nn.MSELoss()
        # optimizer is Adam to adapt while counting gradients
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)

    def forward(self, input):
        encoded = self.encoder(input)
        decoded = self.decoder(encoded)
        return decoded
      
    def save(self):
        torch.save(self, self.name)

    def train_undercomplete(self, training_data, epochs = 2000, print_epochs = True):
        def train_step(engine, batch):
            self.train() # set to training mode to update weights
            self.optimizer.zero_grad() # reset gradients
            output = self.forward(batch) # count output of autoencoder
            loss = self.criterion(output, batch) # count MSE
            loss.backward() # set directions for gradients
            self.optimizer.step() # update weights of neurons based on gradients
            self.epochs_trained += 1
            if (print_epochs): print(f'Epoch:{engine.state.epoch}, Loss:{loss.item():.10f}')
            return loss.item()
        
        trainer = Engine(train_step)
        
        # score for early stopping
        def score(engine):
            return -engine.state.output

        if self.early_stop['enabled']:
            es_handler = EarlyStopping(
                patience=self.early_stop['patience'],
                score_function=score,
                trainer=trainer,
                min_delta=self.early_stop['min_delta']
                )
            trainer.add_event_handler(
                Events.EPOCH_COMPLETED(lambda _, e: e > self.early_stop['min_epochs']),
                es_handler
                )
        
        # save model after every 50 epochs
        @trainer.on(Events.EPOCH_COMPLETED(every=50))
        def save_model(engine):
            if (print_epochs): print(f"Epoch: {engine.state.epoch}, Loss: {engine.state.output:.10f} - Saving model...")
            self.save() 
            
        trainer.run([training_data], max_epochs=epochs)
        
        return trainer.state.epoch

def train_model(args: list, print_epochs: bool = False, out_csv_file: str = None):
    config = load_config()
    # set random seed manually to have same train and test split with multiple runs
    
    # get inputs for training
    all_data = torch.load(args[1])
    # count split from config
    split = int(config['train_split'] * len(all_data))
    train_data = all_data[:split]
    test_data = all_data[split:]

    model_name = get_model_file(config)

    if Path(model_name).exists():
        model = torch.load(model_name, weights_only=False)
        print('Existing model loaded from file:', model_name)
        loaded = True
    else:
        print('Model not found, creating new:', model_name)
        model = Undercomplete(config['model_dimensions'], model_name)
        model.save() 
        loaded = False
        
    # set learning rate from config
    model.optimizer.param_groups[0]['lr'] = config['lr']
    model.name = model_name
    model.early_stop = config['early_stopping']
    
    # train
    train_time = time.time()
    trained_epochs = model.train_undercomplete(train_data,epochs = int(args[2]) if len(args) > 2 else 100, print_epochs=print_epochs)
    train_time = time.time() - train_time
    
    model.eval()
    with torch.no_grad():
        test_output = model(test_data)
        test_loss = model.criterion(test_output, test_data)
        
        # set threshold as normal distribution
        all_losses = torch.nn.functional.mse_loss(test_output, test_data, reduction='none').mean(dim=1)
        mean_error = torch.mean(torch.abs(all_losses)).item()
        std_error = torch.std(torch.abs(all_losses)).item()
        
        # set threshold as quantile
        threshold = torch.quantile(all_losses, 0.95).item()*0.9
        model.threshold = threshold
        model.save()
        
        topology, layers = get_model_topology_and_layers(model)
        
        final_train_loss = model.criterion(model(train_data), train_data).item()
        final_test_loss = test_loss.item()
        
        if out_csv_file:
            import os, csv
            if not out_csv_file.endswith('.csv'): 
                out_csv_file = out_csv_file + '.csv'
                
            file_exists = os.path.isfile(out_csv_file)
            
            with open(out_csv_file, mode='a', newline='') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow([
                        'Model', 'Layers', 'Topology', 'Epochs', 
                        'Train_Loss', 'Test_Loss', 'Train_Time', 
                        'Mean_Error', 'Std_Error', 'Threshold', 'New'
                    ])
                
                writer.writerow([
                    model_name, layers, topology, trained_epochs,
                    f"{final_train_loss:.8f}", f"{final_test_loss:.8f}", f"{train_time:.2f}",
                    f"{mean_error:.8f}", f"{std_error:.8f}", f"{threshold}", f"{not loaded}"
                ])
        else:
            print("Training complete, results:")
            print(f"| Trained from file: {args[1]}")
            if loaded: print(f"| Existing model modified")
            print(f"| Modified model: {model_name}")
            print(f"| Topology: {topology}")
            print(f"| Layers: {layers}")
            print(f"| Epochs: {trained_epochs}")
            print(f"| Min delta: {config['early_stopping']['min_delta']}")
            print(f"| Patience: {config['early_stopping']['patience']}")
            print(f"| Train Loss: {final_train_loss:.8f}")
            print(f"| Test Loss (unknown): {final_test_loss:.8f}")
            print(f"| Training time: {train_time:.2f} seconds")
            print(f"| Mean error: {mean_error:.8f}, std_error: {std_error:.8f}")
            print(f"| Setting threshold: {threshold}\n")
            
        with open(str(root_path / thresholds_path), 'r') as f:
            thresholds_file = json.load(f)
            thresholds_file[config['train_model_name']] = threshold
        
        with open(str(root_path / thresholds_path), 'w') as f:
            json.dump(thresholds_file, f, indent=4)

# Running this script will train model defined in config file.
# Model will be saved as .pth file as an object, can be loaded and trained multiple times.
# Division of training and test data for an input data file is set by seed defined in config.
if __name__ == "__main__":
    if (len(sys.argv) < 2 or sys.argv[1] == '-h'):
        print("Usage: python3 Undercomplete.py datafile.pt [epochs] [csv_out_file] [--print_epochs]\nExpecting also config.json file in root.")
        print("Script takes inputed datafile.pt and trains model on provided data. Script divides input data into train and test equally to train_split parameter in config.")
        print("Note: script does not shuffle given dataset.")
        exit(1)
    
    if '--print_epochs' in sys.argv:
        print_epochs = True
        sys.argv.remove('--print_epochs')
    else:
        print_epochs = False
    
    out_csv_file = sys.argv[sys.argv.index(next(filter(lambda x: '.csv' in x, sys.argv)))] if any('.csv' in x for x in sys.argv) else None
    train_model(sys.argv, print_epochs=print_epochs, out_csv_file=out_csv_file)