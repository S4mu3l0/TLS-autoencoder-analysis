"""
file: model/train_model_variants.py
author: Samuel Kudla <xkudlas00@stud.fit.vutbr.cz>
date: 12.05.2026
"""
import sys
import os
import copy
from Undercomplete import Undercomplete, load_config, write_config, train_model
import itertools
import math

out_dir = 'train_results/'
origin_stdout = sys.stdout

def get_filename(config: dict, model_struct: list) -> str:
  """
  @param config: config dict containing parameters for training
  @param model_struct: list of integers representing the structure of the model (number of neurons in each layer)
     
  Function getting filename for model based on its structure and config parameters. Used to save model with correct name.
  """
  suffix = '-'.join([str(x) for x in model_struct]) + '.pth'
  filename = ''
  if config['proportional_scaling']:
    filename = out_dir + 'prop/' + str(model_struct[len(model_struct) - 1]) + '/' + suffix
  else:
    filename = out_dir + 'cstm/' + str(model_struct[len(model_struct) - 1]) + '/' + suffix
  
  return filename

def train_model_struct(model_struct: list, app_train_file_embeddings: str, min_delta: float, all_train_file_embeddings: str = None, config: dict = load_config()) -> None:
  """
  @param model_struct: list of integers representing the structure of the model (number of neurons in each layer)
  @param app_train_file_embeddings: path to file with training data for destination app, in
  @param min_delta: minimum delta for early stopping, used to determine when to stop training based on loss improvement
  @param all_train_file_embeddings: path to file with training data for all apps, used for pretraining before fine-tuning on destination app
  @param config: config dict containing parameters for training, used to get other parameters and save
  
  Function training model with given structure. If all_train_file_embeddings is provided, model is pretrained on all apps before fine-tuning on destination app. 
  Model is saved with name based on its structure and config parameters.
  """
  suffix = '-'.join([str(x) for x in model_struct]) + '.pth'
  # write to config new structure of model
  config['model_dimensions'] = model_struct
  config['train_model_name'] = get_filename(config, model_struct)
  
  base_path = out_dir + ('prop/' if config['proportional_scaling'] else 'cstm/') + str(model_struct[-1]) + '/'
  txt_path = base_path + 'results'
  csv_path = base_path + 'results.csv'
  
  # delta for early stopping in pretraining set to be higher -> no need for higher epoch count
  phase1_delta = max(min_delta * 50, config['early_stopping']['all_apps_fine_tune'])
  
  # train all apps first
  config['lr'] = 0.001
  config['early_stopping']['min_delta'] = phase1_delta
  config['early_stopping']['patience'] = 50
  config['early_stopping']['min_epochs'] = 50
  
  args = ['', all_train_file_embeddings, 150]
  write_config(config)
  
  f = open(txt_path, 'a')
  
  sys.stdout = f

  # pretraining on all apps if file provided, otherwise train only on destination app
  if all_train_file_embeddings:
    train_model(args, print_epochs=False, out_csv_file=csv_path)

  # fine tuning training on destination app 
  config['lr'] = 0.0001
  config['early_stopping']['min_delta'] = min_delta
  # for higher models need to modify ES min_epochs and patience for quality training time
  config['early_stopping']['min_epochs'] = 200 if len(model_struct) < 6 else 600
  inner_layers = len(model_struct) - 2
  config['early_stopping']['patience'] = 20 + (inner_layers * 5)  
  args = ['', app_train_file_embeddings, 2000]
  write_config(config)
  
  train_model(args, print_epochs=False, out_csv_file=csv_path)
  f.flush()
  sys.stdout = origin_stdout
  
  f.close()  
  print(f"Trained variant {suffix}...")

def train_model_variants(app_train_file_embeddings: str, all_train_file_embeddings: str = None) -> None:
  """
  @param app_train_file_embeddings: path to file with training data for destination app, in format of train file embeddings created by make_train_file_embeddings.py script
  @param all_train_file_embeddings: path to file with training data for all apps, used for pretraining before fine-tuning on destination app, in format of train file embeddings created by make
  
  Function training multiple variants of an undercomplete autoencoder models. Functions stroes results in train_results/ directory, creating subdirectories for proportional scaling and custom layers variations. 
  For each model structure defined in config, model is trained and results are saved in txt and csv files. Model is pretrained on all apps if all_train_file_embeddings is provided, otherwise trained only on destination app. 
  Early stopping parameters are modified based on model structure for better training time and quality.
  """
  
  config = load_config()
  old_config = copy.deepcopy(config)
    
  input_size = config['input_size']
  latent_size = config['latent_size']
  base_delta = config['early_stopping']['starting_delta']
  # Create model by propotional scaling
  if config['proportional_scaling']:
    model_count = config['max_layers'] - 1
    
    diff = input_size - latent_size
    
    if not os.path.exists(out_dir + 'prop/' + str(latent_size) ):
      os.makedirs(out_dir + 'prop/' + str(latent_size)  )
      print(f"Created dir {out_dir + 'prop/' +  str(latent_size) }")  
    
    # create model for each layers variations
    # beginning from simple input -> latent -> output (same dim as input)
    for model_inner_layers in range(model_count):  
      # count linear delta difference between each inner layer of autoencoder model
      delta = int(diff / (model_inner_layers + 1))
      # for each layer variation create structure
      inner_layers = [input_size - (x * delta)  for x in range(1, model_inner_layers + 1)]
      model_struct = [input_size] + inner_layers + [latent_size]
      dynamic_min_delta = base_delta / math.pow(len(inner_layers) + 1, 0.5)
      train_model_struct(model_struct, app_train_file_embeddings, dynamic_min_delta, all_train_file_embeddings, config)
  
  # Create variations of layers defined in config
  else:
    if not os.path.exists(out_dir + 'cstm/' + str(latent_size) ):
      os.makedirs(out_dir + 'cstm/' + str(latent_size))
      print(f"Created dir {out_dir + 'cstm/' + str(latent_size)}")  

    requested_layers = config['mid_layers_to_test']
    all_models_structures = []
    
    if config['combine_dimensions']:
      for comb_len in range(1, len(requested_layers) + 1):
        combinations_r = list(itertools.combinations(requested_layers, comb_len))
        all_models_structures.extend(combinations_r)
    else:
      for i in range(0, len(requested_layers) + 1):
        prefix_combination = tuple(requested_layers[:i])
        all_models_structures.append(prefix_combination)
    
    for structure in all_models_structures:
      structure = list(structure)
      structure.sort(reverse=True)
      dynamic_min_delta = base_delta / math.pow(len(structure) + 1, 0.5)
      structure = [input_size] + structure + [latent_size]
      train_model_struct(structure, app_train_file_embeddings, dynamic_min_delta, all_train_file_embeddings, config)
    
  write_config(old_config)
  print('Done!')
  
if __name__ == "__main__":
  if (len(sys.argv) < 2 or sys.argv[1] == '-h'):
    print("Usage: python train_multiple_models.py <AppFile.pt> [AllDataFile.pt].\nScript loads parameters from config.")
    print("Warning: script modifies config.json file. Interupting script will result in change of data in config.")
    exit(0)
  
  if len(sys.argv) == 2:
    train_model_variants(sys.argv[1], None)
  else:
    train_model_variants(sys.argv[1], sys.argv[2])