"""
file: model/test_model_directory.py
author: Samuel Kudla <xkudlas00@stud.fit.vutbr.cz>
date: 12.05.2026
"""
import sys
import copy
from Undercomplete import load_config, write_config, Undercomplete
from model.evaluate import evaluate
from pathlib import Path
import os

out_dir = 'test_results/'
origin_stdout = sys.stdout


def test_multiple_models(args):
  """
  @param args: command line arguments (starting from index 1), expects at least directory of models and reference raw data file, 
  optionally output filename and destination app name(s)
  
  Function iterates through models in a directory, temporarily sets 'test_models_name' in config for each model, and calls evaluate() 
  to test the model using Classifier. Results are saved to output file. If test_result directory does not exist, it is created. After testing all models, original config is restored.
  """
  # Create test_results directory if it doesn't exist
  if not os.path.exists(out_dir):
      os.makedirs(out_dir)
      print(f"Created dir {out_dir} for test results.") 
  
  print("Initialized testing for multiple model variants using Classifier...")
  config = load_config()
  old_config = copy.deepcopy(config)

  models_dir_name = args[1]
  test_file = args[2]
  out_file = sys.argv[3] if len(sys.argv) > 3 else 'results'

  # Optionally override destination app(s)
  if (len(sys.argv) > 4):
    config['dest_app_name'] = sys.argv[4:]

  output = out_dir + out_file
  output_csv = output + '.csv'

  dir_path = Path(models_dir_name)

  for model_path in dir_path.iterdir():
    model_path = str(model_path)
    if not model_path.endswith('.pth'):
      continue

    # Build a temporary mapping for test models to use with Classifier
    tmp_config = copy.deepcopy(old_config)

    # If original config already contains a mapping for this model, reuse it
    mapped_name = None
    if isinstance(old_config.get('test_models_name'), dict):
      mapped_name = old_config['test_models_name'].get(model_path)

    # Fallback to dest_app_name from config (use first if list)
    if not mapped_name:
      print("No existing mapping for model", model_path, "found in config, using dest_app_name as fallback.")
      dest = tmp_config.get('dest_app_name')
      if isinstance(dest, list):
        mapped_name = dest[0] if dest else ''
      else:
        mapped_name = dest

    tmp_config['test_models_name'] = { model_path: mapped_name }
    write_config(tmp_config)

    f = open(output, 'a')
    sys.stdout = f

    evaluate(test_file, out_csv_file=output_csv)
    print()

    sys.stdout = origin_stdout
    print("Tested model", model_path, "...")

  write_config(old_config)
  print('Done!')

if __name__ == "__main__":
  if (len(sys.argv) < 3 or sys.argv[1] == '-h'):
    print("Usage: python test_multiple_models.py <directory_of_models> <reference_raw_data_file> [output_filename] [[dest_app_name]]")
    print("This script tests each model in a directory by temporarily setting 'test_models_name' and calling evaluate() (uses Classifier).")
    exit(1)

  test_multiple_models(sys.argv)