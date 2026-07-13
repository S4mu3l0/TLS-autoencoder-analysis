"""
file: model/evaluate.py
author: Samuel Kudla <xkudlas00@stud.fit.vutbr.cz>
date: 12.05.2026
"""
import csv
import math
import os
import sys
from Classifier import Classifier
from show_loss import load_config
import torch
from Undercomplete import Undercomplete
import json

def evaluate(connections_file: str, out_csv_file: str = None):
    """
    @param connections_file: path to file with connections, can be csv or txt in format from dataset_show script
    @param out_csv_file: optional path to csv file to save results, if not provided results will be printed to console
    
    Returns evaluation results for connections in file, including precision, recall, FPR, specificity, MCC and accuracy. Uses Classifier to classify connections and get losses.
    
    Results can be saved to csv file if out_csv_file parameter is provided, otherwise results will be printed to console. For details on expected file format, see dataset_show 
    script and ConnectionEmbedder
    """
    config = load_config()
    
    dest_app = config['dest_app_name']
    if not isinstance(dest_app, list):
      dest_app = [dest_app]
    
    print(f"Searching for apps: {', '.join(dest_app)}")
    
    model_count = len(config['test_models_name'])
    print(f"Number of models used for classification: {model_count}")
    
    classifier = Classifier(print_results=False)
    classified_connections = classifier.classify_multiple_connections(connections_file)
    
    destination_app_num = 0
    
    false_negative = 0
    false_positive = 0
    true_positive = 0
    true_negative = 0
    wrong_model = 0
    
    for connection in classified_connections:
      predicted_model = connection['NearestModel']
      actual_app = connection['AppName']

      # Classified as known app (positive result)
      if predicted_model:          
          # Predicted app BASED ON MODEL that predicted it
          predicted_app_name = config['test_models_name'][predicted_model]
          
          # App is searched one
          if actual_app in dest_app:
              destination_app_num += 1
              # Correct model predicted it
              if predicted_app_name == actual_app:
                  true_positive += 1
              else:
                  # Correct app but identified by wrong model
                  wrong_model += 1
                  false_negative += 1 # If wrong model, but destination app. the destination model made mistake
          # Not destination app - foreign or other model's app
          else:
              # Models identified correctly app, but it's not destination, so we know it's foreign app
              # System distinguished dest from other learned
              if predicted_app_name == actual_app:
                true_negative += 1
              # System identified app as known, but the best model made mistake
              else:
                # If destination app model made mistake, count as mistake
                if predicted_app_name in dest_app:
                    false_positive += 1
                # If non-destination app model made mistake, count as correct foreign app classification
                else:
                    true_negative += 1 # Even if model is wrong, final classification is FOREIGN (not destination app)
    # Classified as foreign app (negative result)
      else:
          if actual_app in dest_app:
              destination_app_num += 1
              false_negative += 1
          else:
              true_negative += 1
    
    precision_denom = true_positive + false_positive
    precision = true_positive / precision_denom if precision_denom > 0 else 0
    
    recall_denom = true_positive + false_negative
    recall = true_positive / recall_denom if recall_denom > 0 else 0
    
    foreign_total = false_positive + true_negative
    false_positive_rate = false_positive / foreign_total if foreign_total > 0 else 0
    specificity = true_negative / foreign_total if foreign_total > 0 else 0
    
    f1 = 2 * ((precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0
    
    mcc_denom = math.sqrt((true_positive + false_positive) * (true_positive + false_negative) * (true_negative + false_positive) * (true_negative + false_negative))
    mcc = (true_positive * true_negative - false_positive * false_negative) / mcc_denom if mcc_denom > 0 else 0
    
    total = len(classified_connections)
    
    if out_csv_file:
        if not out_csv_file.endswith('.csv'): out_csv_file = out_csv_file+'.csv'
        file_exists = os.path.isfile(out_csv_file)
            
        with open(out_csv_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            
            # Write header if first time
            if not file_exists:
                writer.writerow([
                    'Total', 'Foreign', 'Destination', 
                    'Accuracy', 'Precision', 'Recall', 'Specificity', 'FPR', 'F1', 'MCC', 'Models'
                ])
            
            # write data
            writer.writerow([
                total,
                total - destination_app_num,
                destination_app_num,
                f"{((true_positive + true_negative) / total * 100):.2f}",
                f"{precision:.8f}",
                f"{recall:.8f}",
                f"{specificity:.8f}",
                f"{false_positive_rate:.8f}",
                f"{f1:.8f}",
                f"{mcc:.8f}",
                f"{model_count}"
            ])
        print(f"| True positives: {true_positive}")
        print(f"| True negatives: {true_negative}")
        print(f"| False positives: {false_positive}")
        print(f"| False negatives: {false_negative}")
        
    else:
        print(f"No csv file name inputed...")
        print(f"| Total connections: {total}")
        print(f"| Correctly classified: {true_positive + true_negative}")
        print(f"| Wrong model: {wrong_model}")
        print(f"| False positives: {false_positive}")
        print(f"| False negatives: {false_negative}")
        print(f"| True positives: {true_positive}")
        print(f"| True negatives: {true_negative}")
        print(f"| Precision: {precision:.8f}")
        print(f"| Recall: {recall:.8f}")
        print(f"| FPR: {false_positive_rate:.8f}")
        print(f"| Specificity: {specificity:.8f}")
        print(f"| MCC: {mcc:.8f}")
        print(f"| Accuracy: {(true_positive + true_negative) / total * 100:.8f}%")
        print(f"| F1 score: {f1:.8f}\n")
        print(f"| Models used: {model_count}")
    
if __name__ == "__main__":
  if len(sys.argv) < 2:
    print("Usage: python3 evaluate.py <connections_file> [out_csv_file]\n")
    print("This script tests the evaluation of connections from a file and calculates statistics.")
    print("Requires configuration in config file, where 'dest_app_name' contains destination app names where order depends on order of 'test_models_name' - model at some index is meant to classify connection in 'dest_app_name' at same index.")
    print("Don't forget to set dest_app_name in config as expected in raw file (e.g. 'AirDroidAirDroid' for AirDroid app)")
    exit(1)
  
  evaluate(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)