"""
file: data/shuffle_csv.py
author: Samuel Kudla <xkudlas00@stud.fit.vutbr.cz>
date: 12.05.2026
"""
import pandas as pd
import sys

if len(sys.argv) < 2 or '-h' in sys.argv or '--help' in sys.argv:
  print("Usage: python shuffle_csv.py <csv_file> [output_file]")
  print("Shuffle rows of a CSV file randomly.")
  print("\nArguments:")
  print("  csv_file       Path to input CSV file")
  print("  output_file    Optional output filename (default: shuffled.csv)")
  print("\nOptions:")
  print("  -h, --help     Show this help message")
  exit(1)
  
out_file = sys.argv[2] if len(sys.argv) > 2 else "shuffled.csv"
df = pd.read_csv(sys.argv[1], sep=';')
# use random seed 42 for reproducibility
df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
df_shuffled.to_csv(out_file, sep = ';' ,index=False)