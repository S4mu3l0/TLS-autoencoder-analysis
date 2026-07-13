"""
file: embeddings/compare.py
author: Samuel Kudla <xkudlas00@stud.fit.vutbr.cz>
date: 12.05.2026
"""
import torch
import sys
from ConnectionEmbedder import ConnectionEmbedder

embedder = ConnectionEmbedder()

def compare_tensors(ten1, ten2):
  if ten1.shape != ten2.shape:
    print("Tensors have different shapes:", ten1.shape, ten2.shape)
    return
  
  for i in range(ten1.shape[0]):
    if ten1[i].item() != ten2[i].item():
      print(f"{i}: {ten1[i].item()} vs {ten2[i].item()}")
  print("Comparison complete.")

if __name__ == "__main__":
  if (len(sys.argv) < 3 or '-h' in sys.argv or '--help' in sys.argv):
    print("Usage: python3 compare.py <file1> <file2>")
    print("Compare embeddings from two connection files.")
    print("\nArguments:")
    print("  file1              Path to first connections file")
    print("  file2              Path to second connections file")
    print("\nOptions:")
    print("  -h, --help         Show this help message")
    exit(1)
  
  conns1 = embedder.get_embedding_for_file_connections(sys.argv[1])
  conns2 = embedder.get_embedding_for_file_connections(sys.argv[2])
  ten1 = next(iter(conns1.values()))['Embedding']
  ten2 = next(iter(conns2.values()))['Embedding']
  compare_tensors(ten1, ten2)