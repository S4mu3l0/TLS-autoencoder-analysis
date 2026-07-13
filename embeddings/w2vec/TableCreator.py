"""
file: embeddings/w2vec/TableCreator.py
author: Samuel Kudla <xkudlas00@stud.fit.vutbr.cz>
date: 12.05.2026
"""
import json
import os
import sys

import torch
import torch.nn as nn


class TableCreator:
  def __init__(self, delimiter: str, feature_dir_name: str, embedding_dim: int = 16) -> None:
    self.delimiter = delimiter
    self.feature_dir_name = feature_dir_name
    self.embedding_dim = embedding_dim
    base_dir = os.path.dirname(os.path.abspath(__file__))
    self.feature_dir = os.path.join(base_dir, "..", feature_dir_name)

  def _read_existing_values(self) -> list[str]:
    existing_path = os.path.join(self.feature_dir, "existing_vals")
    with open(existing_path, "r") as file:
      line = file.read().strip()

    if not line:
      return []
    return [v.strip() for v in line.split(",")]

  def _merge_training_values(self, vals: list[str]) -> list[str]:
    train_vals_path = os.path.join(self.feature_dir, "train_vals")
    with open(train_vals_path, "r") as file:
      for line in file:
        line = line.strip()
        if not line:
          continue
        seq = line.split(self.delimiter)
        for item in seq:
          if item not in vals:
            vals.append(item)
    return vals

  def createTable(self) -> None:
    vals = self._read_existing_values()
    vals = self._merge_training_values(vals)

    mapping = {e: index for index, e in enumerate(vals)}
    embedding = nn.Embedding(len(vals), self.embedding_dim)

    mapping_path = os.path.join(self.feature_dir, "mapping.json")
    embedding_path = os.path.join(self.feature_dir, "embedding.pt")

    with open(mapping_path, "w") as file:
      json.dump(mapping, file)

    torch.save(embedding.state_dict(), embedding_path)


def _print_usage() -> None:
  print("python3 table_creator.py <delimiter_in_train_vals> <feature_dir_name>")
  print('This script expects files "existing_vals" and "train_vals" in feature directory.')


if __name__ == "__main__":
  if len(sys.argv) < 3 or sys.argv[1] == "-h":
    _print_usage()
    exit(1)

  creator = TableCreator(delimiter=sys.argv[1], feature_dir_name=sys.argv[2])
  creator.createTable()