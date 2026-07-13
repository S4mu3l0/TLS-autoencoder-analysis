"""
file: embeddings/w2vec/EmbeddingDistanceTester.py
author: Samuel Kudla <xkudlas00@stud.fit.vutbr.cz>
date: 12.05.2026
"""
import torch
import torch.nn as nn
import json
import os
import sys
from EmbeddingViewer import EmbeddingViewer

class EmbeddingDistanceTester:
    def __init__(self, path: str = "extensions", embedding_dim: int = 16, delim: str = "") -> None:
        self.path = path
        self.embedding_dim = embedding_dim
        self.delim = delim
        self.base_path = os.path.dirname(os.path.abspath(__file__))
        self.mapping = self._load_mapping()
        self.embedding = self._load_embedding()

    def _load_mapping(self):
        mapping_path = os.path.join(self.base_path, "..", self.path, "mapping.json")
        with open(mapping_path, "r") as file:
            return json.load(file)

    def _load_embedding(self):
        embedding = nn.Embedding(len(self.mapping), self.embedding_dim)
        model_path = os.path.join(self.base_path, "..", self.path, "embedding.pt")
        embedding.load_state_dict(torch.load(model_path))
        embedding.eval()
        return embedding

    def distance(self, first_value: str, second_value: str) -> float:
        if not self.delim:
            first_vector = self.embedding(torch.tensor([self.mapping[first_value]]))
            second_vector = self.embedding(torch.tensor([self.mapping[second_value]]))
        else:
            viewer = EmbeddingViewer(embedding_dim=self.embedding_dim)
            first_vector = viewer.get_average_embedding(first_value, self.delim, self.path)
            second_vector = viewer.get_average_embedding(second_value, self.delim, self.path)
        return torch.norm(first_vector - second_vector, p=2).item()

if __name__ == "__main__":
    if len(sys.argv) < 4 or '-h' in sys.argv or '--help' in sys.argv:
        print("Usage: python3 test.py <value1> <value2> <directory_of_model> [delimiter]")
        print("Calculate distance between two embeddings.")
        print("\nArguments:")
        print("  value1             First value to compare")
        print("  value2             Second value to compare")
        print("  directory_of_model Directory containing embedding model files (.pt and mapping.json)")
        print("  delimiter          Optional delimiter for multi-value embeddings")
        print("\nOptions:")
        print("  -h, --help         Show this help message")
        exit(1)

    delim = sys.argv[4] if len(sys.argv) > 4 else ""
    tester = EmbeddingDistanceTester(path=sys.argv[3], delim=delim)
    try:
        dist = tester.distance(sys.argv[1], sys.argv[2])
        print(f"Distance between {sys.argv[1]} and {sys.argv[2]}: {dist:.10f}")
    except KeyError as e:
        print(f"Error: {e} not found in mapping. Please ensure the values and delimiter are correct and exist in the model's mapping.")