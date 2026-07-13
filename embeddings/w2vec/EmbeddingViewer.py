"""
file: embeddings/w2vec/EmbeddingViewer.py
author: Samuel Kudla <xkudlas00@stud.fit.vutbr.cz>
date: 12.05.2026
"""
import torch 
import torch.nn as nn 
import json
import sys
import os
import torch.nn.functional as F

_MODELS_CACHE = {}

class EmbeddingViewer:
    def __init__(self, embedding_dim: int = 16) -> None:
        self.embedding_dim = embedding_dim
        self.base_path = os.path.dirname(os.path.abspath(__file__))

    def _load_model(self, parameter_dir: str) -> dict | None:
        """
        @param parameter_dir: directory where the embedding model and mapping are stored, expected to contain 'mapping.json' and 'embedding.pt'
        
        Loads the model and mapping to indexes from the specified parameter directory, utilizing caching for efficiency (used in test_model_directory.py).
        """
        if parameter_dir in _MODELS_CACHE:
            return _MODELS_CACHE[parameter_dir]

        mapping_path = os.path.join(self.base_path, "..", parameter_dir, "mapping.json")
        model_path = os.path.join(self.base_path, "..", parameter_dir, "embedding.pt")

        if not os.path.exists(mapping_path) or not os.path.exists(model_path):
            return None

        with open(mapping_path, "r") as file:
            mapping = json.load(file)

        num_embeddings = len(mapping)
        embeddings = nn.Embedding(num_embeddings, self.embedding_dim)
        embeddings.load_state_dict(torch.load(model_path, weights_only=True))
        embeddings.eval()

        cache = {"mapping": mapping, "embeddings": embeddings}
        _MODELS_CACHE[parameter_dir] = cache
        return cache

    def get_average_embedding(self, input_vals: str, delimiter: str, parameter_dir: str) -> torch.Tensor | None:
        """
        @param input_vals: string containing the input values to be embedded, expected to be separated by the specified delimiter
        @param delimiter: string used to separate individual values in input_vals   
        @param parameter_dir: directory where the embedding model and mapping are stored, expected to contain 'mapping.json' and 'embedding.pt'
        
        Function calculating the average embedding for a list of input values.
        """
        cache = self._load_model(parameter_dir)
        if cache is None:
            print(f"Error: Could not load model from directory '{parameter_dir}'. Please ensure the directory exists and contains 'mapping.json' and 'embedding.pt'.")
            return None

        mapping = cache["mapping"]
        embeddings = cache["embeddings"]

        vals = input_vals.strip().split(delimiter)
        vals = [val.strip() for val in vals]
        vecs = []

        with torch.no_grad():
            for val in vals:
                if val in mapping:
                    index = torch.tensor([mapping[val]], dtype=torch.long)
                    vecs.append(embeddings(index))

        if not vecs:
            return None
        vecs = torch.vstack(vecs)
        summed_vec = vecs.sum(dim=0)
        normalized_vec = F.normalize(summed_vec, p=2, dim=0)

        max_lengths = {
            "extensions": 30.0,
            "ciphersuite": 40.0,
            "supportedgroup": 10.0,
        }
        norm_val = None
        for key in max_lengths:
            if parameter_dir in key:
                norm_val = max_lengths[key]
                break
        if norm_val is None:
            print(f"Warning: No specific normalization value found for parameter directory '{parameter_dir}'. Using default normalization value of 20.0.")
            norm_val = 20.0

        len_val = torch.tensor([len(vals) / max_lengths[parameter_dir]])

        return torch.cat([normalized_vec, len_val])

if __name__ == "__main__":
    if len(sys.argv) < 4 or '-h' in sys.argv or '--help' in sys.argv:
        print('Usage: python3 show_embedding.py <value> <delimiter> <dir_of_model>')
        print("Get the average embedding for a list of values.")
        print("\nArguments:")
        print("  value              Value(s) to embed (separated by delimiter)")
        print("  delimiter          Character separating values")
        print("  dir_of_model       Directory (extensions/ciphersuite/supportedgroup) containing embedding model files (.pt and mapping.json)")
        print("\nOptions:")
        print("  -h, --help         Show this help message")
        exit(1)
    viewer = EmbeddingViewer()
    result = viewer.get_average_embedding(sys.argv[1], sys.argv[2], sys.argv[3])
    print(result)