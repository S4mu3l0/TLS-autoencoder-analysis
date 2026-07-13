"""
file: embeddings/SNI/SNIEmbedder.py
author: Samuel Kudla <xkudlas00@stud.fit.vutbr.cz>
date: 12.05.2026
"""
from sklearn.feature_extraction.text import HashingVectorizer 
import sys
import torch 
import numpy as np 
import math
from collections import Counter

max_entropy_val = 5.25 # log2(38) - chars possible in SNI

class SNIEmbedder:
  def __init__(self, n_features: int = 128, ngram_range: tuple[int, int] = (3, 3), sni_len_normalizer: float = 64.0) -> None:
    self.n_features = n_features
    self.ngram_range = ngram_range
    self.sni_len_normalizer = sni_len_normalizer

  def _shannon_entropy(self, txt):
    """Function counts entropy based on Shannon's formula.

    Probability is counted by number of occurences in word divided by length of word.
    """
    counts = Counter(txt)
    length = len(txt)
    if length == 0: return 0.0

    # count sum
    entropy = 0
    for count in counts.values():
      pst_i = count / length
      entropy -= pst_i * math.log2(pst_i)

    normalized_entropy = min(entropy / max_entropy_val, 1.0)
    return normalized_entropy

  def get_hash_vector(self, data_arr) -> torch.Tensor:
    """Function embedding SNI to hashed vector using hashing trick.

    Function also adds entropy of chars.
    Note: input is array of strings, for one string need to wrap it in array.
    """
    # setup vectorizer
    h_vectorizer = HashingVectorizer(
      n_features = self.n_features,
      analyzer='char',
      ngram_range=self.ngram_range,
      alternate_sign = True # set randomly but deterministically sign of value to avoid collisions
    )

    tranformed = h_vectorizer.transform(data_arr).toarray()

    # add entropy and length
    extra_features = []
    for sni in data_arr:
        ent = self._shannon_entropy(sni)
        length_norm = len(sni) / self.sni_len_normalizer
        extra_features.append([ent, length_norm])
    ef_np = np.array(extra_features)

    combined = np.hstack((tranformed, ef_np))

    transformed_tensor = torch.from_numpy(combined).float()

    return transformed_tensor



# script fetches all SNI in array and stores in
if __name__ == '__main__':
  if '-h' in sys.argv or '--help' in sys.argv or len(sys.argv) < 2:
    print("Usage: python3 embed_sni.py <sni_file/SNI> --inline")
    print("Script loads all SNI values from specified file, computes their embeddings using SNIEmbedder, and saves the resulting embeddings along with original SNI values in 'embedding.pt' file.")
    print("If --inline flag is provided, the script prints embedding of given SNI, else prints embedding for SNI values in file.")
    sys.exit(0)

  if '--inline' in sys.argv:
    sys.argv.remove('--inline')
    embedder = SNIEmbedder()
    file = sys.argv[1]
    embedding = embedder.get_hash_vector([file]).clone().detach().float().view(-1)
    print(f"SNI: {file}, Embedding: {embedding.tolist()}")
    sys.exit(0)
    
  data_arr = []
  file = sys.argv[1]
  with open(file, "r") as data:
    for line in data:
      line = line.strip()
      if line:
        data_arr.append(line)

  if not data_arr:
    print("No data found in file.")
    sys.exit(1)

  embedder = SNIEmbedder()
  transformed_tensor = embedder.get_hash_vector(data_arr)

  for sni, embedding in zip(data_arr, transformed_tensor):
    print(f"SNI: {sni}, Embedding: {embedding.tolist()}")