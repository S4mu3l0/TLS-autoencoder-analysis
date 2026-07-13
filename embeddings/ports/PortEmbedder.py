"""
file: embeddings/ports/PortEmbedder.py
author: Samuel Kudla <xkudlas00@stud.fit.vutbr.cz>
date: 12.05.2026
"""
import sys
from typing import List

class PortEmbedder:
  def __init__(self, max_port: int = 65535, weight_factor: float = 0.4) -> None:
    self.max_port = max_port
    self.weight_factor = weight_factor
    # weight factor to decrease the importance of higher digits
    # smaller the factor, more importance to lower digits
    # count weights for each digit
    self.max_port_embedding = self._compute_max_port_embedding()

  def _get_weights(self, N: int) -> List[float]:
    weights = []
    for i in range(1, N + 1):
      numerator = (2 ** (N - i)) * (3 ** ((N + 1 - i) * (N + 2 - i) ** self.weight_factor))
      denominator = N * (N + 1) * (N + 2)
      weights.append(numerator / denominator)
    return weights

  def _compute_max_port_embedding(self) -> float:
    """This function computes the maximum possible embedding value for the largest port number (65535).
    This is used to normalize the embedding values to a range of [0, 1]."""
    max_port_str = str(self.max_port)
    weights = self._get_weights(len(max_port_str))
    return sum([int(d) * w for d, w in zip(max_port_str, weights)])

  def get_port_embedding(self, number: int) -> float:
    digits = [int(d) for d in str(number)]
    N = len(digits)
    weights = self._get_weights(N)
    # count embedding value
    val = sum([d * w for d, w in zip(digits, weights)])
    return val / self.max_port_embedding

if __name__ == "__main__":
  if (len(sys.argv) < 2 or '-h' in sys.argv or '--help' in sys.argv):
    print("Usage: python3 PortEmbedder.py <port_number>")
    print("Embed a port number into a normalized vector.")
    print("\nArguments:")
    print("  port_number        Port number to embed (0-65535)")
    print("\nOptions:")
    print("  -h, --help         Show this help message")
    exit(1)
  number = int(sys.argv[1])
  embedder = PortEmbedder()
  curr_port_embedding = embedder.get_port_embedding(number)
  print( curr_port_embedding )