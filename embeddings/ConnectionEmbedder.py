import csv
import json
import os
import sys
from typing import Any

import torch 
from SNI.SNIEmbedder import SNIEmbedder
from ports.PortEmbedder import PortEmbedder
from w2vec.EmbeddingViewer import EmbeddingViewer

embedded_vals = ['SNI', 'SrcPort', 'Client CipherSuite', 'Client Extensions', 'Client Supported Groups']

class ConnectionEmbedder:
  def __init__(self) -> None:
    self.embedding_viewer = EmbeddingViewer()
    self.port_embedder = PortEmbedder()
    self.sni_embedder = SNIEmbedder()

  def _embed_value(self, name: str, value: str) -> torch.Tensor:
    """
    @param name: name of connection parameter, must be one of embedded_vals list
    @param value: value of connection parameter, string representation of parameter value
    
    Returns embedding vector for given parameter value, concatenated if multiple values.
    
    Function embedding value of connection parameter based on its name. Function uses classes defining
    embedding for values in embedded_vals list.
    """
    value = value.strip()
    if name == 'SrcPort':
      return torch.tensor(self.port_embedder.get_port_embedding(int(value))).float().view(-1)

    if name == 'SNI':
      return self.sni_embedder.get_hash_vector([value]).clone().detach().float().view(-1)

    if name == 'Client Supported Groups':
      res = self.embedding_viewer.get_average_embedding(value, ',', 'supportedgroup')
    elif name == 'Client CipherSuite':
      res = self.embedding_viewer.get_average_embedding(value, '-', 'ciphersuite')
    elif name == 'Client Extensions':
      res = self.embedding_viewer.get_average_embedding(value, '-', 'extensions')

    if res is None:
      raise ValueError(f"Could not embed value for '{name}'")

    return res.clone().detach().float().view(-1)

  def _merge_vector_dict(self, transformed: dict[str, torch.Tensor]) -> torch.Tensor:
    """
    @param transformed: dictionary with embedding vectors for connection parameters, keys are parameter names,
    values are embedding vectors.
    
    Returns merged vector with embedding vectors concatenated in order defined by embedded_vals list
    
    Function merging vectors from transformed dictionary into single vector in order defined by embedded_vals list, 
    raise ValueError if any value is missing"""
    arr = []
    for name in embedded_vals:
      if name not in transformed:
        raise ValueError(f"Missing embedding for '{name}'")
      arr.append(transformed[name])
    return torch.cat(arr, dim=0)

  def _embed_from_csv(self, source: str, print_apps: bool) -> dict[str, dict[str, Any]]:
    """
    @param source: path to csv file with connections, expects delimiter ';' and first line as headers
    @param print_apps: if True, print app name for each connection
    
    Returns dictionary with connection names as keys and dictionaries with connection parameters and
    embedding vector under 'Embedding' key as values
    
    Function embedding all connections from a csv file into dictionary.
    """
    connections: dict[str, dict[str, Any]] = {}
    with open(source, 'r') as file:
      reader = csv.reader(file, delimiter=';')
      headers = next(reader)
      index = 0
      for row in reader:
        text_with_val = dict(zip(headers, row))
        if not 'AppName' in text_with_val:
          continue
        
        if print_apps:
          print("Vector for app: ", text_with_val['AppName'])
        transformed: dict[str, torch.Tensor] = {}
        for name in embedded_vals:
          if name in text_with_val:
            transformed[name] = self._embed_value(name, text_with_val[name])
        text_with_val['Embedding'] = self._merge_vector_dict(transformed)
        key = f"{text_with_val.get('AppName', 'connection')}_{index}"
        connections[key] = text_with_val
        index += 1
    return connections

  def _embed_from_text_file(self, source: str, print_apps: bool) -> dict[str, dict[str, Any]]:
    """
    @param source: path to text file with connections, expects lines in format 'ParameterName: ParameterValue'
    and empty line between connections
    @param print_apps: if True, print app name for each connection
    
    Returns dictionary with connection names as keys and dictionaries with connection parameters and embedding
    vector under 'Embedding' key as values
    
    Function embedding all connections from a text file into dictionary.
    """
    connections: dict[str, dict[str, Any]] = {}
    with open(source) as file:
      # Temporary dict storing connection parameters
      connection: dict[str, str] = {}
      index = 0
      for line in file:
        line = line.strip()
        # Expecting end of line end of connection, if empty line, embed connection and add to connections dictionary
        if not line:
          if connection and 'AppName' in connection:
            transformed: dict[str, torch.Tensor] = {}
            # Embed each defined parameter for connection
            for name in embedded_vals:
              if name in connection:
                transformed[name] = self._embed_value(name, connection[name])
            connection['Embedding'] = self._merge_vector_dict(transformed)
            key = f"{connection.get('AppName', 'connection')}_{index}"
            connections[key] = connection.copy()
            index += 1
          connection = {}
          continue
        # Parse line in format 'ParameterName: ParameterValue', if not in this format, skip line
        key_value = line.split(':', 1)
        if len(key_value) < 2:
          continue
        name = key_value[0]
        value = key_value[1].strip()
        connection[name] = value
        if name == 'AppName' and print_apps:
          print("Vector for app: ", value)
          
      # For the last connection in file if not followed by empty line
      if connection and 'AppName' in connection:
        transformed: dict[str, torch.Tensor] = {}
        for name in embedded_vals:
          if name in connection:
            transformed[name] = self._embed_value(name, connection[name])
        connection['Embedding'] = self._merge_vector_dict(transformed)
        key = f"{connection.get('AppName', 'connection')}_{index}"
        connections[key] = connection.copy()
    return connections

  def get_embedding_for_file_connections(self, source: str, print_apps: bool = False) -> dict[str, dict[str, Any]]:
    """
    @param source: path to file with connections, can be csv or txt in format from dataset_show script
    @param print_apps: if True, print app name for each connection
    
    Returns dictionary with connection names as keys and dictionaries with connection parameters and embedding vector under 'Embedding' key as values
    
    Function embedding all connections from given file. Determines file type and calls appropriate embedding function.
    For details on expected file formats, see _embed_from_csv and _embed_from_text_file functions.
    """
    if source.endswith('.csv'):
      return self._embed_from_csv(source, print_apps=print_apps)
    else:
      return self._embed_from_text_file(source, print_apps=print_apps)

  def get_embedding_for_connection(self, source: dict[str, Any]) -> torch.Tensor:
    """
    @param source: dictionary with connection parameters, keys are parameter names, values are parameter values
    
    Returns embedding vector for given connection parameters
    
    Function getting embedding for connection from dictionary.
    """
    transformed: dict[str, torch.Tensor] = {}
    for name, value in source.items():
      if name in embedded_vals:
        transformed[name] = self._embed_value(name, str(value))
    return self._merge_vector_dict(transformed)


if __name__ == "__main__":
  if len(sys.argv) < 2 or '-h' in sys.argv or '--help' in sys.argv:
    print("Usage: python3 embedConnection.py <filename> [--one-line]")
    print("Embed connections from a file into tensor vectors.")
    print("\nArguments:")
    print("  filename           Path to connections file (CSV or text format)")
    print("\nOptions:")
    print("  --one-line         Print entire result on one line")
    print("  -h, --help         Show this help message")
    exit(1)

  embedder = ConnectionEmbedder()
  result = embedder.get_embedding_for_file_connections(sys.argv[1])

  config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config.json")
  with open(config_path, "r") as f:
    config = json.load(f)
  
  if '--one-line' in sys.argv:
    print(result)
    exit(0)
  sni_len = config['vector_sizes']['SNI']
  port_len = config['vector_sizes']['SrcPort']
  cipher_len = config['vector_sizes']['Client CipherSuite']
  ext_len = config['vector_sizes']['Client Extensions']
  group_len = config['vector_sizes']['Client Supported Groups']

  for name, connection in result.items():
    embedding = connection['Embedding']
    print(f"Connection: {name}")
    print("Final embedding size: ", embedding.shape)
    print("SNI: ", embedding[0:sni_len])
    print(f"SrcPort: {embedding[sni_len]:.10f}")
    print("Client CipherSuite: ", embedding[sni_len+1:sni_len+1+cipher_len])
    print("Client Extensions: ", embedding[sni_len+1+cipher_len:sni_len+1+cipher_len+ext_len])
    print("Client Supported Groups: ", embedding[sni_len+1+cipher_len+ext_len:sni_len+1+cipher_len+ext_len+group_len])