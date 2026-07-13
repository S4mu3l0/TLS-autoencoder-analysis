"""
file: model/Classifier.py
author: Samuel Kudla <xkudlas00@stud.fit.vutbr.cz>
date: 12.05.2026
"""
import json
import sys
from Undercomplete import Undercomplete
from Undercomplete import load_config
from embeddings.ConnectionEmbedder import ConnectionEmbedder
from Undercomplete import thresholds_path
import torch

class Classifier():
  """
  Class classifying connections to nearest model based on reconstruction loss. 
  It loads all models specified in config and their thresholds from thresholds file or model attribute. 
  """
  def __init__(self, print_results = False, embedder: ConnectionEmbedder | None = None):
    config = load_config()
    with open(thresholds_path, 'r') as thresholds_file:
      thresholds = json.load(thresholds_file)
      
    self.print_results = print_results
    # allow injection of ConnectionEmbedder for reuse/testing
    self.embedder = embedder or ConnectionEmbedder()
    
    # Class stores model objects in list  
    self.models = []
    # Class stores model thresholds separately in dictionary with model name as key and threshold as value, loaded from thresholds file or model attribute
    self.model_thresholds = {}
    models = config['test_models_name']
    for model_name, _ in models.items():
      model = torch.load(model_name, weights_only=False)
      model.name = model_name
      self.models.append(model)
      try:
        self.model_thresholds[model.name] = thresholds[model.name]
      except KeyError:
        try:
          self.model_thresholds[model.name] = model.threshold
        except AttributeError:
          print(f"Model {model.name} has no threshold attribute or value in thresholds file.")
          exit(1)
  
  def get_reconstruction_loss(self, model: Undercomplete, encoded_vec: torch.Tensor) -> float:
    """
    @param model: model to calculate loss for
    @param encoded_vec: embedding vector, expected in tensor format
    
    Returns reconstruction loss value
    
    Computes reconstruction loss for given model and embedding vector.
    """
    model.eval()
    with torch.no_grad():
      if not torch.is_tensor(encoded_vec):
        encoded_vec = torch.tensor(encoded_vec)
      input_tensor = encoded_vec.clone().detach().float().unsqueeze(0)
      reconstructed = model(input_tensor)
      loss = model.criterion(reconstructed, input_tensor)
    return loss.item()
  
  def get_file_losses(self, model: Undercomplete, file_path: str, print_loss=False, print_progress=False) -> list:
    """
    @param model: model to test
    @param file_path: path to file with connections
    @param embedder: ConnectionEmbedder object to get embeddings for connections in file
    @param print_loss: if True, prints connection name and loss
    @param print_progress: if True, prints progress of parsing
    
    Returns list of dictionaries with connection parameters and their losses
    
    Computes reconstruction loss for all connections in file and returns parsed connections with losses.
    """
    parsed_connections = list(self.embedder.get_embedding_for_file_connections(file_path).values())
    
    for i, connection_dict in enumerate(parsed_connections):
      loss = self.get_reconstruction_loss(model, connection_dict['Embedding'])
      connection_dict['Loss'] = loss
      
      if print_loss:
        app_name = connection_dict.get('AppName', 'Unknown')
        print(f"{app_name:<25} : {loss:.16f}")
        
      # Print progress every 100 connections
      if print_progress and (i + 1) % 100 == 0:
        print(f'Calculated loss for {i + 1} connections...')
        
    return parsed_connections
  
  def classify_multiple_connections(self, file: str) -> list:
    """
    @param file: path to file with connections, can be csv or txt in format from dataset_show script
    @param embedder: ConnectionEmbedder object to get embeddings for connections in file
    
    Returns list of dictionaries with connection parameters, nearest model name under 'NearestModel' key and reconstruction loss under 'Loss' key. 
    
    Function classifying multiple connections from given file to nearest model based on reconstruction loss. Function uses Embedder class to get 
    embeddings for connections and then calculates reconstruction loss for each model. If loss is below model threshold, it is considered as candidate model
    and the one with lowest loss is selected as nearest model. Results are printed if print_results attribute is True.
    Only connections with AppName parameter are parsed (needed for classification).
    """
    print(f"Classifying connections from file {file} to nearest model...")
    parsed_connections = list(self.embedder.get_embedding_for_file_connections(file).values())
    
    for connection in parsed_connections:
      candidate_model_losses = {}
      connection['NearestModel'] = None
      connection['Loss'] = None
      
      for m in self.models:
        loss = self.get_reconstruction_loss(m, connection['Embedding'])
        if loss < self.model_thresholds[m.name]:
          candidate_model_losses[m.name] = loss
      
      if candidate_model_losses:
        nearest_model = min(candidate_model_losses, key = candidate_model_losses.get)
        connection['NearestModel'] = nearest_model
        connection['Loss'] = candidate_model_losses[nearest_model]
      
      if self.print_results: print(f"App: {connection['AppName']}, Nearest Model: {connection['NearestModel']}")
    return parsed_connections
      
  def classify_one_connection(self, file: str) -> str:
    """ 
    @param file: path to file with one connection, can be csv or txt in format from dataset_show script,
    only one connection should be present in file.
    @param embedder: ConnectionEmbedder object to get embedding for connection in file
    
    Returns name of nearest model based on reconstruction loss
    
    Function classifying one connection from given file to nearest model based on reconstruction loss. Function uses get_model_loss_file.
    """
    losses = {}
    for m in self.models:
      loss = self._get_model_loss_file(m, file)
      losses[m.name] = loss
    return min(losses, key = losses.get)
        
  def _get_model_loss_file(self, model: Undercomplete, file: str) -> float:
    """
    @param model: model to calculate loss for 
    @param file: path to file with one connection, can be csv or txt in format.
    @param embedder: ConnectionEmbedder object to get embedding for connection in file
    
    Returns reconstruction loss value for connection in file and given model
    
    Function calculating reconstruction loss for given model and connection file. Uses ConnectionEmbedder to get embedding for connection in 
    file and compute_reconstruction_loss to calculate loss for model.
    """
    
    connections = self.embedder.get_embedding_for_file_connections(file)
    if not connections:
      raise ValueError(f"No connections parsed from file {file}")
    # pick only first connection embedding
    first_conn = next(iter(connections.values()))
    embedding = first_conn['Embedding']
    loss = self.get_reconstruction_loss(model, embedding)
    if self.print_results: print(f"Loss for model '{model.name}': {loss}")
    return loss
    
if __name__ == '__main__':
  if len(sys.argv) < 2 or '-h' in sys.argv:
    print("Usage: python3 MultipleClassifier <connection_file> [--multiple]")
    print("This scripts shows losses of model inputed for a connection file and classifies the app to the model based on lowest loss.")
    print("If --multiple is used, it classifies all connections in file to nearest model and prints results.")
    exit(1)
    
  file = sys.argv[1]
  
  embedder = ConnectionEmbedder()
  classifier = Classifier(print_results=True, embedder=embedder)
  if ('--multiple' in sys.argv):
    sys.argv.remove('--multiple')
    file = sys.argv[1]
    classifier.print_results = False
    result = classifier.classify_multiple_connections(file)
    for connection in result:
      print(f"App: {connection['AppName']}, Nearest Model: {connection['NearestModel']}, Loss: {connection['Loss']}")
  else:
    nearest_model = classifier.classify_one_connection(file)
    print(f'Nearest model to file {file}: {nearest_model}')