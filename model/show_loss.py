"""
file: model/show_loss.py
author: Samuel Kudla <xkudlas00@stud.fit.vutbr.cz>
date: 12.05.2026
"""
import torch 
import torch.nn as nn 
from pathlib import Path
import sys
import numpy as np

root_path = (Path(__file__).resolve().parent.parent)
undercomplete_path = root_path / "model"
sys.path.append(str(undercomplete_path))

from Undercomplete import Undercomplete
from Undercomplete import load_config
from Undercomplete import get_model_file
from Classifier import Classifier

from Undercomplete import embeddings_path
sys.path.append(str(embeddings_path))
from embeddings.ConnectionEmbedder import ConnectionEmbedder


def _load_classifier_and_model():
    config = load_config()
    model_file = get_model_file(config)
    model = torch.load(model_file, weights_only=False)
    classifier = Classifier()
    return classifier, model

def show_loss_one_file(file: str) -> None:
    """
    @param file: path to file with connections, can be csv or txt in format from dataset_show script
    
    Function iterating through connections in file and showing reconstruction loss for each connection. Uses Classifier to get losses.
    """
    classifier, model = _load_classifier_and_model()
    # one connection
    if ('--one' in sys.argv):
            connections = classifier.embedder.get_embedding_for_file_connections(file)
            if connections:
                first_conn = next(iter(connections.values()))
                encoded_vec = first_conn['Embedding']
                loss = classifier.get_reconstruction_loss(model, encoded_vec)
                print('Reconstruction error:', f'{loss:.16f}')
    # Multiple connections - detailed result
    elif ('--stats' in sys.argv):
            connections = classifier.get_file_losses(model, file)
            losses = np.array([connection['Loss'] for connection in connections])
            print("Connections:", len(losses))
            print("Median:",np.median(losses))
            print("Mean:", np.mean(losses))
            print("Std_dev:", np.std(losses))
            min = np.min(losses)
            max = np.max(losses)
            print("Min loss:",min)
            print('Max loss:',max)
            loss_range = max - min
            # Margin that defines closness to extremes - can be modified
            # Can be useful for finding ideal threshold value
            offset = 0.4
            margin = loss_range * offset
            near_min = np.sum(losses <= (min + margin))
            near_max = np.sum(losses >= (max - margin))
            print("Margin set to:",margin,'offset:',offset*100,'%')
            print('Values count near min:',near_min,'which is:',near_min / len(losses) * 100,'%')
            print('Values count near max:',near_max,'which is:',near_max / len(losses) * 100,'%')
            print("90% Quantile:", np.quantile(losses, 0.90))
            print("95% Quantile:", np.quantile(losses, 0.95))
    # Multiple connections
    else:
        classifier.get_file_losses(model, file, print_loss=True)

def show_loss_multiple_files(directory: str) -> None:
    """
    @param directory: path to directory with files containing connections
    
    Function iterating through files in directory and showing reconstruction loss for each file. Uses Classifier to get losses for each file.
    """
    classifier, model = _load_classifier_and_model()
    from pathlib import Path
    p = Path(directory)
    for file_data in p.iterdir():
        connections = classifier.embedder.get_embedding_for_file_connections(str(file_data))
        if not connections:
            continue
        # pick only first connection embedding in file (one file represents one connection)
        first = next(iter(connections.values()))
        encoded_vec = first['Embedding']
        loss = classifier.get_reconstruction_loss(model, encoded_vec)
        print(file_data.name, ' '* (25 - len(file_data.name)), f': {loss:.16f}')
        
        
if __name__ == "__main__":
    config = load_config()
    model_file = get_model_file(config)
    print('Model loaded from file:', model_file)
    
    if ('-h' in sys.argv or '--help' in sys.argv):
        print("Usage: python show_loss.py [connections_file] [--one/--stats]")
        exit(0)
    
    # one file
    if len(sys.argv) >= 2:
        show_loss_one_file(sys.argv[1])
                        
    # iterate through test_data directory
    else:
        show_loss_multiple_files('test_data')