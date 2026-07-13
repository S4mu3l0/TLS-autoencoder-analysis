"""
file: model/make_test_data.py
author: Samuel Kudla <xkudlas00@stud.fit.vutbr.cz>
date: 12.05.2026
"""
from pathlib import Path
import sys
import torch

root_path = (Path(__file__).resolve().parent.parent)
sys.path.append(str(root_path))
embeddings_path = root_path / "embeddings"
sys.path.append(str(embeddings_path))

from embeddings.ConnectionEmbedder import ConnectionEmbedder, embedded_vals

# Keep track of state of parsing in a dictionary.
state = {
    "count": 0,
    "skipped": 0,
    "embedded_inputs": [],
}

def append_connection(connection: dict[str, str]) -> None:
    """
    @param connection: dictionary with connection parameters, expects keys in embedded_vals list and 'AppName' key
    
    Function checking if connection has all required parameters for embedding, if not, it is skipped and a message is printed. If connection is complete, its embedding vector is computed and added to state under 'embedded_inputs' key.
    The function also keeps track of how many connections have been processed and prints a message every 100 connections.
    """
    missing = [name for name in embedded_vals if name not in connection]
    if missing:
        state["skipped"] += 1
        print(f"Skipping incomplete connection missing: {', '.join(missing)}")
        return

    state["embedded_inputs"].append(connection_embedder.get_embedding_for_connection(connection))
    state["count"] += 1
    if state["count"] % 100 == 0:
        print(f"Processed {state['count']} connections.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 make_test_data.py raw_data_file [output_file_name] (expected data format from dataset_show script).")
        exit(1)

    output = 'embedded_vectors.pt'
    if len(sys.argv) > 2:
        output = sys.argv[2]

    connection_embedder = ConnectionEmbedder()

    with open(sys.argv[1], "r") as f:
        data = f.read().splitlines()
        connection = {}
        print("Processing connections...")

        for line in data:
            if line:
                parts = line.split(':', 1)
                if len(parts) == 2:
                    key, value = parts[0], parts[1].strip()
                    connection[key] = value
            else:
                if connection:
                    append_connection(connection)
                    connection = {}

        if connection:
            append_connection(connection)

    if state["skipped"]:
        print(f"Skipped {state['skipped']} incomplete connections.")

    embedded_inputs = torch.stack(state["embedded_inputs"]) if state["embedded_inputs"] else torch.empty((0,))
    torch.save(embedded_inputs, output if output.endswith('.pt') else output + '.pt')