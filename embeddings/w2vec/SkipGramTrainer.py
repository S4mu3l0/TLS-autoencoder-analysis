"""
file: embeddings/w2vec/SkipGramTrainer.py
author: Samuel Kudla <xkudlas00@stud.fit.vutbr.cz>
date: 12.05.2026
"""
# Training data as skip-gram
# Embedding words (extension numbers) that are often together to be similar

import argparse
import json
import os
import random 

import torch
import torch.nn as nn
import torch.optim as optim


class SkipGramTrainer:
    def __init__(
        self,
        path: str,
        delim: str,
        window: int = 2,
        epochs: int = 10,
        lr: float = 0.001,
        batch_size: int = 64,
        num_negatives: int = 10,
        margin: float = 10.0,
        embedding_dim: int = 16,
    ) -> None:
        self.path = path
        self.delim = delim
        self.window = window
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.num_negatives = num_negatives
        self.margin = margin
        self.embedding_dim = embedding_dim

        if args.path is None:
            print("Error: Please specify the path to training data using --path. Use --help for more information.")
            exit(1)
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.feature_dir = os.path.join(base_dir, "..", path)

        self.mapping: dict[str, int] = {}
        self.training_data: list[list[str]] = []
        self.pairs: list[tuple[str, str]] = []
        self.embedding: nn.Embedding

    def _load_mapping(self) -> None:
        with open(os.path.join(self.feature_dir, "mapping.json"), "r") as file:
            self.mapping = json.load(file)

    def _load_training_data(self) -> None:
        data: list[list[str]] = []
        with open(os.path.join(self.feature_dir, "train_vals"), "r") as file:
            for line in file:
                seq = line.strip().split(self.delim)
                data.append(seq)
        self.training_data = data

    def _load_embedding(self) -> None:
        self.embedding = nn.Embedding(len(self.mapping), self.embedding_dim)
        embedding_path = os.path.join(self.feature_dir, "embedding.pt")
        self.embedding.load_state_dict(torch.load(embedding_path))

    def _build_pairs(self) -> None:
        pairs: list[tuple[str, str]] = []
        print("Creating windows from training data...")
        for seq in self.training_data:
            for i, center in enumerate(seq):
                for j in range(max(0, i - self.window), min(len(seq), i + self.window + 1)):
                    if i != j:
                        pairs.append((center, seq[j]))
        self.pairs = pairs

    def _save_embedding(self) -> None:
        torch.save(self.embedding.state_dict(), os.path.join(self.feature_dir, "embedding.pt"))

    def train(self) -> None:
        self._load_mapping()
        self._load_training_data()

        self._load_embedding()
        self._build_pairs()

        if not self.pairs:
            print("ERROR: No pairs created! Check your training_data format or window size.")
            exit(1)

        optimizer = optim.Adam(self.embedding.parameters(), lr=self.lr)
        all_indices = list(self.mapping.values())

        print("Starting training...")
        for epoch in range(self.epochs):
            total_loss = 0.0
            # computing on batches of pairs, needed to shuffle pairs each epoch
            random.shuffle(self.pairs)

            for i in range(0, len(self.pairs), self.batch_size):
                batch = self.pairs[i : i + self.batch_size]
                if not batch:
                    continue

                center_idxs = torch.tensor([self.mapping[c] for c, _ in batch])
                positive_idxs = torch.tensor([self.mapping[p] for _, p in batch])

                optimizer.zero_grad()

                center_vecs = self.embedding(center_idxs)
                positive_vecs = self.embedding(positive_idxs)
                pos_dist = torch.norm(center_vecs - positive_vecs, p=2, dim=1)

                # for each positive pair include distance to num_negatives random negative samples
                neg_dist_total = 0.0
                for _ in range(self.num_negatives):
                    neg_idxs = torch.tensor([random.choice(all_indices) for _ in range(len(batch))])
                    negative_vecs = self.embedding(neg_idxs)
                    neg_dist = torch.norm(center_vecs - negative_vecs, p=2, dim=1)
                    # if distance to negative is not greater than positive + margin, add to loss
                    neg_dist_total += torch.relu(self.margin + pos_dist - neg_dist).mean()

                loss = (neg_dist_total / self.num_negatives)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            print(f"Epoch {epoch}, Loss: {total_loss / (len(self.pairs) / self.batch_size)}")
            self._save_embedding()

        print("Done!")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", type=int, default=2, help="Context window size")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--path", type=str, default=None, help="Directory of training data")
    parser.add_argument("--delim", type=str, default="-", help="Delimiter in training data")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate of training")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    trainer = SkipGramTrainer(
        path=args.path,
        delim=args.delim,
        window=args.window,
        epochs=args.epochs,
        lr=args.lr,
    )

    trainer.train()