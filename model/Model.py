import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange
import hdbscan
import numpy as np

from datasets.Prepocessor import Prepocessor
from datasets.CBOWDataset import get_data as cbow_data
from datasets.SkipGramDataset import get_data as sg_data
from drawing.draw import visualize_clustering
from metrics.metrics import accuracy
from model.CBOWModel import CBOWModel
from model.SGModel import SGModel


class Model:
    def __init__(self, preproc_cfg, model_cfg, file_path=""):
        self.preprocessor = Prepocessor(**preproc_cfg)
        data, self.word_to_idx, self.idx_to_word, vocab_size = self.preprocessor(file_path, model_cfg["type_model"])
        self.cluster_representatives = {}
        if model_cfg["type_model"] == "CBOW":
            self.model = CBOWModel(vocab_size, model_cfg["embedding_dim"])
            dataloader = cbow_data(data, self.word_to_idx, model_cfg["batch_size"])

        if model_cfg["type_model"] == "SG":
            self.model = SGModel(vocab_size, model_cfg["embedding_dim"])
            dataloader = sg_data(data, self.word_to_idx, model_cfg["batch_size"])
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.train(dataloader, model_cfg["epochs"])

        self.clusterer = None

    def train(self, dataloader, epochs):
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-3)
        best_accuracy = 0
        best_epoch = 0
        best_loss = 0
        for epoch in trange(epochs, desc="Train model"):
            total_loss = 0
            acc = 0
            for context, target in dataloader:
                self.model.zero_grad()
                context = context.to(self.device)
                target = target.to(self.device)
                log_probs = self.model(context)
                acc += accuracy(log_probs, target)
                loss = self.loss_function(log_probs, target)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            acc = acc / len(dataloader)
            if acc > best_accuracy:
                best_accuracy = acc
                best_epoch = epoch
                best_loss = total_loss
                self.embeddings = self.model.embeddings.weight.data.clone()

        print(f"Accuracy by train: {best_accuracy} and loss: {best_loss}, in epoch: {best_epoch}")

    def clustering(self, min_cluster_size=2, min_samples=1):
        word_indices = list(self.word_to_idx.values())
        word_embeddings = self.embeddings[word_indices].cpu().numpy()

        if self.clusterer is None:
            self.clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                gen_min_span_tree=True,
                cluster_selection_epsilon=0.1
            )

        cluster_labels = self.clusterer.fit_predict(word_embeddings)

        unique_labels = np.unique(cluster_labels)

        for label in unique_labels:
            if label == -1:
                continue
            cluster_indices = np.where(cluster_labels == label)[0]

            if len(cluster_indices) == 0:
                continue

            cluster_embeddings = word_embeddings[cluster_indices]
            centroid = np.mean(cluster_embeddings, axis=0)

            distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
            closest_indices = np.argsort(distances)[0]
            representative_words = self.idx_to_word[cluster_indices[closest_indices]]

            self.cluster_representatives[label] = {
                'word': representative_words,
                'centroid': centroid,
            }

    def draw(self, output_file):
        visualize_clustering(self.cluster_representatives, output_file=output_file)


    def __call__(self, word):
        if not word in self.word_to_idx:
            return None
        idx = self.word_to_idx[word]
        emb = self.embeddings[idx].reshape(1, -1)
        return emb