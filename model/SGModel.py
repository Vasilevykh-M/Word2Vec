import torch.nn as nn

class SGModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SGModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs)
        output = self.linear(embeds)
        return output