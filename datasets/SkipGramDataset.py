import torch
from torch.utils.data import Dataset, DataLoader


class SkipGramDataset(Dataset):
    def __init__(self, data, word_to_ix):
        self.data = data
        self.word_to_ix = word_to_ix

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        center_word, context_word = self.data[idx]
        return (
            torch.tensor(self.word_to_ix[center_word], dtype=torch.long),
            torch.tensor(self.word_to_ix[context_word], dtype=torch.long)
        )

def get_data(data, word_to_ix, batch_size):
    dataset = SkipGramDataset(data, word_to_ix)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader