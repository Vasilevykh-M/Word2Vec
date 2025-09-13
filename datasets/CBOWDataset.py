import torch
from torch.utils.data import Dataset, DataLoader


class CBOWDataset(Dataset):
    def __init__(self, data, word_to_ix):
        self.data = data
        self.word_to_ix = word_to_ix

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        context, target = self.data[idx]
        context_ids = [self.word_to_ix[word] for word in context]
        target_id = self.word_to_ix[target]
        return torch.tensor(context_ids, dtype=torch.long), torch.tensor(target_id, dtype=torch.long)

def get_data(data, word_to_ix, batch_size):
    dataset = CBOWDataset(data, word_to_ix)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader