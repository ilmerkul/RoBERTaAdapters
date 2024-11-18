from torch.utils.data import Dataset


class RoBERTaDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        return self.dataset.num_rows
