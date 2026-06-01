import os
from torch.utils.data import Dataset

class Dataset(Dataset):

    def __init__(self, dataset_path, namespace, split='train'):

        if split=='valid': split='evaluate'
        dataset_path = os.path.join(dataset_path, namespace, f"{split}_gene_label")
        self.data = self._process_data(dataset_path)

    def __getitem__(self, index):
        return self.data[index]
            
    def __len__(self):
        return len(self.data)
    
    def _process_data(self, path):
        data = []
        with open(path, 'r', encoding='utf-8') as f1:
            for line in f1:
                parts = line.strip().split()
                protein_id = parts[0]
                goa = set(parts[1].split(','))

                data.append((protein_id, goa))
        
        return data
        