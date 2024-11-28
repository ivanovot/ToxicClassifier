from torch.utils.data import DataLoader, Dataset
import torch
from scr.sbert import vectorize as vec
import pandas as pd
from tqdm import tqdm

tqdm.pandas()

df = pd.read_csv('data/data.csv')

class TextDataset(Dataset):
    def __init__(self, df):
        self.vectors = torch.stack(list(df['comment'].progress_apply(lambda x: vec(x).squeeze(0))))
        self.labels = torch.tensor(df['toxic'].values)

    def __getitem__(self, index):
        return self.vectors[index], self.labels[index]

    def __len__(self):
        return len(self.labels)

if __name__ == '__main__':
    # Сохраняем векторизованный датасет
    dataset = TextDataset(df)
    torch.save(dataset, 'data/dataset.pt')
    
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, pin_memory=True)

    # Загружаем данные с DataLoader
    for texts, labels in dataloader:
        print(texts, labels)
        break