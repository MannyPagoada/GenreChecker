# dataset.py
import os
import torch
import torchaudio
from torch.utils.data import Dataset

class GTZANDataset(Dataset):
    def __init__(self, file_list, labels, transform=None):
        self.file_list = file_list
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        audio_path = self.file_list[idx]
        label = self.labels[idx]

        waveform, sample_rate = torchaudio.load(audio_path)

        if self.transform:
            waveform = self.transform(waveform)

        return waveform, label

def load_gtzan_data(data_dir):
    genres = os.listdir(data_dir)
    file_list = []
    labels = []
    for genre in genres:
        genre_dir = os.path.join(data_dir, genre)
        for file in os.listdir(genre_dir):
            if file.endswith(".wav"):
                file_list.append(os.path.join(genre_dir, file))
                labels.append(genre)
    return file_list, labels

def prepare_datasets(data_dir, transform, test_size=0.2):
    from sklearn.model_selection import train_test_split

    file_list, labels = load_gtzan_data(data_dir)

    label_to_index = {label: idx for idx, label in enumerate(sorted(set(labels)))}
    labels = [label_to_index[label] for label in labels]

    train_files, test_files, train_labels, test_labels = train_test_split(file_list, labels, test_size=test_size, random_state=42)

    train_dataset = GTZANDataset(train_files, train_labels, transform=transform)
    test_dataset = GTZANDataset(test_files, test_labels, transform=transform)

    return train_dataset, test_dataset
