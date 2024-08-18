# train.py
import torch
from torch.utils.data import DataLoader
from model import GenreClassifier
from dataset import prepare_datasets
from torchaudio.transforms import MFCC

def train(model, train_loader, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs.unsqueeze(1))  # Add channel dimension
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}')

def evaluate(model, test_loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs.unsqueeze(1))
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy: {100 * correct / total}%')

if __name__ == "__main__":
    data_dir = "./data/gtzan/genres"
    transform = MFCC(sample_rate=22050, n_mfcc=40)

    train_dataset, test_dataset = prepare_datasets(data_dir, transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = GenreClassifier()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    train(model, train_loader, criterion, optimizer)
    evaluate(model, test_loader)
