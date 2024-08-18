import os
from model import GenreClassifier
from data_preprocessing import prepare_data
from train import train
from evaluate import evaluate
from downloader import download_gtzan_dataset

def main():
    dataset_url = "http://marsyas.info/downloads/datasets/gtzan-dataset.zip"  # Update with actual URL if needed
    download_path = "./data"
    extract_path = "./data/gtzan"

    if not os.path.exists(extract_path):
        download_gtzan_dataset(dataset_url, download_path, extract_path)

    train_loader, test_loader = prepare_data(extract_path)

    model = GenreClassifier()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    train(model, train_loader, criterion, optimizer, num_epochs=10)
    evaluate(model, test_loader)

if __name__ == "__main__":
    main()
