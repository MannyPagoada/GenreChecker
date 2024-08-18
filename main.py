from train import train, evaluate
from dataset import prepare_datasets
from model import GenreClassifier
from torchaudio.transforms import MFCC
import torch
from torch.utils.data import DataLoader
import os

if __name__ == "__main__":
    try:
        # Set the data directory to where your downloaded data is located
        data_dir = r"C:\Users\MPago\Documents\genre checker\data\gtzan\archive\data\genres_original"
        parent_dir = os.path.dirname(data_dir)
        
        # Debugging: Print the data directory and its parent directory
        print(f"Data directory: {data_dir}")
        print(f"Parent directory: {parent_dir}")
        
        if not os.path.exists(data_dir):
            # Debugging: Print the contents of the parent directory
            print(f"Contents of parent directory ({parent_dir}): {os.listdir(parent_dir)}")
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
        print(f"Data directory exists: {data_dir}")

        # Define the transformation to apply to the audio data
        transform = MFCC(sample_rate=22050, n_mfcc=40)
        print("Transformation defined")

        # Prepare the datasets
        train_dataset, test_dataset = prepare_datasets(data_dir, transform)
        print("Datasets prepared")

        # Create DataLoader objects for training and testing
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        print("DataLoaders created")

        # Initialize the model
        model = GenreClassifier()
        print("Model initialized")

        # Train the model
        train(model, train_loader)
        print("Model trained")

        # Evaluate the model
        evaluate(model, test_loader)
        print("Model evaluated")

    except Exception as e:
        print(f"An error occurred: {e}")