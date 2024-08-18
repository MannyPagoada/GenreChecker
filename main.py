import torch
from torch import nn
from torch.utils.data import DataLoader 

from downloader import download_gtzan_dataset








if __name__ == "__main__":
    url = "http://opihi.cs.uvic.ca/sound/genres.tar.gz"
    download_path = "./data"  # This will create a "data" folder in the same directory as the script
    extract_path = "./data/gtzan"  # The dataset will be extracted inside the "data/gtzan" folder




    download_gtzan_dataset(url, download_path, extract_path)
    print("GTZAN dataset download process completed.")