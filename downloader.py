import os
import requests
from zipfile import ZipFile

def download_gtzan_dataset(url, download_path, extract_path):
    if not os.path.exists(download_path):
        os.makedirs(download_path)

    dataset_zip_path = os.path.join(download_path, "gtzan-dataset.zip")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(dataset_zip_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

    with ZipFile(dataset_zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

    print("GTZAN dataset downloaded and extracted.")
