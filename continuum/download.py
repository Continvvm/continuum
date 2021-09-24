import os
import zipfile
import tarfile
from six.moves import urllib

import requests


# Fix error when downloading dataset from torchvision
# https://github.com/pytorch/vision/issues/1938#issuecomment-790730080
opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)


def download(url, path):
    if not os.path.exists(path):
        os.makedirs(path)

    file_name = os.path.join(path, url.split("/")[-1])

    if os.path.exists(file_name):
        print(f"Dataset already downloaded at {file_name}.")
    else:
        urllib.request.urlretrieve(url, file_name, ProgressBar().update)

    return file_name


def unzip(path):
    directory_path = os.path.dirname(path)
    with zipfile.ZipFile(path, 'r') as zip_file:
        zip_file.extractall(directory_path)


def untar(path):
    directory_path = os.path.dirname(path)
    with tarfile.open(path) as tar_file:
        tar_file.extractall(directory_path)


def download_file_from_google_drive(id, destination):
    """Taken from https://stackoverflow.com/a/39225272"""
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = None
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            token = value
            break

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)


class ProgressBar:
    """Basic Progress Bar.

    Inspired from: https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
    """

    def __init__(self):
        self.count = 0

    def update(self, tmp, block_size, total_size):
        self.count += block_size

        percent = f"{int(100 * self.count / total_size)}"
        filled_length = int(100 * self.count // total_size)
        pbar = "#" * filled_length + '-' * (100 - filled_length)

        print("\r|%s| %s%%" % (pbar, percent), end="\r")
        if self.count == total_size:
            print()

    def end(self, total_size):
        self.count = total_size
        self.update(None, 0, total_size)
