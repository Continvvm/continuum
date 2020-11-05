import os
import urllib.request
import zipfile


def download(url, path):
    """
    Download a file from a url.

    Args:
        url: (str): write your description
        path: (str): write your description
    """
    if not os.path.exists(path):
        os.makedirs(path)

    file_name = os.path.join(path, url.split("/")[-1])

    if os.path.exists(file_name):
        print(f"Dataset already downloaded at {file_name}.")
    else:
        opener = urllib.request.build_opener()
        opener.addheaders = [('User-agent', 'Wget/1.20.3 (linux-gnu)')]
        urllib.request.install_opener(opener)
        urllib.request.urlretrieve(url, file_name, ProgressBar().update)

    return file_name


def unzip(path):
    """
    Unzip a directory.

    Args:
        path: (str): write your description
    """
    directory_path = os.path.dirname(path)

    with zipfile.ZipFile(path, 'r') as zip_ref:
        zip_ref.extractall(directory_path)


class ProgressBar:
    """Basic Progress Bar.

    Inspired from: https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
    """

    def __init__(self):
        """
        Initialize the state

        Args:
            self: (todo): write your description
        """
        self.count = 0

    def update(self, tmp, block_size, total_size):
        """
        Prints a progress bar

        Args:
            self: (todo): write your description
            tmp: (array): write your description
            block_size: (int): write your description
            total_size: (int): write your description
        """
        self.count += block_size

        percent = f"{int(100 * self.count / total_size)}"
        filled_length = int(100 * self.count // total_size)
        pbar = "#" * filled_length + '-' * (100 - filled_length)

        print("\r|%s| %s%%" % (pbar, percent), end="\r")
        if self.count == total_size:
            print()
