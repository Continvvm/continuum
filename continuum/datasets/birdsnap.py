import os
from typing import Tuple

import numpy as np
import pandas as pd
from continuum.datasets import _ContinuumDataset
from continuum.download import download, untar
from continuum.tasks import TaskType
from sklearn.model_selection import train_test_split
from torchvision import datasets as torchdata


class Birdsnap(_ContinuumDataset):
    """Birdsnap dataset.

    * Birdsnap: Large-scale Fine-grained Visual Categorization of Birds
      T. Berg, J. Liu, S. W. Lee, M. L. Alexander, D. W. Jacobs, and P. N. Belhumeur
      CVPR 2014
    """
    meta_url = "http://thomasberg.org/datasets/birdsnap/1.1/birdsnap.tgz"

    def __init__(self, data_path, train: bool = True, download: bool = True, test_split: float = 0.2,
                 random_seed=1):
        self.test_split = test_split
        self.random_seed = random_seed
        super().__init__(data_path, train, download)

    @property
    def data_type(self) -> TaskType:
        return TaskType.IMAGE_PATH

    def _download(self):
        if not os.path.exists(os.path.join(self.data_path, "birdsnap")):
            archive_path = os.path.join(self.data_path, "birdsnap.tgz")

            if not os.path.exists(archive_path):
                print("Downloading archive of metadata...", end=' ')
                download(self.meta_url, self.data_path)
                print('Done!')

            print('Extracting archive...', end=' ')
            untar(archive_path)
            print('Done!')

        with open(os.path.join(self.data_path, "birdsnap", "images.txt")) as f:
            next(f)  # skip header
            for line in f:

                line = list(filter(lambda x: len(x) > 0, line.split(" ")))
                url = line[0]
                md5 = line[1]
                path = line[2]
                class_id = line[3]
                x1, y1, x2, y2 = line[4:9]






    def get_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        dataset = torchdata.ImageFolder(os.path.join(self.data_path, "Animals_with_Attributes2", "JPEGImages"))
        x, y, _ = self._format(dataset.imgs)

        x_train, x_test, y_train, y_test = train_test_split(
            x, y,
            test_size=self.test_split,
            random_state=self.random_seed
        )

        if self.train:
            return x_train, y_train, None
        return x_test, y_test, None






import argparse
import datetime
import hashlib
import os
import shutil

import requests


class DownloadResult(object):
    NEW_OK = 0
    ALREADY_OK = 1
    DOWNLOAD_FAILED = 2
    SAVE_FAILED = 3
    MD5_FAILED = 4
    MYSTERY_FAILED = 5
    values = (NEW_OK, ALREADY_OK, DOWNLOAD_FAILED, SAVE_FAILED, MD5_FAILED, MYSTERY_FAILED)
    names = ('NEW_OK', 'ALREADY_OK', 'DOWNLOAD_FAILED', 'SAVE_FAILED', 'MD5_FAILED',
             'MYSTERY_FAILED')

def ensure_dir(dirpath):
    if not os.path.exists(dirpath): os.makedirs(dirpath)

def ensure_parent_dir(childpath):
    ensure_dir(os.path.dirname(childpath))

def logmsg(msg, flog=None):
    tmsg = '[%s] %s' % (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), msg)
    print '%s' % tmsg
    if flog is not None:
        flog.write('%s\n' % tmsg)
        flog.flush()

def logexc(msg, exc, flog=None):
    logmsg('{} [{}: {}]'.format(msg, exc.__class__.__name__, exc), flog=flog)

def get_birdsnap(imlistpath, outroot):
    ensure_dir(outroot)
    logpath = os.path.join(outroot, 'log.txt')
    with open(logpath, 'at') as flog:
        logmsg('Starting.', flog=flog)
        temproot, imageroot = (os.path.join(outroot, dirname) for dirname in ('temp', 'images'))
        if os.path.exists(temproot):
            logmsg('Removing existing temp directory.', flog=flog)
            shutil.rmtree(temproot)
        for d in (temproot, imageroot): ensure_dir(d)
        logmsg('Reading images list from file {}...'.format(imlistpath), flog=flog)
        images = read_list_of_dicts(imlistpath)
        n_images = len(images)
        logmsg('{} images in list.'.format(n_images), flog=flog)
        results = dict((result, 0) for result in DownloadResult.values)
        successpath, failpath = (os.path.join(outroot, fname)
                                 for fname in ('success.txt', 'fail.txt'))
        with open(successpath, 'wt') as fsuccess, open(failpath, 'wt') as ffail:
            for i_image, image in enumerate(images):
                logmsg('Start image {} of {}: {}'.format(i_image + 1, n_images, image['path']),
                       flog=flog)
                result = download_image(image, temproot, imageroot, flog=flog)
                (fsuccess if result in (DownloadResult.NEW_OK, DownloadResult.ALREADY_OK)
                 else ffail).write('{}\t{}\n'.format(image['url'], image['path']))
                results[result] += 1
                logmsg('Finished image {} of {} with result {}.  Progress is {}.'.format(
                        i_image + 1, n_images, DownloadResult.names[result],
                        ', '.join('{}:{}'.format(DownloadResult.names[k], v)
                                  for k, v in results.items())),
                       flog=flog)
        logmsg('Finished.', flog=flog)

def check_image(image, imagepath):
    if not os.path.exists(imagepath): return False
    with open(imagepath, 'rb') as fin:
        return (hashlib.md5(fin.read()).hexdigest() == image['md5'])

def download_image(image, temproot, imageroot, flog=None):
    # Check existing file.
    try:
        temppath, imagepath = (os.path.join(root, image['path']) for root in (temproot, imageroot))
        if check_image(image, imagepath):
            logmsg('Already have the image and contents are correct.', flog=flog)
            return DownloadResult.ALREADY_OK
        else:
            logmsg('Need to get the image.', flog=flog)
            if os.path.exists(imagepath):
                logmsg('Deleting existing bad file {}.'.format(imagepath), flog=flog)
                os.remove(imagepath)
    except Exception as e:
        logexc('Unexpected exception before attempting download of image {!r}.'.format(image), e,
               flog=flog)
        return DownloadResult.MYSTERY_FAILED
    # GET and save to temp location.
    try:
        r = requests.get(image['url'])
        if r.status_code == 200:
            ensure_parent_dir(temppath)
            with open(temppath, 'wb') as fout:
                for chunk in r.iter_content(1024): fout.write(chunk)
            logmsg('Saved  {}.'.format(temppath), flog=flog)
        else:
            logmsg('Status code {} when requesting {}.'.format(r.status_code, image['url']))
            return DownloadResult.DOWNLOAD_FAILED
    except Exception as e:
        logexc('Unexpected exception when downloading image {!r}.'.format(image), e, flog=flog)
        return DownloadResult.DOWNLOAD_FAILED
    # Check contents.
    try:
        if check_image(image, temppath):
            logmsg('Image contents look good.', flog=flog)
        else:
            logmsg('Image contents are wrong.', flog=flog)
            return DownloadResult.MD5_FAILED
    except Exception as e:
        logexc('Unexpected exception when checking file contents for image {!r}.'.format(image), e,
               flog=flog)
        return DownloadResult.MYSTERY_FAILED
    # Move image to final location.
    try:
        ensure_parent_dir(imagepath)
        os.rename(temppath, imagepath)
    except Exception as e:
        logexc('Unexpected exception when moving file from {} to {} for image {!r}.'.format(
                temppath, imagepath, image), e, flog=flog)
        return DownloadResult.MYSTERY_FAILED
    return DownloadResult.NEW_OK

def read_list_of_dicts(path):
    rows = []
    with open(path, 'r') as fin:
        fieldnames = fin.readline().strip().split('\t')
        for line in fin:
            vals = line.strip().split('\t')
            assert len(vals) == len(fieldnames)
            rows.append(dict(zip(fieldnames, vals)))
        return rows

def testargs():
    imlistpath = '/media/lacie_alpha/thomas/cubirds/data/dist/1.0-rec-debug-20150125-143026/images.txt'
    outroot = '/media/lacie_alpha/thomas/cubirds/data/dist/testout'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Download the images of the Birdsnap dataset.')
    parser.add_argument(
        '--output_dir',
        default='download',
        help='directory in which to save the images and output files')
    parser.add_argument(
        '--images-file',
        default='images.txt',
        help='image list file')
    args = parser.parse_args()
    get_birdsnap(args.images_file, args.output_dir)
