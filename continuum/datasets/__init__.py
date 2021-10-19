# pylint: disable=C0401
# flake8: noqa
from continuum.datasets.base import (
    ImageFolderDataset, InMemoryDataset, PyTorchDataset, _ContinuumDataset, H5Dataset
)
from continuum.datasets.cifar100 import CIFAR100
from continuum.datasets.core50 import (Core50, Core50v2_79, Core50v2_196, Core50v2_391)
from continuum.datasets.fellowship import (CIFARFellowship, Fellowship, MNISTFellowship)
from continuum.datasets.imagenet import ImageNet100, ImageNet1000, TinyImageNet200
from continuum.datasets.synbols import Synbols
from continuum.datasets.nlp import MultiNLI
from continuum.datasets.pytorch import (
    CIFAR10, EMNIST, KMNIST, MNIST, QMNIST, FashionMNIST
)
from continuum.datasets.svhn import SVHN
from continuum.datasets.colored_mnist import ColoredMNIST
from continuum.datasets.rainbow_mnist import RainbowMNIST
from continuum.datasets.cub200 import CUB200
from continuum.datasets.awa2 import AwA2
from continuum.datasets.pascalvoc import PascalVOC2012
from continuum.datasets.stream51 import Stream51
from continuum.datasets.dtd import DTD
from continuum.datasets.vlcs import VLCS
from continuum.datasets.pacs import PACS
from continuum.datasets.domain_net import DomainNet
from continuum.datasets.office_home import OfficeHome
from continuum.datasets.terra_incognita import TerraIncognita
from continuum.datasets.domain_net import DomainNet
from continuum.datasets.rainbow_mnist import RainbowMNIST
from continuum.datasets.car196 import Car196
from continuum.datasets.caltech import Caltech101, Caltech256
from continuum.datasets.fgvc_aircraft import FGVCAircraft
from continuum.datasets.stl10 import STL10
from continuum.datasets.food101 import Food101
from continuum.datasets.omniglot import Omniglot
from continuum.datasets.birdsnap import Birdsnap
from continuum.datasets.ctrl import CTRL, CTRLplus, CTRLminus, CTRLin, CTRLout, CTRLplastic
from continuum.datasets.flowers102 import OxfordFlower102
from continuum.datasets.oxford_pet import OxfordPet
from continuum.datasets.gtsrb import GTSRB
from continuum.datasets.sun397 import SUN397
from continuum.datasets.fer2013 import FER2013
from continuum.datasets.eurosat import EuroSAT
