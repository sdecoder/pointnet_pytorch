from collections import defaultdict
from glob import glob

import torch
from pathlib import Path
import random
from pathlib import Path
import importlib
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import Dataset, DataLoader

from IPython.display import display
import pandas as pd
import numpy as np

from PIL import Image
from matplotlib import pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm

from scipy.spatial.distance import euclidean
from imageio import imread
from skimage.transform import resize

# some of blocks below are not used.

# Data manipulation
import numpy as np
import pandas as pd

# Data visualisation
import matplotlib.pyplot as plt

# Fastai
# from fastai.vision import *
# from fastai.vision.models import *

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.utils
import torchvision.datasets as dset

from torch import optim
from torch.utils.data import DataLoader, Dataset
from torchvision.models import *
from torchvision.datasets import ImageFolder
from torch.autograd import Variable
# import pretrainedmodels

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
import os
import matplotlib.pyplot as plt
import torch.nn.functional as F
from alive_progress import alive_bar
from torch import nn, optim
from torchvision import transforms as T, datasets, models
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from collections import OrderedDict
from tqdm import tqdm

import argparse

import onnx
import tensorrt as trt
import os
import torch
import torchvision
import pycuda.driver as cuda
import pycuda.autoinit
from PIL import Image
import numpy as np
import onnx
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from enum import Enum

pd.options.plotting.backend = "plotly"
from torch import nn, optim
from torch.autograd import Variable

TEST = 'test'
TRAIN = 'train'
VAL = 'val'


def build_engine_common_routine(network, builder, config, runtime, engine_file_path):
  plan = builder.build_serialized_network(network, config)
  print(f'[trace] done with builder.build_serialized_network')
  if plan == None:
    print("[trace] builder.build_serialized_network failed, exit -1")
    exit(-1)
  engine = runtime.deserialize_cuda_engine(plan)
  print("[trace] Completed creating Engine")
  with open(engine_file_path, "wb") as f:
    f.write(plan)
  return engine


class CalibratorMode(Enum):
  INT8 = 0
  FP16 = 1
  TF32 = 2
  FP32 = 3


class Calibrator(trt.IInt8EntropyCalibrator2):

  def __init__(self, training_loader, cache_file, element_bytes, batch_size=16, ):
    # Whenever you specify a custom constructor for a TensorRT class,
    # you MUST call the constructor of the parent explicitly.
    trt.IInt8EntropyCalibrator2.__init__(self)
    self.cache_file = cache_file
    self.data_provider = training_loader
    self.batch_size = batch_size
    self.current_index = 0

    # we assume single element is 4 byte
    mem_size = element_bytes * batch_size
    print(f'[trace] allocated mem_size: {mem_size}')
    self.device_input0 = cuda.mem_alloc(mem_size)

  def get_batch_size(self):

    return self.batch_size

  # TensorRT passes along the names of the engine bindings to the get_batch function.
  # You don't necessarily have to use them, but they can be useful to understand the order of
  # the inputs. The bindings list is expected to have the same ordering as 'names'.
  def get_batch(self, names):

    max_data_item = len(self.data_provider.dataset)
    if self.current_index + self.batch_size > max_data_item:
      return None

    _imgs0, labels = next(iter(self.data_provider))
    _elements0 = _imgs0.ravel().numpy()
    cuda.memcpy_htod(self.device_input0, _elements0)
    self.current_index += self.batch_size
    return [self.device_input0]

  def read_calibration_cache(self):
    # If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
    print(f'[trace] Calibrator: read_calibration_cache: {self.cache_file}')
    if os.path.exists(self.cache_file):
      with open(self.cache_file, "rb") as f:
        return f.read()

  def write_calibration_cache(self, cache):
    print(f'[trace] Calibrator: write_calibration_cache: {cache}')
    with open(self.cache_file, "wb") as f:
      f.write(cache)


def GiB(val):
  return val * 1 << 30
