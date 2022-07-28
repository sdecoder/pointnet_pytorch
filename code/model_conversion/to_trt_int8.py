import os
import sys
import glob
import argparse

import numpy
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm

import torch
import torchvision.transforms
from pycuda.gpuarray import GPUArray
from torchvision import transforms, datasets

import cv2
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
from torch.utils.data import DataLoader, Dataset
from collections import OrderedDict
from tqdm import tqdm
from os import listdir
from os.path import isfile, join

import tensorrt as trt
from PIL import Image
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import time
import os
import cv2
import torch
import argparse
import utility

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
batch_size = 32
class HostDeviceMem(object):
  def __init__(self, host_mem, device_mem):
    self.host = host_mem
    self.device = device_mem

  def __str__(self):
    return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

  def __repr__(self):
    return self.__str__()


def allocate_buffers_for_encoder(engine, batch_size):

  print('[trace] reach func@allocate_buffers')
  inputs = []
  outputs = []
  bindings = []
  stream = cuda.Stream()

  binding_to_type = {}
  binding_to_type['input'] = np.float32
  binding_to_type['output'] = np.float32

  # Current NMS implementation in TRT only supports DataType.FLOAT but
  # it may change in the future, which could brake this sample here
  # when using lower precision [e.g. NMS output would not be np.float32
  # anymore, even though this is assumed in binding_to_type]


  for binding in engine:
    print(f'[trace] current binding: {str(binding)}')
    _binding_shape = engine.get_binding_shape(binding)
    _volume = trt.volume(_binding_shape)
    #size = _volume * engine.max_batch_size
    size = abs(_volume) # dynamic batch size
    print(f'[trace] current binding size: {size}')
    dtype = binding_to_type[str(binding)]
    # Allocate host and device buffers
    host_mem = cuda.pagelocked_empty(size, dtype)
    device_mem = cuda.mem_alloc(host_mem.nbytes)
    # Append the device buffer to device bindings.
    bindings.append(int(device_mem))
    # Append to the appropriate list.
    if engine.binding_is_input(binding):
      inputs.append(HostDeviceMem(host_mem, device_mem))
    else:
      outputs.append(HostDeviceMem(host_mem, device_mem))
  return inputs, outputs, bindings, stream


def generate_trt_engine():
  import pandas as pd

  onnx_file_path = '../../models/point_net_cls-simplified.onnx'
  print(f'[trace] convert onnx file {onnx_file_path} to TensorRT engine')
  if not os.path.exists(onnx_file_path):
    print(f'[trace] target file {onnx_file_path} not exist, exiting')
    exit(-1)

  print(f'[trace] exec@generate_trt_engine')
  sys.path.append("..")
  from pointnet.dataset import ShapeNetDataset, ModelNetDataset
  print(f'[trace] done with import ShapeNetDataset, ModelNetDataset')
  dataset = '../../dataset/shapenetcore_partanno_segmentation_benchmark_v0'
  num_points = 2500
  dataset = ShapeNetDataset(
    root=dataset,
    classification=True,
    npoints=num_points)

  dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=2)

  _imgs0, _labels = iter(dataloader).next()
  cache_file = "calibration.cache"
  # 4 is the lenghth of fp32
  element_bytes = _imgs0.shape[1] * _imgs0.shape[2] * 4
  batch_size = 32

  calib = utility.Calibrator(dataloader, cache_file, element_bytes, batch_size=batch_size)
  # engine = build_engine_from_onnxmodel_int8(onnxmodel, calib)

  mode: utility.CalibratorMode = utility.CalibratorMode.FP32
  TRT_LOGGER = trt.Logger()
  EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

  with trt.Builder(TRT_LOGGER) as builder, \
      builder.create_network(EXPLICIT_BATCH) as network, \
      builder.create_builder_config() as config, \
      trt.OnnxParser(network, TRT_LOGGER) as parser, \
      trt.Runtime(TRT_LOGGER) as runtime:

    # Parse model file
    print("[trace] loading ONNX file from path {}...".format(onnx_file_path))
    with open(onnx_file_path, "rb") as model:
      print("[trace] beginning ONNX file parsing")
      if not parser.parse(model.read()):
        print("[error] failed to parse the ONNX file.")
        for error in range(parser.num_errors):
          print(parser.get_error(error))
        return None
      print("[trace] completed parsing of ONNX file")

    builder.max_batch_size = batch_size
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, utility.GiB(4))

    if mode == utility.CalibratorMode.INT8:
      config.set_flag(trt.BuilderFlag.INT8)
    elif mode == utility.CalibratorMode.FP16:
      config.set_flag(trt.BuilderFlag.FP16)
    elif mode == utility.CalibratorMode.TF32:
      config.set_flag(trt.BuilderFlag.TF32)
    elif mode == utility.CalibratorMode.FP32:
      # do nothing since this is the default branch
      # config.set_flag(trt.BuilderFlag.FP32)
      pass
    else:
      print(f'[trace] unknown calibrator mode: {mode.name}, exit')
      exit(-1)

    config.int8_calibrator = calib
    engine_file_path = f'../../models/siamese_network.{mode.name}.engine'
    input_channel = 3
    input_image_width = 2500
    network.get_input(0).shape = [batch_size, input_channel, input_image_width]
    print(f'[trace] utility.build_engine_common_routine')
    return utility.build_engine_common_routine(network, builder, config, runtime, engine_file_path)
  pass

def main():

  print("[trace] reach the main entry")
  generate_trt_engine()
  print(f'[trace] end of the main point')
  pass

if __name__ == "__main__":
  main()
  pass
