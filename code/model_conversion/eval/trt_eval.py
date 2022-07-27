import os
import sys
import glob
import argparse
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm

import torch
import torchvision.transforms
from torchvision import transforms, datasets

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
from tqdm import tqdm

from pointnet.dataset import ShapeNetDataset

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
parser = argparse.ArgumentParser()
parser.add_argument(
  '--batchSize', type=int, default=32, help='input batch size')
parser.add_argument(
  '--num_points', type=int, default=2500, help='input batch size')
parser.add_argument(
  '--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument(
  '--nepoch', type=int, default=250, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='cls', help='output folder')
parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--dataset', type=str, required=True, help="dataset path")
parser.add_argument('--dataset_type', type=str, default='shapenet', help="dataset type shapenet|modelnet40")
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')

'''
parser.add_argument('--feature_transform', action='store_true', help="use feature transform")
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')

parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--embedding-size', type=int, default=32, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--results_path', type=str, default='results/', metavar='N',
                    help='Where to store images')
parser.add_argument('--models', type=str, default='AE', metavar='N',
                    help='Which architecture to use')
parser.add_argument('--dataset', type=str, default='MNIST', metavar='N',
                    help='Which dataset to use')

'''

opt = parser.parse_args()
print(opt)


class HostDeviceMem(object):
  def __init__(self, host_mem, device_mem):
    self.host = host_mem
    self.device = device_mem

  def __str__(self):
    return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

  def __repr__(self):
    return self.__str__()


def load_engine(trt_runtime, engine_path):
  with open(engine_path, 'rb') as f:
    engine_data = f.read()
  engine = trt_runtime.deserialize_cuda_engine(engine_data)
  return engine


def main():
  print("[trace] reach the main entry")
  args = parser.parse_args()
  args.cuda = not args.no_cuda and torch.cuda.is_available()
  device = torch.device("cuda" if args.cuda else "cpu")

  engine_file = '../models/point_net_cls-simplified.engine'
  if not os.path.exists(engine_file):
    print(f'[trace] target engine file {engine_file} not found, exit')
    exit(-1)

  engine = load_engine(trt.Runtime(TRT_LOGGER), engine_file)
  print('[trace] reach func@allocate_buffers')
  inputs = []
  outputs = []
  bindings = []

  binding_to_type = {}
  binding_to_type['input'] = np.float32
  binding_to_type['output'] = np.float32
  binding_to_type['onnx::MatMul_113'] = np.float32

  for binding in engine:
    print(f'[trace] current binding: {str(binding)}')
    _binding_shape = engine.get_binding_shape(binding)
    _volume = trt.volume(_binding_shape)
    size = _volume * engine.max_batch_size
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

  print('[trace] initiating TensorRT object')
  test_dataset = ShapeNetDataset(
    root=opt.dataset,
    classification=True,
    split='test',
    npoints=opt.num_points,
    data_augmentation=False)

  testdataloader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=opt.batchSize,
    shuffle=True,
    num_workers=int(opt.workers))

  batch_size = 1
  total_correct = 0
  total_testset = 0
  context = engine.create_execution_context()
  stream = cuda.Stream()
  for i, data in tqdm(enumerate(testdataloader, 0)):
    print(f'[trace] current working index: {i}')
    points, target = data
    target = target[:, 0]
    points = points.transpose(2, 1)
    if points.size() != (32, 3, 2500):
      print(f'[warn] wrong data input found at index: {i}')
      continue
    np.copyto(inputs[0].host, points.ravel())
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    stream.synchronize()

    retObj = torch.from_numpy(outputs[1].host)
    tensor = torch.reshape(retObj, (32, 16))
    pred_choice = tensor.data.max(1)[1]
    correct = pred_choice.eq(target.data).cpu().sum()
    total_correct += correct.item()
    total_testset += points.size()[0]

    '''
    #points, target = points.cuda(), target.cuda()
    classifier = classifier.eval()
    pred, _, _ = classifier(points)
    pred_choice = pred.data.max(1)[1]
    correct = pred_choice.eq(target.data).cpu().sum()
    total_correct += correct.item()
    total_testset += points.size()[0]
    '''

  print("final accuracy {}".format(total_correct / float(total_testset)))
  print(f'[trace] end of the main point')
  pass


if __name__ == "__main__":
  main()
  pass
