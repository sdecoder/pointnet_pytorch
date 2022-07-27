import argparse
import os
import random
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from pointnet.dataset import ShapeNetDataset, ModelNetDataset
from pointnet.model import PointNetCls, feature_transform_regularizer
import torch.nn.functional as F
from tqdm import tqdm
import io
import numpy as np
from torch import nn
import torch.onnx


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
parser.add_argument('--feature_transform', action='store_true', help="use feature transform")

opt = parser.parse_args()
print(opt)

def _load_model():
  dataset = ShapeNetDataset(
    root=opt.dataset,
    classification=True,
    npoints=opt.num_points)

  print(f'[trace] in the main function')
  num_classes = len(dataset.classes)
  print('classes', num_classes)
  classifier = PointNetCls(k=num_classes, feature_transform=opt.feature_transform)

  pth_file = '../utils/cls/cls_model_19.pth'
  if not os.path.exists(pth_file):
    print(f'[trace] specified path file {pth_file} not exist, exit')
    exit(-1)

  classifier.load_state_dict(torch.load(pth_file))
  classifier.eval()
  print(f'[trace] the model has been restored.')
  return classifier


def _export_onnx(torch_model):

  print(f'[trace] start to export the onnx file')
  batch_size = 1
  input = torch.randn(32, 3, 2500, requires_grad=True)
  torch_out = torch_model(input)

  output_file = 'models/point_net_cls.onnx'
  torch.onnx.export(torch_model,  # model being run
                    input,  # model input (or a tuple for multiple inputs)
                    output_file,  # where to save the model (can be a file or file-like object)
                    export_params=True,  # store the trained parameter weights inside the model file
                    opset_version=15,  # the ONNX version to export the model to
                    do_constant_folding=True,  # whether to execute constant folding for optimization
                    input_names=['input'],  # the model's input names
                    output_names=['output'])

  print('f[trace] export to onnx file done')
  pass


def main():

  classifier = _load_model()
  _export_onnx(classifier)
  pass


if __name__ == '__main__':
  main()
