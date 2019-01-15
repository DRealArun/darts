import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import genotypes
import torch.utils
# import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
from model import NetworkCIFAR as Network
from PIL import Image
from sklearn.metrics import confusion_matrix
os.environ["CUDA_VISIBLE_DEVICES"]="0"


parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--dataset', type=str, default='cifar10', help='which dataset:\
                    cifar10, mnist, emnist, fashion, svhn, stl10, devanagari')
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=200, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
parser.add_argument('--layers', type=int, default=20, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP_TEST', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='DARTS', help='which architecture to use')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
args = parser.parse_args()

args.save = 'eval-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

CIFAR_CLASSES = 10
MNIST_CLASSES = 10
FASHION_CLASSES = 10
EMNIST_CLASSES = 47
SVHN_CLASSES = 10
STL10_CLASSES = 10
DEVANAGARI_CLASSES = 46 

class_dict = {'cifar10': CIFAR_CLASSES,
              'mnist' : MNIST_CLASSES,
              'emnist': EMNIST_CLASSES,
              'fashion': FASHION_CLASSES,
              'svhn': SVHN_CLASSES,
              'stl10': STL10_CLASSES,
              'devanagari' : DEVANAGARI_CLASSES}

inp_channel_dict = {'cifar10': 3,
                    'mnist' : 1,
                    'emnist': 1,
                    'fashion': 1,
                    'svhn': 3,
                    'stl10': 3,
                    'devanagari' : 1}

def get_train_test_queues(args, train_transform, valid_transform):
  print("Getting",args.dataset,"data")
  if args.dataset == 'cifar10':
    print("Using CIFAR10")
    train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
    valid_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)
  elif args.dataset == 'mnist':
    print("Using MNIST")
    train_data = dset.MNIST(root=args.data, train=True, download=True, transform=train_transform)
    valid_data = dset.MNIST(root=args.data, train=False, download=True, transform=valid_transform)
  elif args.dataset == 'emnist':
    print("Using EMNIST")
    train_data = dset.EMNIST(root=args.data, split='balanced', train=True, download=True, transform=train_transform)
    valid_data = dset.EMNIST(root=args.data, split='balanced', train=False, download=True, transform=valid_transform)
  elif args.dataset == 'fashion':
    print("Using Fashion")
    train_data = dset.FashionMNIST(root=args.data, train=True, download=True, transform=train_transform)
    valid_data = dset.FashionMNIST(root=args.data, train=False, download=True, transform=valid_transform)
  elif args.dataset == 'svhn':
    print("Using SVHN")
    train_data = dset.SVHN(root=args.data, split='train', download=True, transform=train_transform)
    valid_data = dset.SVHN(root=args.data, split='test', download=True, transform=valid_transform)
  elif args.dataset == 'stl10':
    print("Using STL10")
    train_data = dset.STL10(root=args.data, split='train', download=True, transform=train_transform)
    valid_data = dset.STL10(root=args.data, split='test', download=True, transform=valid_transform)
  elif args.dataset == 'devanagari':
    print("Using DEVANAGARI")
    # Ensure dataset is present in the directory args.data. Does not support auto download
    print(args.data)
    train_data = dset.ImageFolder(root=os.path.join(args.data,"Train"), transform=train_transform, loader = grey_pil_loader)
    valid_data = dset.ImageFolder(root=os.path.join(args.data, "Test"), transform=valid_transform, loader = grey_pil_loader)
  else:
    assert False, "Cannot get training queue for dataset"

  train_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=1)

  valid_queue = torch.utils.data.DataLoader(
      valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=1)

  return train_queue, valid_queue

def grey_pil_loader(path):
  # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
  with open(path, 'rb') as f:
      img = Image.open(f)
      img = img.convert('L')
      return img

def main():
  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)

  np.random.seed(args.seed)
  torch.cuda.set_device(args.gpu)
  cudnn.benchmark = True
  torch.manual_seed(args.seed)
  cudnn.enabled=True
  torch.cuda.manual_seed(args.seed)
  logging.info('gpu device = %d' % args.gpu)
  logging.info("args = %s", args)

  genotype = eval("genotypes.%s" % args.arch)
  number_of_classes = class_dict[args.dataset]
  in_channels = inp_channel_dict[args.dataset]
  print(number_of_classes, in_channels)
  model = Network(args.init_channels, number_of_classes, args.layers, args.auxiliary, genotype, in_channels)
  model = model.cuda()

  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()
  optimizer = torch.optim.SGD(
      model.parameters(),
      args.learning_rate,
      momentum=args.momentum,
      weight_decay=args.weight_decay
      )

  train_transform, valid_transform = utils.get_data_transforms(args)
  train_queue, valid_queue = get_train_test_queues(args, train_transform, valid_transform)
  # train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
  # valid_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)

  # train_queue = torch.utils.data.DataLoader(
  #     train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=2)

  # valid_queue = torch.utils.data.DataLoader(
  #     valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=2)

  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))

  for epoch in range(args.epochs):
    scheduler.step()
    logging.info('epoch %d lr %e', epoch, scheduler.get_lr()[0])
    model.drop_path_prob = args.drop_path_prob * epoch / args.epochs

    train_acc, train_obj = train(train_queue, model, criterion, optimizer)
    logging.info('train_acc %f', train_acc)

    valid_acc, valid_obj = infer(valid_queue, model, criterion)
    logging.info('valid_acc %f', valid_acc)

    utils.save(model, os.path.join(args.save, 'weights.pt'))


def train(train_queue, model, criterion, optimizer):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.train()
  # pred_labels = []
  # true_labels = []

  for step, (input, target) in enumerate(train_queue):
    # print("input shape",np.shape(input))
    input = Variable(input).cuda()
    target = Variable(target).cuda(async=True)

    optimizer.zero_grad()
    logits, logits_aux = model(input)
    loss = criterion(logits, target)
    if args.auxiliary:
      loss_aux = criterion(logits_aux, target)
      loss += args.auxiliary_weight*loss_aux
    loss.backward()
    nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
    optimizer.step()

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    # print(np.shape(logits.detach().cpu().numpy()),np.shape(target.detach().cpu().numpy()))
    # pred_labels.extend(np.argmax(logits.detach().cpu().numpy(), axis=1))
    # true_labels.extend(target.detach().cpu().numpy())
    n = input.size(0)
    objs.update(loss.data[0], n)
    top1.update(prec1.data[0], n)
    top5.update(prec5.data[0], n)

    if step % args.report_freq == 0:
      logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  # print("\nSize of predicted and true labels",np.shape(pred_labels),np.shape(true_labels))
  # print("Confusion matrix is :\n")
  # cm = confusion_matrix(true_labels, pred_labels)
  # for i in range(np.shape(cm)[0]):
  #    print(cm[i,:])

  return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.eval()
  pred_labels = []
  true_labels = []

  for step, (input, target) in enumerate(valid_queue):
    input = Variable(input, volatile=True).cuda()
    target = Variable(target, volatile=True).cuda(async=True)

    logits, _ = model(input)
    loss = criterion(logits, target)
    pred_labels.extend(np.argmax(logits.detach().cpu().numpy(), axis=1))
    true_labels.extend(target.detach().cpu().numpy())

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    n = input.size(0)
    objs.update(loss.data[0], n)
    top1.update(prec1.data[0], n)
    top5.update(prec5.data[0], n)

    if step % args.report_freq == 0:
      logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  print("\nSize of predicted and true labels",np.shape(pred_labels),np.shape(true_labels))
  print("Confusion matrix is :\n")
  cm = confusion_matrix(true_labels, pred_labels)
  for i in range(np.shape(cm)[0]):
     print(cm[i,:])
  sys.stdout.flush()
  return top1.avg, objs.avg


if __name__ == '__main__':
  main() 

