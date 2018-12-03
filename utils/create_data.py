from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import readMTP
import readDFW


class Dataset(object):

  def __init__(self, X, y):
    self.data = X
    self.target = y


def get_data(dataname):
  """Get datasets using keras API and return as a Dataset object."""
  if dataname == 'multipie':
    train, test = cifar10.load_data()
  elif dataname == 'dfw':
    train, test = cifar100.load_data('coarse')
  else:
    raise NotImplementedError('Dataset not supported')

  X = np.concatenate((train[0], test[0]))
  y = np.concatenate((train[1], test[1]))

  y = y.flatten()
  data = Dataset(X, y)
  return data
