"""Run active learner on classification tasks.

Supported datasets include mnist, letter, cifar10, newsgroup20, rcv1,
wikipedia attack, and select classification datasets from mldata.
See utils/create_data.py for all available datasets.

For binary classification, mnist_4_9 indicates mnist filtered down to just 4 and
9.
By default uses logistic regression but can also train using kernel SVM.
2 fold cv is used to tune regularization parameter over a exponential grid.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle
import sys
from time import gmtime
from time import strftime

import numpy as np
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler

from google.apputils import app
import gflags as flags
from tensorflow import gfile

from sampling_methods.constants import AL_MAPPING
from sampling_methods.constants import get_AL_sampler
from sampling_methods.constants import get_wrapper_AL_mapping
from utils import utils

flags.DEFINE_string("dataset", "multipie", "Dataset name")
flags.DEFINE_string("sampling_method", "margin",
                    ("Name of sampling method to use, can be any defined in "
                     "AL_MAPPING in sampling_methods.constants"))
flags.DEFINE_float(
    "warmstart_size", 0.02,
    ("Can be float or integer.  Float indicates percentage of training data "
     "to use in the initial warmstart model"))
flags.DEFINE_float(
    "batch_size", 0.02,
    ("Can be float or integer.  Float indicates batch size as a percentage "
     "of training data size."))
flags.DEFINE_integer("trials", 1, "Number of curves to create using different seeds")

flags.DEFINE_string("confusions", "0.", "Percentage of labels to randomize")
flags.DEFINE_string("active_sampling_percentage", "1.0", "Mixture weights on active sampling.")
flags.DEFINE_string( "score_method", "logistic", "Method to use to calculate accuracy.")
flags.DEFINE_string( "select_method", "None", "Method to use for selecting points.")
flags.DEFINE_integer("max_dataset_size", 0, ("maximum number of datapoints to include in data"," zero indicates no limit"))
flags.DEFINE_float("train_horizon", 1.0, "how far to extend learning curve as a percent of train")
FLAGS = flags.FLAGS


get_wrapper_AL_mapping()


def generate_one_curve(X,
                       y,
                       sampler,
                       score_model,
                       seed,
                       warmstart_size,
                       batch_size,
                       confusion=0.,
                       active_p=1.0,
                       train_horizon=0.5):
  """Creates one learning curve for both active and passive learning.

  Will calculate accuracy on validation set as the number of training data
  points increases for both PL and AL.
  Caveats: training method used is sensitive to sorting of the data so we
    resort all intermediate datasets

  Args:
    X: training data
    y: training labels
    sampler: sampling class from sampling_methods, assumes reference
      passed in and sampler not yet instantiated.
    score_model: model used to score the samplers.  Expects fit and predict
      methods to be implemented.
    seed: seed used for data shuffle and other sources of randomness in sampler
      or model training
    warmstart_size: float. indicates percentage of train data
      to use for initial model
    batch_size: float or int.  float indicates batch size as a percent of
      training data
    confusion: percentage of labels of one class to flip to the other
    active_p: percent of batch to allocate to active learning
    train_horizon: how long to draw the curve for.  Percent of training data.

  Returns:
    results: dictionary of results for all samplers
    sampler_states: dictionary of sampler objects for debugging
  """
  # TODO(lishal): add option to find best hyperparameter setting first on
  # full dataset and fix the hyperparameter for the rest of the routine
  # This will save computation and also lead to more stable behavior for the
  # test accuracy

  # TODO(lishal): remove mixture parameter and have the mixture be specified as
  # a mixture of samplers strategy
  def select_batch(sampler, uniform_sampler, mixture, N, already_selected,
                   **kwargs):
    n_active = int(mixture * N)
    n_passive = N - n_active
    kwargs["N"] = n_active
    kwargs["already_selected"] = already_selected
    batch_AL = sampler.select_batch(**kwargs)
    already_selected = already_selected + batch_AL
    kwargs["N"] = n_passive
    kwargs["already_selected"] = already_selected
    batch_PL = uniform_sampler.select_batch(**kwargs)
    return batch_AL + batch_PL

  np.random.seed(seed)
  seed_batch = int(warmstart_size * len(y))

  # Load all training data
  (X_train, y_train), (pool_X, pool_Y), (X_test, y_test) = readMTP.personsplit(raw_data,
                                        split_ratio=FLAGS.split_ratio,
                                        target_resolution=GlobalConstants.low_res,
                                        num_train=100) # 100 faces for training, 237 for testing
  X_train, y_train = np.concatenate((X_train, pool_X)), np.concatenate((y_train, pool_Y))

  # Preprocess data
  
  print("active percentage: " + str(active_p) + " warmstart batch: " +
        str(seed_batch) + " batch size: " + str(batch_size) + 
        " confusion: " + str(confusion))

  # Initialize samplers
  uniform_sampler = AL_MAPPING["uniform"](X_train, y_train, seed)
  sampler = sampler(X_train, y_train, seed)

  results = {}
  data_sizes = []
  accuracy = []
  selected_inds = range(seed_batch)

  # If select model is None, use score_model
  same_score_select = True

  n_batches = int(np.ceil((train_horizon * len(y) - seed_batch) *
                          1.0 / batch_size)) + 1
  for b in range(n_batches):
    n_train = seed_batch + min(len(y) - seed_batch, b * batch_size)
    print("Training model on " + str(n_train) + " datapoints")

    assert n_train == len(selected_inds)
    data_sizes.append(n_train)

    # Sort active_ind so that the end results matches that of uniform sampling
    partial_X = X_train[sorted(selected_inds)]
    partial_y = y_train[sorted(selected_inds)]
    
    score_model.fit(partial_X, partial_y)
    acc = score_model.score(X_test, y_test)
    accuracy.append(acc)
    print("Sampler: %s, Accuracy: %.2f%%" % (sampler.name, accuracy[-1]*100))

    n_sample = min(batch_size, len(y) - len(selected_inds))
    select_batch_inputs = {
        "model": score_model,
        "labeled": dict(zip(selected_inds, y_train[selected_inds])),
        "eval_acc": accuracy[-1],
        "X_test": X_val,
        "y_test": y_val,
        "y": y_train
    }
    new_batch = select_batch(sampler, uniform_sampler, active_p, n_sample,
                             selected_inds, **select_batch_inputs)
    selected_inds.extend(new_batch)
    print('Requested: %d, Selected: %d' % (n_sample, len(new_batch)))
    assert len(new_batch) == n_sample
    assert len(list(set(selected_inds))) == len(selected_inds)

  results["accuracy"] = accuracy
  results["selected_inds"] = selected_inds
  results["data_sizes"] = data_sizes
  return results, sampler


def main(argv):
  del argv

  confusions = [float(t) for t in FLAGS.confusions.split(" ")]
  mixtures = [float(t) for t in FLAGS.active_sampling_percentage.split(" ")]
  max_dataset_size = None if FLAGS.max_dataset_size == 0 else FLAGS.max_dataset_size
  X, y = utils.get_data(FLAGS.dataset)
  starting_seed = 42

  for c in confusions:
    for m in mixtures:
      for seed in range(starting_seed, starting_seed + FLAGS.trials):
        sampler = get_AL_sampler(FLAGS.sampling_method)
        score_model = utils.get_model(FLAGS.score_method, seed)
        results, sampler_state = generate_one_curve(
            X, y, sampler, score_model, seed, FLAGS.warmstart_size,
            FLAGS.batch_size, c, m, max_dataset_size,
            False, False, FLAGS.train_horizon)


if __name__ == "__main__":
  app.run()
