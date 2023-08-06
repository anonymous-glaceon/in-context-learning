"""Utility functions."""
import os
import pickle
import numpy as np


def save_pickle(filename, data):
  if os.path.exists(filename):
    return

  with open(filename, 'wb') as file:
    pickle.dump(data, file)
    print(f'Saved pickle at {filename}')


def load_pickle(filename):
  with open(filename, 'rb') as file:
    return pickle.load(file)


def convert_to_inputs_targets(train_x, train_y, test_x, test_y):
  """Uses the default format and converts inputs/targets into prompts.

  Args:
    train_x: Array of shape (n, k, 1)
    train_y: Array of shape (n, k, 1)
    test_x: Array of shape (n, 1)
    test_y: Array of shape (n, 1)

  Returns:
    Dictionary of prompts to targets
  """
  inputs_to_targets = {}

  # Construct each input
  for train_x_curr, train_y_curr, test_x_curr, test_y_curr in zip(
      train_x, train_y, test_x, test_y
  ):
    input_curr = ''

    target = test_y_curr
    if not isinstance(test_y_curr, str) and not isinstance(
        test_y_curr, np.str_
    ):
      target = str(test_y_curr[0])

    # Construct training examples
    for example, label in zip(train_x_curr, train_y_curr):
      if not isinstance(label, str) and not isinstance(label, np.str_):
        label = label[0]

      if isinstance(example, np.str_):
        input_curr += f'Input: {example}\nOutput: {label}\n'
      else:
        example = [str(x) for x in example]
        example = ', '.join(example)
        input_curr += f'Input: {example}\nOutput: {label}\n'

    # Add test example
    test_x_curr = [str(x) for x in test_x_curr]
    test_x_curr = ', '.join(test_x_curr)
    input_curr += f'Input: {test_x_curr}\nOutput:'

    inputs_to_targets[input_curr] = target

  return inputs_to_targets
