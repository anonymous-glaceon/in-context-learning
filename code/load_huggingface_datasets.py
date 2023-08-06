"""Load saved datasets into prompts."""
import random
import numpy as np
import utils


def load_dataset(name):
  train = utils.load_pickle(f'data/{name}_train_data.pickle')
  test = utils.load_pickle(f'data/{name}_test_data.pickle')

  return train, test


def prebuilt_dataset(train_dict, test_dict, num_tests, k, custom_labels):
  """Builds a dataset using existing Huggingface dictionaries.

  Args:
    train_dict: Dictionary of {example: label} for in-context exemplars
    test_dict: Dictionary of {example: label} for eval example
    num_tests: Number of evaluation examples to generate
    k: Number of in-context exemplars per class to provide
    custom_labels: List of custom labels to use

  Returns:
    train_x: Array of shape (n, k, 1)
    train_y: Array of shape (n, k, 1)
    test_x: Array of shape (n, 1)
    test_y: Array of shape (n, 1)
    num_tests: Number of successfully generated tests (n)
  """
  train_x, train_y = [], []

  # Default: use all examples for testing
  if num_tests == -1:
    num_tests = len(test_dict)
  elif num_tests > len(test_dict):
    print(f'Only {len(test_dict)} available test examples')
    num_tests = len(test_dict)

  # Compile inputs
  train_questions = list(train_dict.keys())
  test_questions = list(test_dict.keys())

  # Compute test questions
  random.shuffle(test_questions)
  test_x = test_questions[:num_tests]
  test_y = [custom_labels[int(test_dict[x])] for x in test_x]

  # Reshape test prompts
  test_x = np.reshape(np.array(test_x), (len(test_x), 1))
  test_y = np.reshape(np.array(test_y), (len(test_y), 1))

  # Add examples
  while len(train_x) < num_tests:
    random.shuffle(train_questions)

    train_x_curr = []

    # Loop through each label
    for label in custom_labels:
      # Number of examples for this class
      counter = 0

      # Get first k examples of this class
      for question in train_questions:
        if counter >= k:
          break
        if str(train_dict[question]) == label:
          train_x_curr.append(question)
          counter += 1

      # Not enough examples
      if counter < k:
        print(f'Only found {counter}/{k} examples for class {label}')
        exit()

    assert len(train_x_curr) == k * len(custom_labels)

    random.shuffle(train_x_curr)

    # Compile target labels
    train_y_curr = [custom_labels[int(train_dict[x])] for x in train_x_curr]

    train_x.append(train_x_curr)
    train_y.append(train_y_curr)

  assert len(test_x) == num_tests

  return (
      np.array(train_x),
      np.array(train_y),
      np.array(test_x),
      np.array(test_y),
      num_tests,
  )


#       MAIN        #
n = 100
labels = ['0', '1']
# labels = ['0', '1', '2', '3', '4', '5']  # For TREC

datasets = [
    # 'subj',
    # 'trec',
    # 'sst2',
    'super_glue_rte',
    # 'qnli',
    # 'wsc',
    # 'qqp'
]

for dataset in datasets:
  for n_exemplars in [16]:  # add number of in-context exemplars
    train_dict_curr, test_dict_curr = load_dataset(dataset)
    train_x_res, train_y_res, test_x_res, test_y_res, num_tests_curr = (
        prebuilt_dataset(
            train_dict_curr, test_dict_curr, n, n_exemplars, labels
        )
    )
    inputs_to_targets = utils.convert_to_inputs_targets(
        train_x_res, train_y_res, test_x_res, test_y_res
    )

    print(train_x_res.shape)
    print(train_y_res.shape)
    print(test_x_res.shape)
    print(test_y_res.shape)

    for key, value in inputs_to_targets.items():
      print(key)
      print()
      print(value)
      break

    utils.save_pickle(
        f'data/{dataset}_{num_tests_curr}_{n_exemplars}_'
        + '|'.join(labels)
        + '.pickle',
        inputs_to_targets,
    )

    print('------------------------------------------------')

