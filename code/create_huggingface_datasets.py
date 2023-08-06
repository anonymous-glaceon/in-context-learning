"""Downloads data from HuggingFace."""
import random
from datasets import load_dataset
from utils import save_pickle


# returns 2 dictionaries of key: input, value: label
def build_datasets(dataset_dict, label_name, text_fields, labels_to_skip):
  """Uses inputted dataset dictionary to build dictionary of train/test values.

  Args:
    dataset_dict: Dictionary of dataset splits to data dictionaries
    label_name: String name of index for getting labels
    text_fields: List of indices for getting the text fields
    labels_to_skip: List of labels whose examples should not get processed

  Returns:
    train_dict: Dictionary of training examples and their labels
    test_dict: Dictionary of testing examples and their labels
  """
  train_dict, test_dict = {}, {}
  train_data, test_data = None, None

  if 'validation' in dataset:
    train_data, test_data = dataset_dict['train'], dataset_dict['validation']
  elif 'test' in dataset:
    train_data, test_data = dataset_dict['train'], dataset_dict['test']
  else:
    temp = list(dataset_dict['train'])
    random.shuffle(temp)
    num_test_examples = int(.2 * len(temp))
    train_data, test_data = temp[:-num_test_examples], temp[-num_test_examples:]

  for data, dict_ in zip([train_data, test_data], [train_dict, test_dict]):
    for i in range(len(data)):
      text = '\n'.join([data[i][x] for x in text_fields])
      label = data[i][label_name]

      if str(label) in labels_to_skip:
        continue

      dict_[text] = str(label)

      # Print one sample
      if i == 0:
        print(text)
        print(label)
        print('------------------------')

  print(f'Found {len(train_dict)} training examples')
  print(f'Found {len(test_dict)} testing examples')

  return train_dict, test_dict


#       MAIN        #
dataset_name = 'super_glue'
subset = 'rte'
label_type = 'label'
fields = ['premise', 'hypothesis']
skip_labels = []


dataset = None
if not subset:
  dataset = load_dataset(dataset_name)
else:
  dataset = load_dataset(dataset_name, name=subset)

train, test = build_datasets(dataset, label_type, fields, skip_labels)

if not subset:
  dataset_name = dataset_name.split('/')[-1]
else:
  dataset_name += '_' + subset.split('/')[-1]

save_pickle(f'data/{dataset_name}_train_data.pickle', train)
save_pickle(f'data/{dataset_name}_test_data.pickle', test)
