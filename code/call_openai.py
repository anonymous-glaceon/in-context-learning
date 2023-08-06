"""Run OpenAI models on prompts."""
import datetime
import time

import openai
from sklearn import metrics
from utils import load_pickle


def run_on_pickle_file(filepath, model_engine):
  """Runs the given model on all inputs from the specified file.

  Args:
    filepath: Path to pickle file to run on
    model_engine: Model type to use (e.g., text-ada-001)

  Returns:
    accuracy: Model accuracy
  """
  # Load dictionary from save file
  inputs_to_targets = load_pickle(filepath)

  inputs, targets = [], []

  for input_, target in inputs_to_targets.items():
    inputs.append(input_)
    targets.append(target)

  predictions = get_predictions(inputs, model_engine, custom_stop='\n')
  assert len(targets) == len(predictions)

  accuracy = metrics.accuracy_score(targets, predictions)

  print(f'Model: {model_engine} | Accuracy: {round(accuracy, 4)}')

  # Get timestamp
  now = datetime.datetime.now()
  date_time = now.strftime('%m|%d|%Y|%H|%M|%S')

  # Write results
  out_path = 'results/' + filepath.split('/')[-1].split('.')[0]
  with open(f'{out_path}_{model_engine}_{date_time}.txt', 'w') as file:
    file.write('-----Accuracy-----\n')
    file.write(str(accuracy))
    file.write('\n')

    file.write('-----Inputs-----\n')
    file.write(str(inputs))
    file.write('\n')
    file.write('\n')

    file.write('-----Targets-----\n')
    file.write(str(targets))
    file.write('\n')
    file.write('\n')

    file.write('-----Predictions-----\n')
    file.write(str(predictions))
    file.write('\n')
    file.write('\n')

  return accuracy


# Inputs: list of string inputs
# Model: str of model engine to use
# custom_stop: str
# max_tokens: int, default = 256
def get_predictions(inputs, model, custom_stop='', max_tokens=256):
  """Given list of inputs, return GPT-3 outputs."""
  outputs = []
  num_inputs = len(inputs)
  rate_limit_sleep = 0
  stop_tokens = ['Q:']

  if custom_stop:
    stop_tokens.append(custom_stop)

  for i, input_text in enumerate(inputs):
    successful_single_input = False

    while not successful_single_input:
      try:
        time.sleep(rate_limit_sleep)
        response = openai.Completion.create(
            engine=model,
            prompt=input_text,
            temperature=0,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=stop_tokens,
        )
        successful_single_input = True
        rate_limit_sleep = max(0, rate_limit_sleep - 1)
      except openai.error.RateLimitError:
        rate_limit_sleep = min(50, rate_limit_sleep + 1)
        print(f'======> Rate limit error on example #{i+1}: Increasing '
              f'sleep to {rate_limit_sleep}s <======')
      except openai.error.ServiceUnavailableError:
        rate_limit_sleep = min(rate_limit_sleep + 1, 20)
        print(f'======> Service unavailable error: Increasing sleep to '
              f'{rate_limit_sleep} <======')
      except openai.error.InvalidRequestError:
        rate_limit_sleep = int(rate_limit_sleep / 2)
        input_text = input_text[200:]
        print('---------------------------------------------------------------')
        print('---------------------CONTEXT LENGTH ERROR----------------------')
        print('---------------------------------------------------------------')
        print('Shortening context')
      except openai.error.APIConnectionError:
        rate_limit_sleep = min(rate_limit_sleep + 1, 20)
        print('======> Connection Error <======')
      except openai.error.APIError:
        rate_limit_sleep = min(rate_limit_sleep + 1, 20)
        print('======> API Error <======')
      except KeyboardInterrupt:
        print('Interrupted')
        try:
          exit()
        except SystemExit:
          exit()
      except:
        print('Error')

    output_str = response['choices'][0]['text'].strip()
    outputs.append(output_str)

    print(f'{i+1}/{num_inputs} at rate_limit_sleep={rate_limit_sleep} '
          f'\n[{model} INPUT]:\n\n{input_text}\n\n'
          f'[{model} OUTPUT]:\n\n{output_str}\n')

  return outputs


#       MAIN        #
datasets = [
    # 'sst2',
    # 'subj',
    # 'rte',
    # 'qqp',
    # 'qnli',
    # 'trec',
    # 'wsc',
    # 'financial_phrasebank_sentences_allagree',
    # 'ethos_binary',
    # 'super_glue_rte'
]

label_styles = [
    # '0|1|2|3|4|5',
    # '5|4|3|2|1|0',
    # 'A|B|C|D|E|F',
    # 'F|E|D|C|B|A',
    # 'Apple|Orange|Banana|Peach|Cherry|Kiwi',
    # 'Kiwi|Cherry|Peach|Banana|Orange|Apple',
    # 'Foo|Bar|Iff|Roc|Ket|Dal',
    # 'Dal|Ket|Roc|Iff|Bar|Foo',
    # '0|1',
    # '1|0',
    # 'A|B',
    # 'B|A',
    # 'Apple|Orange',
    # 'Orange|Apple',
    'Foo|Bar',
    # 'Bar|Foo',
    # 'Abbreviation|Entity|DescriptionandAbstractConcept|HumanBeing|Location|NumericValue'
    # 'NegativeSentiment|PositiveSentiment'
]

ks = [
    # 1,
    # 2,
    # 3,
    # 4,
    # 8,
    16,
    # 32,
    # 64,
]

models = [
    'code-davinci-002',
    # 'code-davinci-001',
    # 'code-cushman-001',
    # 'text-curie-001',
    # 'text-babbage-001',
    # 'text-ada-001',
    # 'text-davinci-002',
    # 'text-davinci-001',
    # 'davinci',
    # 'curie',
    # 'babbage',
    # 'ada'
]

incorrects = [
    '0',
    # '025',
    # '05',
    # '075',
    # '10',
]

permuteds = [
    '0',
    # '025',
    # '05',
    # '075',
    # '10'
]

formats = [
    'Default'
    # '([x],[y])',
    # 'Q:[x]A:[y]',
    # 'Question:[x]Answer:[y]',
    # 'Student:[x]Teacher:[y]',
    # 'X=[x]Y=[y]',
    # '[x]->[y]',
    # '[x]FooorBar?answer:[y]'
]

num_eval = 100

key_num = 1
api_key_path = f'openai_api_key_path_{key_num}.txt'
openai.api_key_path = api_key_path

print(f'Datasets: {datasets}')
print(f'Label Styles: {label_styles}')
print(f'k = {ks}')
print(f'Models: {models}')
print(f'Noise: {incorrects}')
print(f'Formats: {formats}')
print(f'Eval Examples: {num_eval}')
time.sleep(10)

for dataset_name in datasets:
  for k in ks:
    for label_style in label_styles:
      for incorrect in incorrects:
        for format_style in formats:
          for permuted in permuteds:
            path = f'data/{dataset_name}_{num_eval}_{k}_{label_style}'

            if incorrect != '0':
              path += f'_incorrect{incorrect}'

            if format_style != 'Default':
              path += f'_format{format_style}'

            if permuted != '0':
              path += f'_permuted{permuted}'

            path += '.pickle'

            for model_curr in models:
              acc = run_on_pickle_file(path, model_curr)
