import os
import pickle
from abc import ABC

from torch.utils.data import Dataset
from tqdm import tqdm
import pandas as pd
import re
from sklearn import preprocessing
import logging

logger = logging.getLogger(__name__)


class CodeChangeDataset(Dataset, ABC):
    manual_features_columns = ['la', 'ld', 'nf', 'ns', 'nd', 'entropy', 'ndev',
                               'lt', 'nuc', 'age', 'exp', 'rexp', 'sexp', 'fix']

    def __init__(self, tokenizer, args, file_path: str, mode):
        self.tokenizer = tokenizer
        self.args = args
        changes_filename, features_filename = file_path.split("&")

        if changes_filename.find("train") != -1:
            cache_filename = "preprocessed_train_dataset"
        elif changes_filename.find("valid") != -1:
            cache_filename = "preprocessed_valid_dataset"
        else:
            cache_filename = "preprocessed_test_dataset"
        cache_file = os.path.join(self.args.cache_dataset_dir, cache_filename)
        if os.path.exists(cache_file) and not self.args.overwrite_cached_dataset:
            logger.info(f"Loading proprocessed dataset from cache: {cache_file}")
            self.examples = pickle.load(open(cache_file, 'rb'))
        else:
            self.data = []
            pd_data = pd.read_pickle(changes_filename)

            features_data = pd.read_pickle(features_filename)
            features_data = self.convert_dtype_dataframe(features_data, self.manual_features_columns)

            features_data = features_data[['commit_hash'] + self.manual_features_columns]

            scaler = preprocessing.StandardScaler()
            manual_features = scaler.fit_transform(features_data[self.manual_features_columns])
            features_data[self.manual_features_columns] = manual_features

            commit_ids, labels, msgs, codes = pd_data
            for commit_id, label, msg, files in zip(commit_ids, labels, msgs, codes):
                manual_features = features_data[features_data['commit_hash'] == commit_id][
                    self.manual_features_columns].to_numpy().squeeze()
                self.data.append((commit_id, files, msg, label, manual_features))

            # Todo: comment the down-sampling line below after development
            # self.data = self.data[:50]

            self.examples = [self.convert_examples_to_input_tokens(x, no_abstraction=args.no_abstraction) for x in
                             tqdm(self.data, total=len(self.data))]

            def get_text_from_tokens(item):
                res = item
                res.text = self.tokenizer.convert_tokens_to_string(item.input_tokens)
                return res

            self.examples = list(map(get_text_from_tokens, self.examples))
            pickle.dump(self.examples, open(cache_file, 'wb'))

    def __len__(self):
        return len(self.examples)

    @staticmethod
    def convert_dtype_dataframe(df, feature_name):
        df['fix'] = df['fix'].apply(lambda x: float(bool(x)))
        df = df.astype({i: 'float32' for i in feature_name})
        return df

    @staticmethod
    def preprocess_code_line(code, remove_python_common_tokens=False):
        code = code.replace('(', ' ').replace(')', ' ').replace('{', ' ').replace('}', ' ').replace('[', ' ').replace(
            ']',
            ' ').replace(
            '.', ' ').replace(':', ' ').replace(';', ' ').replace(',', ' ').replace(' _ ', '_')

        code = re.sub('``.*``', '<STR>', code)
        code = re.sub("'.*'", '<STR>', code)
        code = re.sub('".*"', '<STR>', code)
        code = re.sub('\d+', '<NUM>', code)

        code = code.split()
        code = ' '.join(code)
        if remove_python_common_tokens:
            new_code = ''
            python_common_tokens = []
            for tok in code.split():
                if tok not in [python_common_tokens]:
                    new_code = new_code + tok + ' '

            return new_code.strip()

        else:
            return code.strip()

    def convert_examples_to_input_tokens(self, item, no_abstraction=True):
        # source
        commit_id, files, msg, label, manual_features = item
        added_tokens = []
        removed_tokens = []
        msg_tokens = self.tokenizer.tokenize(msg)
        msg_tokens = msg_tokens[:min(self.args.max_msg_length, len(msg_tokens))]
        # for file_codes in files:
        file_codes = files
        if no_abstraction:
            added_codes = [' '.join(line.split()) for line in file_codes['added_code']]
        else:
            added_codes = [self.preprocess_code_line(line, False) for line in file_codes['added_code']]
        codes = '[ADD]'.join([line for line in added_codes if len(line)])
        added_tokens.extend(self.tokenizer.tokenize(codes))
        if no_abstraction:
            removed_codes = [' '.join(line.split()) for line in file_codes['removed_code']]
        else:
            removed_codes = [self.preprocess_code_line(line, False) for line in file_codes['removed_code']]
        codes = '[DEL]'.join([line for line in removed_codes if len(line)])
        removed_tokens.extend(self.tokenizer.tokenize(codes))

        input_tokens = msg_tokens + ['[ADD]'] + added_tokens + ['[DEL]'] + removed_tokens

        input_tokens = input_tokens[:512 - 2]
        input_tokens = [self.tokenizer.cls_token] + input_tokens + [self.tokenizer.sep_token]
        return InputFeatures(commit_id=commit_id,
                             input_tokens=input_tokens,
                             manual_features=manual_features,
                             label=int(label))


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, commit_id, input_tokens, label, manual_features, text=None):
        self.commit_id = commit_id
        self.input_tokens = input_tokens
        self.label = label
        self.manual_features = manual_features
        self.text = text


