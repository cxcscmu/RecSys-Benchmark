# Copyright (c) 2024 westlake-repl
# Copyright (c) 2024 Bytedance Ltd. and/or its affiliate
# SPDX-License-Identifier: MIT
# This file has been modified by Junyi Chen.
#
# Original file was released under MIT, with the full license text
# available at https://choosealicense.com/licenses/mit/.
#
# This modified file is released under the same license.

import copy
import pickle
import os
import yaml
from collections import Counter
from logging import getLogger

import numpy as np
import pandas as pd
import torch

from REC.utils import set_color
from REC.utils.enum_type import InputType
from torch_geometric.utils import degree

import pandas as pd
import numpy as np
import csv

def _read_typed_table(path):
    """
    Typed-header reader for two cases:
      1) Tab-delimited rows (e.g., item file with text): sep='\t' (quoting ON).
      2) Whitespace-delimited numeric rows (e.g., *.inter): sep=r'\s+' (no quoting).
    First line must be a typed header like:
      item_id:token  title:token  ...  price:float
      or with FeatureType.*.
    """
    # Read header and peek first data line to detect delimiter
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\n").lstrip("\ufeff")  # strip BOM just in case
        peek = ""
        for line in f:
            if line.strip():
                peek = line.rstrip("\n")
                break
    if not header:
        raise ValueError(f"Empty header in {path}")

    # Parse typed header (split by any whitespace; we’ll choose delimiter for the data below)
    parts = header.split()
    cols, dtypes = [], {}
    for p in parts:
        name, t = (p.split(":", 1) + ["token"])[:2] if ":" in p else (p, "token")
        t = t.lower().replace("featuretype.", "")  # accept FeatureType.FLOAT etc.
        cols.append(name)
        if t == "token":
            dtypes[name] = str
        elif t == "float":
            dtypes[name] = float
        elif t == "int":
            dtypes[name] = np.int64
        else:
            dtypes[name] = str

    # Choose delimiter: if TAB appears in header or first row, treat file as TSV (safe for text)
    is_tsv = ("\t" in header) or ("\t" in peek)

    if is_tsv:
        # Tab-delimited with quoting (handles long titles safely)
        df = pd.read_csv(
            path,
            sep="\t",
            skiprows=1,
            names=cols,
            engine="python",
            quoting=csv.QUOTE_NONE,   # <-- don't interpret " as quote
            quotechar="\u0000",       # <-- impossible quotechar, belt & suspenders
            escapechar="\\",
            dtype={k: v for k, v in dtypes.items() if v is not float},
            keep_default_na=False,    # optional: don't turn bare strings like "NA" into NaN
        )
    else:
        # Plain whitespace-delimited (e.g., *.inter)
        df = pd.read_csv(
            path,
            sep=r"\s+",
            skiprows=1,
            names=cols,
            engine="python",
            dtype={k: v for k, v in dtypes.items() if v is not float},
        )

    # Cast float-ish ints (e.g., 1544671100000.0) to int64 for declared int fields
    for c, t in dtypes.items():
        if t is np.int64 and c in df.columns and pd.api.types.is_float_dtype(df[c]):
            vals = df[c].to_numpy()
            r = np.rint(vals)
            if np.allclose(vals, r, rtol=0, atol=1e-6):
                df[c] = r.astype("int64")

    return df

class Data:
    def __init__(self, config):
        self.config = config
        self.dataset_path = config['data_path']
        self.dataset_name = config['dataset']
        self.data_split = config['data_split']
        self.item_data = config['item_data']
        self.logger = getLogger()
        # self._from_scratch()

    def _from_scratch(self):
        self.logger.info(set_color(f'Loading {self.__class__} from scratch with {self.data_split = }.', 'green'))
        self._load_inter_feat(self.dataset_name, self.dataset_path, self.item_data)
        self._data_processing()

    def _load_inter_feat(self, token, dataset_path, item_data=None):
        inter_feat_path = os.path.join(dataset_path, f"{token}.inter")
        if not os.path.isfile(inter_feat_path):
            raise ValueError(f"File {inter_feat_path} not exist.")

        df = _read_typed_table(inter_feat_path)

        # normalize timestamp if floatish ints
        if "timestamp" in df.columns and pd.api.types.is_float_dtype(df["timestamp"]):
            rounded = np.rint(df["timestamp"])
            if np.allclose(df["timestamp"], rounded, rtol=0, atol=1e-6):
                df["timestamp"] = rounded.astype("int64")

        self.inter_feat = df
        self.logger.info(f"Interaction feature loaded successfully from [{inter_feat_path}]. "
                        f"Columns: {list(df.columns)}; rows: {len(df)}.")

        # optional: item side features if provided as typed table named exactly `item`
        if item_data:
            item_path = os.path.join(dataset_path, f"{item_data}")
            if not os.path.isfile(item_path):
                raise ValueError(f"File {item_path} not exist.")
            self.item_feat = _read_typed_table(item_path)
            self.logger.info(f"Item feature loaded successfully from [{item_data}].")

    def _data_processing(self):

        self.id2token = {}
        self.token2id = {}
        remap_list = ['user_id', 'item_id']
        for feature in remap_list:
            if feature == 'item_id' and self.item_data:
                feats = self.item_feat[feature]
                feats_raw = self.inter_feat[feature]
            else:
                feats = self.inter_feat[feature]
            new_ids_list, mp = pd.factorize(feats)
            mp = ['[PAD]'] + list(mp)
            token_id = {t: i for i, t in enumerate(mp)}
            if feature == 'item_id' and self.item_data:
                _, raw_mp = pd.factorize(feats_raw)
                for x in raw_mp:
                    if x not in token_id:
                        token_id[x] = len(token_id)
                        mp.append(x)
            mp = np.array(mp)

            self.id2token[feature] = mp
            self.token2id[feature] = token_id
            self.inter_feat[feature] = self.inter_feat[feature].map(token_id)

        self.user_num = len(self.id2token['user_id'])
        self.item_num = len(self.id2token['item_id'])
        self.logger.info(f"{self.user_num = } {self.item_num = }")
        self.logger.info(f"{self.inter_feat['item_id'].isna().any() = } {self.inter_feat['user_id'].isna().any() = }")
        self.inter_num = len(self.inter_feat)
        self.uid_field = 'user_id'
        self.iid_field = 'item_id'
        self.user_seq = None
        self.train_feat = None
        self.feat_name_list = ['inter_feat']  # self.inter_feat


    def _load_pre_split_data(self):
        """
        Load data from pre-split files (train/valid/test) in a way that's compatible with HLLM pipeline
        """
        self.logger.info(set_color(f'Loading pre-split data for {self.dataset_name}.', 'green'))
        base = os.path.join(self.dataset_path, self.dataset_name)

        train_path = os.path.join(base, "train.inter")
        valid_path = os.path.join(base, "valid.inter")
        test_path  = os.path.join(base, "test.inter")
        item_path  = os.path.join(base, "item")  # optional

        train_data = _read_typed_table(train_path)
        valid_data = _read_typed_table(valid_path)
        test_data  = _read_typed_table(test_path)
        self.logger.info(f"Train/valid/test loaded from [{train_path}], [{valid_path}], [{test_path}].")
        if os.path.exists(item_path):
            self.item_feat = _read_typed_table(item_path)
            self.logger.info(f"Item details loaded successfully from [{item_path}].")

        all_data = pd.concat([train_data, valid_data, test_data], ignore_index=True)
        self.inter_feat = all_data

        # same mapping pipeline as before
        self._data_processing()

        # map raw ids to internal ids for each split
        for df in (train_data, valid_data, test_data):
            df['user_id'] = df['user_id'].map(self.token2id['user_id'])
            df['item_id'] = df['item_id'].map(self.token2id['item_id'])

        # sort train by timestamp like before
        if 'timestamp' in train_data.columns:
            train_data.sort_values(by='timestamp', ascending=True, inplace=True)

        # Build sequences from the full inter_feat (matches your current logic)
        user_list = self.inter_feat['user_id'].values
        item_list = self.inter_feat['item_id'].values
        timestamp_list = self.inter_feat['timestamp'].values if 'timestamp' in self.inter_feat else np.zeros(len(self.inter_feat), dtype=np.int64)
        grouped_index = self._grouped_index(user_list)

        user_seq, time_seq = {}, {}
        for uid, index in grouped_index.items():
            user_seq[uid] = item_list[index]
            time_seq[uid] = timestamp_list[index]
        self.user_seq = user_seq
        self.time_seq = time_seq

        # train_feat (unchanged logic)
        train_feat = {}
        indices = []
        for index in grouped_index.values():
            indices.extend(list(index)[:-2])
        for k in self.inter_feat:
            train_feat[k] = self.inter_feat[k].values[indices]

        if self.config['MODEL_INPUT_TYPE'] == InputType.AUGSEQ:
            train_feat = self._build_aug_seq(train_feat)
        elif self.config['MODEL_INPUT_TYPE'] == InputType.SEQ:
            train_feat = self._build_seq(train_feat)
        self.train_feat = train_feat

        # valid/test dicts by user
        self.valid_data = {uid: grp['item_id'].tolist() for uid, grp in valid_data.groupby('user_id')}
        self.test_data  = {uid: grp['item_id'].tolist() for uid, grp in test_data.groupby('user_id')}
        self.logger.info(f"Pre-split done. Train sequences: {len(self.user_seq)}, Valid users: {len(self.valid_data)}, Test users: {len(self.test_data)}")

    def build(self, use_pre_split):
        if use_pre_split:
            self._load_pre_split_data()
        else:
            self.logger.info(f"build {self.dataset_name} dataload")
            self.sort(by='timestamp')
            user_list = self.inter_feat['user_id'].values
            item_list = self.inter_feat['item_id'].values
            timestamp_list = self.inter_feat['timestamp'].values
            grouped_index = self._grouped_index(user_list)

            user_seq = {}
            time_seq = {}
            for uid, index in grouped_index.items():
                user_seq[uid] = item_list[index]
                time_seq[uid] = timestamp_list[index]

            self.user_seq = user_seq
            self.time_seq = time_seq
            train_feat = dict()
            indices = []

            for index in grouped_index.values():
                indices.extend(list(index)[:-2])
            for k in self.inter_feat:
                train_feat[k] = self.inter_feat[k].values[indices]

            if self.config['MODEL_INPUT_TYPE'] == InputType.AUGSEQ:
                train_feat = self._build_aug_seq(train_feat)
            elif self.config['MODEL_INPUT_TYPE'] == InputType.SEQ:
                train_feat = self._build_seq(train_feat)

            self.train_feat = train_feat

    def _grouped_index(self, group_by_list):
        index = {}
        for i, key in enumerate(group_by_list):
            if key not in index:
                index[key] = [i]
            else:
                index[key].append(i)
        return index

    def _build_seq(self, train_feat):
        max_item_list_len = self.config['MAX_ITEM_LIST_LENGTH']+1

        uid_list, item_list_index = [], []
        seq_start = 0
        save = False
        user_list = train_feat['user_id']
        user_list = np.append(user_list, -1)
        last_uid = user_list[0]
        for i, uid in enumerate(user_list):
            if last_uid != uid:
                save = True
            if save:
                if (self.data_split is None or self.data_split == True) and i - seq_start > max_item_list_len:
                    offset = (i - seq_start) % max_item_list_len
                    seq_start += offset
                    x = torch.arange(seq_start, i)
                    sx = torch.split(x, max_item_list_len)
                    for sub in sx:
                        uid_list.append(last_uid)
                        item_list_index.append(slice(sub[0], sub[-1]+1))
                else:
                    uid_list.append(last_uid)
                    item_list_index.append(slice(seq_start, i))  # maybe too long but will be truncated in dataloader

                save = False
                last_uid = uid
                seq_start = i

        seq_train_feat = {}
        seq_train_feat['user_id'] = np.array(uid_list)
        seq_train_feat['item_seq'] = []
        seq_train_feat['time_seq'] = []
        for index in item_list_index:
            seq_train_feat['item_seq'].append(train_feat['item_id'][index])
            seq_train_feat['time_seq'].append(train_feat['timestamp'][index])

        return seq_train_feat

    def _build_aug_seq(self, train_feat):
        max_item_list_len = self.config['MAX_ITEM_LIST_LENGTH']+1

        # by = ['user_id', 'timestamp']
        # ascending = [True, True]
        # for b, a in zip(by[::-1], ascending[::-1]):
        #     index = np.argsort(train_feat[b], kind='stable')
        #     if not a:
        #         index = index[::-1]
        #     for k in train_feat:
        #         train_feat[k] = train_feat[k][index]

        uid_list, item_list_index = [], []
        seq_start = 0
        save = False
        user_list = train_feat['user_id']
        user_list = np.append(user_list, -1)
        last_uid = user_list[0]
        for i, uid in enumerate(user_list):
            if last_uid != uid:
                save = True
            if save:
                if i - seq_start > max_item_list_len:
                    offset = (i - seq_start) % max_item_list_len
                    seq_start += offset
                    x = torch.arange(seq_start, i)
                    sx = torch.split(x, max_item_list_len)
                    for sub in sx:
                        uid_list.append(last_uid)
                        item_list_index.append(slice(sub[0], sub[-1]+1))
                else:
                    uid_list.append(last_uid)
                    item_list_index.append(slice(seq_start, i))
                save = False
                last_uid = uid
                seq_start = i

        seq_train_feat = {}
        aug_uid_list = []
        aug_item_list = []
        for uid, item_index in zip(uid_list, item_list_index):
            st = item_index.start
            ed = item_index.stop
            lens = ed - st
            for sub_idx in range(1, lens):
                aug_item_list.append(train_feat['item_id'][slice(st, st+sub_idx+1)])
                aug_uid_list.append(uid)

        seq_train_feat['user_id'] = np.array(aug_uid_list)
        seq_train_feat['item_seq'] = aug_item_list

        return seq_train_feat

    def sort(self, by, ascending=True):

        if isinstance(self.inter_feat, pd.DataFrame):
            self.inter_feat.sort_values(by=by, ascending=ascending, inplace=True)

        else:
            if isinstance(by, str):
                by = [by]

            if isinstance(ascending, bool):
                ascending = [ascending]

            if len(by) != len(ascending):
                if len(ascending) == 1:
                    ascending = ascending * len(by)
                else:
                    raise ValueError(f'by [{by}] and ascending [{ascending}] should have same length.')
            for b, a in zip(by[::-1], ascending[::-1]):
                index = np.argsort(self.inter_feat[b], kind='stable')
                if not a:
                    index = index[::-1]
                for k in self.inter_feat:
                    self.inter_feat[k] = self.inter_feat[k][index]

    @property
    def avg_actions_of_users(self):
        """Get the average number of users' interaction records.

        Returns:
            numpy.float64: Average number of users' interaction records.
        """
        if isinstance(self.inter_feat, pd.DataFrame):
            return np.mean(self.inter_feat.groupby(self.uid_field).size())
        else:
            return np.mean(list(Counter(self.inter_feat[self.uid_field]).values()))

    @property
    def avg_actions_of_items(self):
        """Get the average number of items' interaction records.

        Returns:
            numpy.float64: Average number of items' interaction records.
        """
        if isinstance(self.inter_feat, pd.DataFrame):
            return np.mean(self.inter_feat.groupby(self.iid_field).size())
        else:
            return np.mean(list(Counter(self.inter_feat[self.iid_field]).values()))

    @property
    def sparsity(self):
        """Get the sparsity of this dataset.

        Returns:
            float: Sparsity of this dataset.
        """
        return 1 - self.inter_num / self.user_num / self.item_num

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        info = [set_color(self.dataset_name, 'pink')]
        if self.uid_field:
            info.extend([
                set_color('The number of users', 'blue') + f': {self.user_num}',
                set_color('Average actions of users', 'blue') + f': {self.avg_actions_of_users}'
            ])
        if self.iid_field:
            info.extend([
                set_color('The number of items', 'blue') + f': {self.item_num}',
                set_color('Average actions of items', 'blue') + f': {self.avg_actions_of_items}'
            ])
        info.append(set_color('The number of inters', 'blue') + f': {self.inter_num}')
        if self.uid_field and self.iid_field:
            info.append(set_color('The sparsity of the dataset', 'blue') + f': {self.sparsity * 100}%')

        return '\n'.join(info)

    def copy(self, new_inter_feat):
        """Given a new interaction feature, return a new :class:`Dataset` object,
        whose interaction feature is updated with ``new_inter_feat``, and all the other attributes the same.

        Args:
            new_inter_feat (Interaction): The new interaction feature need to be updated.

        Returns:
            :class:`~Dataset`: the new :class:`~Dataset` object, whose interaction feature has been updated.
        """
        nxt = copy.copy(self)
        nxt.inter_feat = new_inter_feat
        return nxt

    def counter(self, field):
        if isinstance(self.inter_feat, pd.DataFrame):
            return Counter(self.inter_feat[field].values)
        else:
            return Counter(self.inter_feat[field])

    @property
    def user_counter(self):
        return self.counter('user_id')

    @property
    def item_counter(self):
        return self.counter('item_id')

    def get_norm_adj_mat(self):
        r"""Get the normalized interaction matrix of users and items.
        Construct the square matrix from the training data and normalize it
        using the laplace matrix.
        .. math::
            A_{hat} = D^{-0.5} \times A \times D^{-0.5}
        Returns:
            The normalized interaction matrix in Tensor.
        """

        row = torch.tensor(self.train_feat[self.uid_field])
        col = torch.tensor(self.train_feat[self.iid_field]) + self.user_num
        edge_index1 = torch.stack([row, col])
        edge_index2 = torch.stack([col, row])
        edge_index = torch.cat([edge_index1, edge_index2], dim=1)

        deg = degree(edge_index[0], self.user_num + self.item_num)

        norm_deg = 1. / torch.sqrt(torch.where(deg == 0, torch.ones([1]), deg))
        edge_weight = norm_deg[edge_index[0]] * norm_deg[edge_index[1]]

        return edge_index, edge_weight
