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


class Data:
    def __init__(self, config):
        self.config = config
        self.dataset_path = config['data_path'] # /data/group_data/cx_group/REC/data/HLLM 
        self.dataset_name = config['dataset'] # amzn-books
        self.data_split = config['data_split']
        self.item_data = config['item_data'] # amzn-books/item_details
        self.logger = getLogger()
        # self._from_scratch()

    def _from_scratch(self):
        self.logger.info(set_color(f'Loading {self.__class__} from scratch with {self.data_split = }.', 'green'))
        self._load_inter_feat(self.dataset_name, self.dataset_path, self.item_data)
        self._data_processing()

    def _load_inter_feat(self, token, dataset_path, item_data=None):
        inter_feat_path = os.path.join(dataset_path, f'{token}.csv')
        if not os.path.isfile(inter_feat_path):
            raise ValueError(f'File {inter_feat_path} not exist.')

        df = pd.read_csv(
            inter_feat_path, delimiter=',', dtype={'user_id': str, 'item_id': str, 'timestamp': int}, header=0)
        self.logger.info(f'Interaction feature loaded successfully from [{inter_feat_path}].')
        self.inter_feat = df

        if item_data:
            item_data_path = os.path.join(dataset_path, f'{item_data}.csv')
            item_df = pd.read_csv(
                item_data_path, delimiter=',', dtype={'item_id': str}, header=0
            )
            self.item_feat = item_df
            self.logger.info(f'Item feature loaded successfully from [{item_data}].')
            
            if 'item_id' in self.token2id:
       	      self.item_feat['item_id'] = self.item_feat['item_id'].map(self.token2id['item_id'])
            missing = self.item_feat['item_id'].isna().sum()
            if missing > 0:
              self.logger.warning(f"{missing} items in item_details.csv could not be mapped to interaction data")

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
        """
        self.logger.info(set_color(f'Loading pre-split data for {self.dataset_name}.', 'green'))
        
        # Define file paths for pre-split data
        train_path = os.path.join(self.dataset_path, self.dataset_name, 'train_interactions.csv')
        valid_path = os.path.join(self.dataset_path, self.dataset_name, 'valid_interactions.csv')
        test_path = os.path.join(self.dataset_path, self.dataset_name, 'test_interactions.csv')
        item_details_path = os.path.join(self.dataset_path, self.dataset_name, 'item_details.csv')
        
        # Load the data to match expected format
        train_data = pd.read_csv(
            train_path, delimiter=',', dtype={'user_id': str, 'item_id': str, 'timestamp': int}
        )
        self.logger.info(f'Train interactions loaded successfully from [{train_path}].')
        
        valid_data = pd.read_csv(
            valid_path, delimiter=',', dtype={'user_id': str, 'item_id': str, 'timestamp': int}
        )
        self.logger.info(f'Valid interactions loaded successfully from [{valid_path}].')
        
        test_data = pd.read_csv(
            test_path, delimiter=',', dtype={'user_id': str, 'item_id': str, 'timestamp': int}
        )
        self.logger.info(f'Test interactions loaded successfully from [{test_path}].')
        print(f' the length of training data is {len(train_data)}')
        print(f' the length of valid data is {len(valid_data)}')
        print(f' the length of test data is {len(test_data)}')
        
        # Load item details if needed - use same format as expected in original code
        if os.path.exists(item_details_path):
            item_data = pd.read_csv(
                item_details_path, delimiter=',', dtype={'item_id': str}, header=0
            )
            self.item_feat = item_data
            self.logger.info(f'Item details loaded successfully from [{item_details_path}].')
        
        # Combine all interactions for consistent ID mapping
        all_data = pd.concat([train_data, valid_data, test_data], ignore_index=True)
        self.inter_feat = all_data
        
        # Apply the same ID mapping process as the original code
        self._data_processing()
        
        # Now process the same way as in the regular build method
        self.logger.info(f"Building dataloader from pre-split files for {self.dataset_name}")
        
        # Map the raw IDs to internal IDs for all datasets
        train_data['user_id'] = train_data['user_id'].map(self.token2id['user_id'])
        train_data['item_id'] = train_data['item_id'].map(self.token2id['item_id'])
        valid_data['user_id'] = valid_data['user_id'].map(self.token2id['user_id'])
        valid_data['item_id'] = valid_data['item_id'].map(self.token2id['item_id'])
        test_data['user_id'] = test_data['user_id'].map(self.token2id['user_id'])
        test_data['item_id'] = test_data['item_id'].map(self.token2id['item_id'])
        
        # Sort train data by timestamp
        train_data.sort_values(by='timestamp', ascending=True, inplace=True)
        

        all_sorted = self.inter_feat.sort_values(by=['user_id', 'timestamp'])
        user_list  = all_sorted['user_id'].values
        item_list  = all_sorted['item_id'].values
        timestamp_list = all_sorted['timestamp'].values

        grouped_index = self._grouped_index(user_list)
        
        user_seq = {}
        time_seq = {}
        for uid, index in grouped_index.items():
            user_seq[uid] = item_list[index]
            time_seq[uid] = timestamp_list[index]
        
        self.user_seq = user_seq
        self.time_seq = time_seq
        

        train_sorted = train_data.sort_values(by=['user_id', 'timestamp'])
        train_feat   = {k: train_sorted[k].values for k in train_sorted.columns}
        
        if self.config['MODEL_INPUT_TYPE'] == InputType.AUGSEQ:
            train_feat = self._build_aug_seq(train_feat)
        elif self.config['MODEL_INPUT_TYPE'] == InputType.SEQ:
            train_feat = self._build_seq(train_feat)
        
        self.train_feat = train_feat
        
        self.valid_data = {}
        grouped_valid = valid_data.groupby('user_id')
        for uid, group in grouped_valid:
            self.valid_data[uid] = group['item_id'].values.tolist()
        
        self.test_data = {}
        grouped_test = test_data.groupby('user_id')
        for uid, group in grouped_test:
            self.test_data[uid] = group['item_id'].values.tolist()
        
        self.logger.info(f"Pre-split data loading completed. Train sequences: {len(self.user_seq)}, Valid users: {len(self.valid_data)}, Test users: {len(self.test_data)}")
        
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
