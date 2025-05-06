"""
We used RecBole's evaluation code:https://github.com/RUCAIBox/RecBole/blob/master/recbole/evaluator/metrics.py.
########################
"""


import argparse 
import os
import numpy as np
from tqdm import tqdm 

import torch
import numpy as np



def recall(pos_index, pos_len):
    return np.cumsum(pos_index, axis=1) / pos_len.reshape(-1, 1)


def ndcg(pos_index,pos_len):
    len_rank = np.full_like(pos_len, pos_index.shape[1])
    idcg_len = np.where(pos_len > len_rank, len_rank, pos_len)
    iranks = np.zeros_like(pos_index, dtype=np.float)
    iranks[:, :] = np.arange(1, pos_index.shape[1] + 1)
    idcg = np.cumsum(1.0 / np.log2(iranks + 1), axis=1)
    for row, idx in enumerate(idcg_len):
        idcg[row, idx:] = idcg[row, idx - 1]
    ranks = np.zeros_like(pos_index, dtype=np.float)
    ranks[:, :] = np.arange(1, pos_index.shape[1] + 1)
    dcg = 1.0 / np.log2(ranks + 1)
    dcg = np.cumsum(np.where(pos_index, dcg, 0), axis=1)

    result = dcg / idcg
    return result


def get_metrics_dict(rank_indices, n_seq, n_item, Ks, target_item_list, sparse=False):
    if sparse: 
        return get_metrics_dict_sparse(rank_indices, n_seq, n_item, Ks, target_item_list)
    rank_indices = torch.tensor(rank_indices)
    pos_matrix = torch.zeros([n_seq,n_item], dtype=torch.int)
    for i in range(n_seq):
        # item id starts from 1
        pos = target_item_list[i] - 1
        pos_matrix[i][pos] = 1
    pos_len_list = pos_matrix.sum(dim=1, keepdim=True)
    pos_idx = torch.gather(pos_matrix, dim=1, index=rank_indices)
    pos_idx = pos_idx.to(torch.bool).cpu().numpy()
    pos_len_list = pos_len_list.squeeze(-1).cpu().numpy()
    recall_result = recall(pos_idx, pos_len_list)
    avg_recall_result = recall_result.mean(axis=0)
    ndcg_result = ndcg(pos_idx, pos_len_list)
    avg_ndcg_result = ndcg_result.mean(axis=0)
    metrics_dict = {}
    for k in Ks:
        metrics_dict[k] = {}
        metrics_dict[k]['recall'] = round(avg_recall_result[k - 1], 4)
        metrics_dict[k]['ndcg'] = round(avg_ndcg_result[k - 1], 4)
    return metrics_dict



def retrieval_result_read(fname, with_distance=False):
    """
    Read the binary ground truth file in DiskANN format. 
    If with_distance is given as True, distances array will be read. 
    """
    n, d = map(int, np.fromfile(fname, dtype="uint32", count=2))
    print("n: ", n)
    print("d: ", d)

    # validity check 
    if with_distance: 
        assert os.stat(fname).st_size == 8 + n * d * (4 + 4)
    else: 
        assert os.stat(fname).st_size == 8 + n * d * 4
    
    f = open(fname, "rb")
    f.seek(4+4)
    I, D = None, None
    I = np.fromfile(f, dtype="int32", count=n * d).reshape(n, d)
    if with_distance: 
        D = np.fromfile(f, dtype="float32", count=n * d).reshape(n, d)
    return I, D



def read_target_file(fname): 
    target_item_list = []
    lines = open(fname, 'r').readlines()
    for line in lines[1:]: 
        parts = line.strip().split("\t")
        target_item_list.append(int(parts[1]) + 1) # the metric calculation assume 1-indexing
    return target_item_list



def main():

    parser = argparse.ArgumentParser(description="Additional parser over encoding")
    parser.add_argument("--valid", action="store_true", help="Whether or not in the validation mode")
    parser.add_argument("--n_item", type=int, default=87208655, help="Number of items in the corpus")
    parser.add_argument("--retrieval_result_path", type=str, help="Path of the DiskANN format binary")
    parser.add_argument("--target_path", type=str, help="Path of the test_target.tsv")
    parser.add_argument('--Ks', nargs='?', default='[1, 10, 50, 100]', help='Calculate metric@K when evaluating.')
    args = parser.parse_args()

    # get the retrieved I (ground truth KNNS from DiskANN utility app)
    I, _ = retrieval_result_read(args.retrieval_result_path)
    I = I[:, :100]
    I = I.astype(np.int64)

    # get targets 
    target_item_list = read_target_file(args.target_path)

    n_seq = I.shape[0]
    Ks = eval(args.Ks)

    batching = True
    if batching: 
        metrics_dict = None
        bz = 4
        for i in tqdm(range(0, n_seq, bz)): 
            I_batch = I[i:i+bz, :]
            target_item_batch = target_item_list[i:i+bz]
            n_seq_batch = I_batch.shape[0]
            metrics_dict_batch = get_metrics_dict(I_batch, n_seq_batch, args.n_item, Ks, target_item_batch)
            if not metrics_dict: 
                metrics_dict = metrics_dict_batch
                for k in metrics_dict: 
                    for metric in metrics_dict[k]: 
                        metrics_dict[k][metric] *= n_seq_batch 
            else: 
                # add to the results 
                for k in metrics_dict: 
                    for metric in metrics_dict[k]: 
                        metrics_dict[k][metric] += metrics_dict_batch[k][metric]*n_seq_batch
    
        for k in metrics_dict: 
            for metric in metrics_dict[k]: 
                metrics_dict[k][metric] /= n_seq 
                metrics_dict[k][metric] = round(metrics_dict[k][metric], 4)
    else: 
        metrics_dict = get_metrics_dict(I, n_seq, args.n_item, Ks, target_item_list)


    # custom message
    if args.valid: 
        out_msg = "Valid: "
    else: 
        out_msg = "Test: "
    for k in Ks: 
        out_msg += f"Recall@{k}: {metrics_dict[k]['recall']} "
    for k in Ks: 
        out_msg += f"NDCG@{k}: {metrics_dict[k]['ndcg']} "

    print(out_msg)


if __name__ == "__main__": 
    main()