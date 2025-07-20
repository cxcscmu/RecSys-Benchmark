import argparse
import numpy as np 
import sys
import os 

from tqdm import tqdm 

import torch


def read_embed_shape(filename):
    with open(filename, "rb") as f:
        nvecs, dim = np.fromfile(f, count=2, dtype=np.int32)
        return nvecs, dim


def read_fbin(filename, start_idx=0, chunk_size=None):
    with open(filename, "rb") as f:
        nvecs, dim = np.fromfile(f, count=2, dtype=np.int32)
        print("number of queries: ", nvecs)
        print("dimension: ", dim)
        f.seek(4+4)
        nvecs = (nvecs - start_idx) if chunk_size is None else chunk_size
        arr = np.fromfile(f, count=nvecs * dim, dtype=np.float32,
                          offset=start_idx * 4 * dim)
    return arr.reshape(nvecs, dim)


def write_pred_to_binary(prediction, output_binary_path): 

    nrows, ncols = prediction.shape

    with open(output_binary_path, "wb") as f:
        offset = 0
        # header
        f.write(nrows.to_bytes(4, 'little')) # number of points
        f.write(ncols.to_bytes(4, 'little')) # dimension
        offset += 8
        
        # ids
        f.seek(offset)
        f.write(prediction.astype('uint32').tobytes()) 
        offset += prediction.nbytes


def retrieval_result_read(fname, e2e=False):
    """
    Read the binary ground truth file in DiskANN format. 
    If e2e is given as True, no distances array will be read (end of end qrel scenario). 
    """
    n, d = map(int, np.fromfile(fname, dtype="uint32", count=2))
    print("n: ", n)
    print("d: ", d)
    # validity check 
    if e2e: 
        assert os.stat(fname).st_size == 8 + n * d * 4
    else: 
        assert os.stat(fname).st_size == 8 + n * d * (4 + 4)
    
    f = open(fname, "rb")
    f.seek(4+4)

    I, D = None, None
    I = np.fromfile(f, dtype="int32", count=n * d).reshape(n, d)
    if not e2e: 
        D = np.fromfile(f, dtype="float32", count=n * d).reshape(n, d)

    return I, D


def get_seq_data(seq_data_path): 
    seq_data = []
    lines = open(seq_data_path, 'r').readlines()
    for line in tqdm(lines[1:]):
        history_titles = list()
        line = line.strip().split('\t')
        # read data 
        session_id = line[0]
        history = line[1].split(",")
        history = [x + 1 for x in map(int, history)] # integer internal ids with 1-indexing 
        # directly user internal id -> use the internal id to index into item embed
        seq_data.append(history)
    # seq_data = [[1, 2, 3, 4, 5, 6, 7]] ###
    return seq_data


def predict(seq_emb_path, item_emb_path, k, output_binary_path, seq_path=None):

    # sequence query emb 
    seq_emb = read_fbin(seq_emb_path)
    seq_emb = torch.from_numpy(seq_emb).to(torch.bfloat16)
    # hllm normalization 
    seq_emb = seq_emb / seq_emb.norm(dim=-1, keepdim=True)
    num_seq_emb = seq_emb.shape[0]
    seq_batch_size = 128

    full_scores = []

    # iteratively read item embed and compute scores 
    end, vector_dim = read_embed_shape(item_emb_path)
    start = 0 
    shard = 10000 # 1000000
    # start and end is given as 
    while start < end: 

        # read item emb shard 
        if end - start < shard:
            shard = end - start
        print(f"reading the ({start},{start+shard}) vectors...")
        sys.stdout.flush()
        item_emb = read_fbin(item_emb_path, start_idx=start, chunk_size=shard)
        item_emb = torch.from_numpy(item_emb).to(torch.bfloat16)
        # hllm normalization 
        item_emb = item_emb / item_emb.norm(dim=-1, keepdim=True)

        # batched seq for precision purposes 
        current_scores = []
        for seq_start in range(0, num_seq_emb, seq_batch_size):
            seq_end = min(seq_start + seq_batch_size, num_seq_emb)
            seq_emb_batch = seq_emb[seq_start:seq_end]  # Slice batch
            
            # Compute shard scores
            scores = torch.matmul(seq_emb_batch.to("cuda"), item_emb.to("cuda").t())
            current_scores.append(scores)

        del item_emb, seq_emb_batch, scores

        current_scores = torch.cat(current_scores).cpu()
        full_scores.append(current_scores)

        del current_scores

        # # shard scores
        # scores = torch.matmul(seq_emb.to("cuda"), item_emb.to("cuda").t())   
        # full_scores.append(scores)

        start += shard 

    
    del seq_emb 
    print(f"finish all shards...")
    sys.stdout.flush()
    # merge scores from different shards 
    full_scores = torch.cat(full_scores, dim=1) # num_seq * num_item 
    full_scores[:, 0] = -np.inf # remove dummy pad 

    # remove historical items 
    if seq_path: 
        seq_data = get_seq_data(seq_path)
        for i in range(len(seq_data)): 
            history = seq_data[i]
            full_scores[i][history] = -np.inf

    # full_scores = full_scores.view(-1, end)
    print(f"score shape: {full_scores.shape}")
    sys.stdout.flush()

    # _, topk_idx = torch.topk(full_scores, k, dim=-1)  # num_seq x k

    all_indices = []
    num_seqs = full_scores.size(0)
    bz = 10 
    for i in range(0, num_seqs, bz):
        batch_scores = full_scores[i:i + bz].to("cuda")
        _, topk_inds = torch.topk(batch_scores, k, dim=-1)
        all_indices.append(topk_inds)
        del batch_scores, topk_inds
    topk_indices = torch.cat(all_indices, dim=0)

    del full_scores
    prediction = topk_indices.cpu().numpy()
    prediction = prediction - 1 # convert to 0-indexing to align with ClueWeb-Reco format 
    print(f"prediction shape: {prediction.shape}")
    print(f"prediction[:5, :5]: {prediction[:5, :5]}")
    sys.stdout.flush()

    write_pred_to_binary(prediction, output_binary_path)
    


if __name__ == "__main__": 

    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_emb_path", type=str) 
    parser.add_argument("--item_emb_path", type=str) 
    parser.add_argument("--k", type=int, default=1000)
    parser.add_argument("--output_binary_path", type=str) 
    parser.add_argument("--seq_path", type=str, default=None) 
    args = parser.parse_args()

    predict(args.seq_emb_path, args.item_emb_path, args.k, args.output_binary_path, args.seq_path)






