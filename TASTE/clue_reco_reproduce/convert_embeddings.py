import numpy as np
import pickle 
import os



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


def write_embed_to_binary(embeddings, output_path): 
    """
    Write the embedding array into a binary file in ANN-Indexing (DiskANN, SPTAG) format. 
    The content of the output file can be access through: embeds = read_fbin(output_path)
    """
    num, dim = embeddings.shape
    with open(output_path, "wb") as f:
        f.write(num.to_bytes(4, 'little'))
        f.write(dim.to_bytes(4, 'little'))
        f.write(embeddings.tobytes())



def write_docids_to_pkl(docids, output_path): 
    with open(output_path, "wb") as f:
        pickle.dump(docids, f)


def convert_encoded_pkl_to_binary(input_path, embed_output_path): 
    with open(input_path, "rb") as f:
        embeds = pickle.load(f)
    write_embed_to_binary(embeds, embed_output_path)


def convert_encoded_pkls_to_binary(input_dir, input_names, embed_output_path, docid_output_path): 
    
    embeds = []
    ids = []
    for file in input_names: 
        with open(os.path.join(input_dir, file), "rb") as f:
            shard_embeds, shard_ids = pickle.load(f)
        # add to the embed part
        embeds.append(shard_embeds)
        ids.extend(shard_ids)

    embeds = np.concatenate(embeds)
    print("result embed shape: ", embeds.shape)
    write_embed_to_binary(embeds, embed_output_path)
    write_docids_to_pkl(ids, docid_output_path)


# convert the sequence embeddings to biinary 
input_dir = "/data/group_data/cx_group/REC/ClueWeb-Reco/TASTE_exps/clueweb22b_TASTE-checkpoint-72500"
seq_file_name = "clueweb22-seq-test.cweb..pkl"
convert_encoded_pkl_to_binary(
    input_path=os.path.join(input_dir, seq_file_name), 
    embed_output_path=os.path.join(input_dir, "test_input_embed.bin")
)

item_file_names = [
    "clueweb22-corpus.cweb.0.pkl",  
    "clueweb22-corpus.cweb.2.pkl",  
    "clueweb22-corpus.cweb.1.pkl",
    "clueweb22-corpus.cweb.3.pkl"
]

# merge and convert the item embeddings to binary  
convert_encoded_pkls_to_binary(
    input_dir=input_dir, 
    input_names=item_file_names,
    embed_output_path=os.path.join(input_dir, "item_embed.bin"), 
    docid_output_path=os.path.join(input_dir, "item_docids.pkl")
)