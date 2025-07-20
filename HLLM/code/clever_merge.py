import numpy as np
import os 
import sys
import pickle 
from tqdm import tqdm


def read_embed_shape(filename):
    with open(filename, "rb") as f:
        nvecs, dim = np.fromfile(f, count=2, dtype=np.int32)
        return nvecs, dim



def write_fbin_header(f, num, dim):
    f.write(num.to_bytes(4, 'little'))
    f.write(dim.to_bytes(4, 'little'))


base_dir = "/data/group_data/cx_group/REC/ClueWeb-Reco/HLLM_exps/HLLM_amzn-books"
emb_map = {
    "raw": (0, 63, 64), 
}
output_path = "/data/group_data/cx_group/REC/ClueWeb-Reco/HLLM_exps/HLLM_amzn-books/item_embed_full_streamed.bin"


# base_dir = "/data/user_data/jingyuah"
# emb_map = {
#     "HLLM": (0, 0, 1),
# }
# output_path = "/data/user_data/jingyuah/HLLM/item_embed_full_streamed.bin"

# --- Pass 1: Count total number of vectors ---
num_total = 87208656 # 87208655 
dim = 2048

counted_num = 0

print(f"Total vectors: {num_total}, dimension: {dim}")
sys.stdout.flush()

# --- Pass 2: Stream write header + data ---
with open(output_path, "wb") as f_out:
    write_fbin_header(f_out, num_total, dim)
    # for folder_ in ["HLLM_ml-1m", "HLLM_item_encoding_ml-1m"]:
    # for folder_ in ["HLLM"]:
    for folder_ in ["raw"]:
        folder_path = os.path.join(base_dir, folder_)
        start, end, num_shards = emb_map[folder_]
        for i in tqdm(range(start, end+1)):
            file_path = os.path.join(folder_path, f"clueweb-b-en.{i}-of-{num_shards}.pkl")
            print(f"Writing from {file_path}")
            sys.stdout.flush()
            with open(file_path, "rb") as f_in:
                embeds, ids = pickle.load(f_in)
            f_out.write(embeds.tobytes())
            counted_num += embeds.shape[0]
            del embeds

print(f"Counted {counted_num} vectors")
sys.stdout.flush()
