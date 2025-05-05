import os
import pickle 
import numpy as np
        
        
# selected languages -> fixed to en; corpus: B subset 
dataset_dir = "/data/datasets/clueweb22/ClueWeb22_B"
lang = "en"

encode_data = []

lang_dir = os.path.join(dataset_dir, "txt", lang)

# folders under each language (eg. de00) 
for lang_subfolder in os.listdir(lang_dir): 

    lang_subfolder_dir = os.path.join(lang_dir, lang_subfolder)

    # subfolders under each language shard (eg. de0000)
    for subfolder in os.listdir(lang_subfolder_dir): 

        subfolder_dir = os.path.join(lang_subfolder_dir, subfolder)
        # each json shard: .json.gz, .json.gz.checksum, .offset, .offset.checksum
        num_json_shards = len(os.listdir(subfolder_dir)) // 4

        for jsongz_id in range(0, num_json_shards):
            jjsongz_id = str(jsongz_id).zfill(2)
            jsongz_record_path = os.path.join(subfolder_dir, f"{subfolder}-{jjsongz_id}.offset")
            with open(jsongz_record_path, 'r') as fp:
                total_lines_in_jsongz = len(fp.readlines()) - 1 # extra lines per file 
                # record all possible id in the json 
                for doc_id in range(total_lines_in_jsongz): 
                    ddoc_id = str(doc_id).zfill(5)
                    encode_data.append(f"clueweb22-{subfolder}-{jjsongz_id}-{ddoc_id}")
                    
print(f"ClueWeb22B-EN total length: {len(encode_data)}")

ids = list(np.arange(len(encode_data)))

with open("/data/group_data/cx_group/REC/CLUE-Recommendation/cwid_to_id.tsv", "w") as f:
    for cwid, id_ in zip(encode_data, ids): 
        f.write(f"{cwid}\t{id_}\n")