import os
import numpy as np
import argparse
from tqdm import tqdm 
import torch
import pickle
import sys

from cProfile import run
from logging import getLogger
import torch
import json

import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer

from REC.config import Config
from REC.utils import init_logger, get_model, init_seed, set_color
from REC.utils.ClueWeb22Api import ClueWeb22Api, create_shards

from REC.data import *
from REC.data.dataset.collate_fn import customize_rmpad_collate
from REC.data.dataset import BatchTextDataset

from REC.trainer import Trainer


import lightning as L
from lightning.fabric.strategies import DeepSpeedStrategy, DDPStrategy


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




class ClueWebBatchTextDataset(Dataset):
    """
    ClueWeb dataset for item encoding. 
    """

    def __init__(self, config, args):

        self.args = args

        # ClueWeb22 location on disk 
        self.dataset_dir = "/data/datasets/clueweb22/ClueWeb22_B"

        # get the cwid - id map 
        self.id_to_cwid = {}
        # the list of internal item ids 
        self.item_list = ['[PAD]'] 
        with open(self.args.id_map_path, "r") as f:
            for line in f:  
                parts = line.strip().split("\t")
                # the encode data is the internal ids -> 1-indexing conversion 
                self.item_list.append(int(parts[1])+1) # internal id: 1-indexing 
                self.id_to_cwid[int(parts[1])+1] = parts[0] # internal id: 1-indexing 

        print(f"EncodeDataset_ClueWeb22 length: {len(self.item_list)}")
   
        if self.args.dataset_number_of_shards > 1:
            self.item_list = create_shards(
                data=self.item_list, 
                num_shards=self.args.dataset_number_of_shards, 
                index=self.args.dataset_shard_index
            )
        print(f"EncodeDataset_ClueWeb22 shard {self.args.dataset_shard_index} length: {len(self.item_list)}")
        print(f"EncodeDataset_ClueWeb22 shard first example: {self.item_list[0]}")
        print(f"EncodeDataset_ClueWeb22 shard first example: {self.id_to_cwid.get(self.item_list[0], 'not found, pad token')}")

        self.item_num = len(self.id_to_cwid) # total number of items 
        self.max_text_length = config['MAX_TEXT_LENGTH']
        self.device = config['device']
        self.text_keys = config['text_keys']
        self.tokenizer = AutoTokenizer.from_pretrained(config['item_pretrain_dir'], trust_remote_code=True)

        self.item_prompt = config['item_prompt']
        self.item_emb_token_n = config['item_emb_token_n']


    def __len__(self):
        # the current shard 
        return len(self.item_list)
        
    def __getitem__(self, index):

        def get_features(cweb_doc_id): 
            # dummy pad 
            if len(cweb_doc_id) == 0: 
                return {} 
            # extract required webpage contents from ClueWeb22Api
            clueweb_api = ClueWeb22Api(cweb_doc_id, self.dataset_dir)
            clean_txt = eval(clueweb_api.get_clean_text())
            content = clean_txt["Clean-Text"]
            title = content.split('\n')[0].replace("\n", "").replace("\t", "").replace("\r", "").replace("\'", "").replace("\"", "").strip()
            content = content.replace("\n", "").replace("\t", "").replace("\r", "").replace("\'", "").replace("\"", "").strip()
            
            item_feature = {
                "title": title, 
                "description": content
            }
            if 'categories' in self.text_keys: 
                topics = ",".join(clueweb_api.get_topics()) 
                item_feature['categories'] = topics
            elif 'genre' in self.text_keys: 
                tags = ",".join(clueweb_api.get_tags()) 
                item_feature['genre'] = tags
            return item_feature

        def process_item(cweb_doc_id):
            # tokenize webpage 
            item_i = get_features(cweb_doc_id)
            text_str = ""
            if len(item_i):
                text_str = f"{self.item_prompt}"
                for key in self.text_keys: # ['title', 'tag', 'description']
                    value = item_i[key]
                    if value and str(value) != 'nan':
                        text_str += f"{key}: {value}"

            ids = self.tokenizer.encode(text_str)
            ids = ids[:self.max_text_length]
            mask = [1] * len(ids)

            # print(f"{cweb_doc_id}: \n {text_str}") ### 

            return ids, mask

        # document processing 
        id_ = self.item_list[index] # internal_cwid 
        # get the corresponing cwid 
        if index == 0: 
            cweb_doc_id = ""
        else: 
            cweb_doc_id = self.id_to_cwid[id_] 

        # get the tokenized ids
        pos_input_ids, pos_cu_input_lens, pos_position_ids = [], [], []
        ids, _ = process_item(cweb_doc_id)

        # flash attention prep 
        pos_input_ids.extend(ids + [0] * self.item_emb_token_n)
        pos_cu_input_lens.append(len(ids) + self.item_emb_token_n)
        pos_position_ids.extend((torch.arange(len(ids) + self.item_emb_token_n) + (self.max_text_length - len(ids))).tolist())
        outputs =  {
            "pos_item_ids": torch.as_tensor(index, dtype=torch.int64),
            "pos_input_ids": torch.as_tensor(pos_input_ids, dtype=torch.int64),
            "pos_cu_input_lens": torch.as_tensor(pos_cu_input_lens, dtype=torch.int64),
            "pos_position_ids": torch.as_tensor(pos_position_ids, dtype=torch.int64)
        } 
        return outputs


    def __skip__(self, num_to_skip): 
        self.item_list = self.item_list[num_to_skip:]


    def __test__(self): 
        # with open('/data/user_data/jingyuah/HLLM/ml_data.pkl', 'rb') as f:
        #     self.item_list, self.item_text_list = pickle.load(f)
        self.item_list = ['[PAD]'] 
        seq = [x+1 for x in [57218388,53530162,25216193,53530162,68207769,42071675,53530162,15297632,2034,123,432,54,1,3645641,2222222]]
        self.item_list.extend(seq)

    # def __getitem__(self, index):

    #     def get_features(index):
    #         # consider pad 
    #         item_i = self.item_text_list.get(self.item_list[index], {})
    #         return item_i

    #     def process_item(index):
    #         # tokenize webpage 
    #         item_i = get_features(index)
    #         text_str = ""
    #         if len(item_i):
    #             text_str = f"{self.item_prompt}"
    #             for key in self.text_keys: 
    #                 value = item_i[key]
    #                 if value and str(value) != 'nan':
    #                     text_str += f"{key}: {value}"
    #         print(text_str) ###
    #         ids = self.tokenizer.encode(text_str)
    #         ids = ids[:self.max_text_length]
    #         mask = [1] * len(ids)
    #         return ids, mask

    #     # get the tokenized ids
    #     pos_input_ids, pos_cu_input_lens, pos_position_ids = [], [], []
    #     ids, _ = process_item(index)

    #     # flash attention prep 
    #     pos_input_ids.extend(ids + [0] * self.item_emb_token_n)
    #     pos_cu_input_lens.append(len(ids) + self.item_emb_token_n)
    #     pos_position_ids.extend((torch.arange(len(ids) + self.item_emb_token_n) + (self.max_text_length - len(ids))).tolist())
    #     outputs =  {
    #         "pos_item_ids": torch.as_tensor(index, dtype=torch.int64),
    #         "pos_input_ids": torch.as_tensor(pos_input_ids, dtype=torch.int64),
    #         "pos_cu_input_lens": torch.as_tensor(pos_cu_input_lens, dtype=torch.int64),
    #         "pos_position_ids": torch.as_tensor(pos_position_ids, dtype=torch.int64)
    #     } 

    #     return outputs



class ClueWebSeqEvalDataset(Dataset): 
    """
    ClueWeb dataset for sequence encoding. 
    """

    def __init__(self, config, args):
        
        self.max_item_list_length = config['MAX_ITEM_LIST_LENGTH_TEST'] if config['MAX_ITEM_LIST_LENGTH_TEST'] else config['MAX_ITEM_LIST_LENGTH']
        self.seq_data = []
        lines = open(args.seq_data_path, 'r').readlines()
        for line in tqdm(lines[1:]):
            history_titles = list()
            line = line.strip().split('\t')
            # read data 
            session_id = line[0]
            history = line[1].split(",")
            history = [x + 1 for x in map(int, history)] # integer internal ids with 1-indexing 
            # directly user internal id -> use the internal id to index into item embed
            self.seq_data.append(history)

    def __len__(self):
        return len(self.seq_data)

    def _padding_sequence(self, sequence, max_length):
        """
            Return a 1-indexing padded internal cw ID sequence 
        """
        sequence = list(np.array(sequence)) 
        pad_len = max_length - len(sequence)
        sequence = [0] * pad_len + sequence # 0 is dummy token
        sequence = sequence[-max_length:]
        return sequence

    def __getitem__(self, index):
        """
            Return the historical interaction sequence for input at the given index 
        """
        history_seq = self.seq_data[index]
        # pad if needed
        history_seq = self._padding_sequence(history_seq, self.max_item_list_length)
        return torch.tensor(history_seq)

    def _fetch_needed_item_embeddings(self, item_embed_path, ids_needed): 
        """
            Iteratively read needed item vectors to dict to avoid overhead 
            --------
            args: 
                item_embed_path: path to item embed binary file 
                unique_needed_iids: np array of unique needed ids  
        """

        print("reading vectors by shards to avoid memory overhead...")
        sys.stdout.flush()

        item_embed_arr = []
        id_to_idx_dict = {}

        end, vector_dim = read_embed_shape(item_embed_path)
        
        start = 0 
        shard = 500000 # 1000000
        # start and end is given as 
        while start < end: 

            # ids needed in this range -> if none, skip reading this shard 
            ids_needed_shard = ids_needed[(ids_needed >= start) & (start+shard > ids_needed)]

            # read shard 
            if end - start < shard:
                shard = end - start
            print(f"reading the ({start},{start+shard}) vectors for range [{start}, {start+shard})...")
            sys.stdout.flush()
            embeds = read_fbin(item_embed_path, start_idx=start, chunk_size=shard)
            
            # fill the needed dict with the 1-index item ids 
            for id_ in ids_needed_shard: 
                # index in shard to retrieve its emb
                target_emb = embeds[id_ - start]
                # map the id to its idx using dict 
                id_to_idx_dict[id_] = len(item_embed_arr)
                # the actual emb
                item_embed_arr.append(torch.from_numpy(target_emb))

            del embeds

            start += shard

        item_embed_arr = torch.stack(item_embed_arr).squeeze(1)
        assert len(id_to_idx_dict) == item_embed_arr.shape[0], f"len(id_to_idx_dict) {len(id_to_idx_dict)} != item_embed_arr.shape[0] {item_embed_arr.shape[0]}"
        print(f"return {len(id_to_idx_dict)} embeds for {ids_needed.shape[0]} unique iids")
        return id_to_idx_dict, item_embed_arr

    def __test__(self): 
        # with open('/data/user_data/jingyuah/HLLM/test_input.pkl', 'rb') as f:
        #     self.seq_data = pickle.load(f)
        # self.seq_data = self.seq_data.tolist()
        self.seq_data = [[1, 2, 3, 4, 5, 6, 7]]

    def _get_all_items(self, item_embed_path): 

        # store all the sequence in terms of their interacted item ids (1-index + 0 as dummy pads)
        item_seqs = []
        for i in range(self.__len__()):
            item_seqs.append(self.__getitem__(i))
        item_seqs = torch.stack(item_seqs) 

        # unique id needed 
        unique_needed_iids = torch.unique(item_seqs).numpy()
        # fetch item emb shape 
        end, vector_dim = read_embed_shape(item_embed_path)

        # # for test 
        # breakpoint()
        # id_to_idx_dict = {}
        # item_embed_arr = []
        # for id_ in unique_needed_iids: 
        #     if id_ == 0: 
        #         continue 
        #     id_to_idx_dict[id_] = len(item_embed_arr)
        #     item_embed_arr.append(torch.ones(1, 2048) * id_) 
        # item_embed_arr = torch.stack(item_embed_arr).squeeze(1)
        # breakpoint()

        # # get the item embeddings needed by item id (1-index)
        
        # item embedding: 1-indexing where index 0 has dummy pad emb
        id_to_idx_dict, item_embed_arr = self._fetch_needed_item_embeddings(item_embed_path, unique_needed_iids)

        print(f"The number of non-dummy item embedding fetched: {len(id_to_idx_dict)}")
        print(f"The min item id embedding fetched: {sorted(id_to_idx_dict.keys())[0]}")
        print(f"The max item id embedding fetched: {sorted(id_to_idx_dict.keys())[-1]}")
        sys.stdout.flush()

        # get the associated features for eaech seq (seq and item_embed_dict are both in 1-indexing mode)
        seq_item_features = []
        for i in range(item_seqs.shape[0]): 
            item_seq = item_seqs[i]
            emb_idx = [id_to_idx_dict[key.item()] for key in item_seq]
            item_seq_embed = item_embed_arr[emb_idx]
            seq_item_features.append(item_seq_embed) # max_num_item_len * 2048

        seq_item_features = torch.stack(seq_item_features)
        # seq_item_features = item_embeds[item_seqs] # one-shot -> mem overhead 

        # use att mask to skip dummy emb
        attention_mask = (item_seqs > 0).int() 

        return seq_item_features, attention_mask



def convert_str(s):
    try:
        if s.lower() == 'none':
            return None
        if s.lower() == 'true':
            return True
        if s.lower() == 'false':
            return False
        float_val = float(s)
        if float_val.is_integer():
            return int(float_val)
        return float_val
    except ValueError:
        print(f"Unable to convert the string '{s}' to None / Bool / Float / Int, retaining the original string.")
        return s


def to_device(data):
    device = "cuda"
    if isinstance(data, tuple) or isinstance(data, list):
        tdata = ()
        for d in data:
            d = d.to(device)
            tdata += (d,)
        return tdata
    elif isinstance(data, dict):
        for k, v in data.items():
            data[k] = v.to(device)
        return data
    else:
        return data.to(device)


def item_encode(model, config, args, output_path, save_step=50): 

    item_data = ClueWebBatchTextDataset(config, args)
    # item_data.__test__()  ### 
    
    # item_batch_size = config['MAX_ITEM_LIST_LENGTH'] * config['train_batch_size']
    item_batch_size = args.batch_size
    item_loader = DataLoader(item_data, batch_size=item_batch_size, num_workers=args.num_workers, shuffle=False, pin_memory=True, collate_fn=customize_rmpad_collate)
    print(f"Inference item_data with {item_batch_size = } {len(item_loader) = }")
    
    # ckpt file 
    batch_output_path = output_path + ".temp"

    # checkpoint resumption 
    if os.path.exists(batch_output_path):
        with open(batch_output_path, 'rb') as f:
            item_feature, cwids, start_from_idx = pickle.load(f)
            print(f"loaded checkpoint from batch {start_from_idx}")
            print(f"checkpoint contains {len(item_feature)} batches")
    else: 
        item_feature = []
        cwids = []
        start_from_idx = -1

    finished_example = (start_from_idx + 1) * item_batch_size
    # manually skip the ones  
    item_data.__skip__(finished_example)
    print(f"number of examples left: {len(item_data)}")
    item_loader = DataLoader(item_data, batch_size=item_batch_size, num_workers=args.num_workers, shuffle=False, pin_memory=True, collate_fn=customize_rmpad_collate)
    print(f"Inference item_data with {item_batch_size = } {len(item_loader) = }")
    
    print(f"Start encoding from batch {start_from_idx+1}")
    model.eval()

    with torch.no_grad():
        for idx, items in tqdm(enumerate(item_loader), total=len(item_loader)):

            # # skip the finished batches -> already handled in __skip__
            # if idx <= start_from_idx: 
            #     continue 

            cwids.extend(items["pos_item_ids"].tolist())
            items = to_device(items)
            items = model(items, mode='compute_item')
            item_feature.append(items.cpu().numpy())

            # checkpoint resumption 
            if idx % save_step == 0: 
                # first save to temp temp location 
                with open(f"{batch_output_path}_1", 'wb') as f:
                    pickle.dump((item_feature, cwids, idx), f)
                print(f"saved checkpoint at: {output_path}.temp at batch {idx}")
                sys.stdout.flush()
                # avoid preemption when saving ckpt...
                os.rename(f"{batch_output_path}_1", batch_output_path)

        if isinstance(items, tuple):
            item_feature = torch.cat([x[0] for x in self.item_feature]), torch.cat([x[1] for x in self.item_feature])
        else:
            item_feature = np.concatenate(item_feature, 0) ### use np instead 

    print(f"final item features has {item_feature.shape[0]} examples")

    # output embeddings 
    with open(output_path, 'wb') as f:
        pickle.dump((item_feature, cwids), f)

    print(f"Finish saving final checkpoint. Remove intermediate checkpoint...")
    os.remove(batch_output_path)


def prepare_seq_feature(config, args, feature_output_path): 
    
    # configurations initialization
    config = Config(config_file_list=config_file)
    device = torch.device("cuda")
    config['device'] = device
    if len(extra_args):
        for i in range(0, len(extra_args), 2):
            key = extra_args[i][2:]
            if key in ["item_encoding", "seq_encoding", "compute_seq_item_feature"]: 
                continue  
            value = extra_args[i + 1]
            try:
                if '[' in value or '{' in value:
                    # added for text_keys? 
                    if "\\" in value: 
                        value = value.replace("\\", "")
                    value = json.loads(value)
                    if isinstance(value, dict):
                        for k, v in value.items():
                            value[k] = convert_str(v)
                    else:
                        value = [convert_str(x) for x in value]
                else:
                    value = convert_str(value)
                if '.' in key:
                    k1, k2 = key.split('.')
                    config[k1][k2] = value
                else:
                    config[key] = value
            except:
                raise ValueError(f"{key} {value} invalid")
    
    # get seq features in terms of item ids
    seq_data = ClueWebSeqEvalDataset(config, args)
    # seq_data.__test__() ###   
    # get seq features in terms of item emb 
    seq_item_features, attention_mask = seq_data._get_all_items(args.item_embed_path) # seq * max_item_length

    print(f"seq_item_features shape: {seq_item_features.shape}")
    print(f"attention_mask shape: {attention_mask.shape}")
    sys.stdout.flush()
    with open(feature_output_path, "wb") as f:
        pickle.dump((seq_item_features, attention_mask), f)


def seq_encode(model, config, args, output_path, device): 

    with open(args.feature_output_path, "rb") as f:
        pos_embedding, attention_mask = pickle.load(f) # pos_embedding = item_feature[item_seq]

    pos_embedding = pos_embedding.to(torch.bfloat16)
    attention_mask = attention_mask.to(torch.bfloat16)

    pos_embedding = to_device(pos_embedding)
    attention_mask = to_device(attention_mask)

    batch_size = args.batch_size
    seq_emb_list = [] 
    with torch.no_grad():
        for i in range(0, pos_embedding.shape[0], batch_size):
            batch_pos_embedding = pos_embedding[i : i + batch_size]
            batch_attention_mask = attention_mask[i : i + batch_size]

            user_embedding = model.user_llm(inputs_embeds=batch_pos_embedding, attention_mask=batch_attention_mask).hidden_states[-1]
            batch_seq_emb = user_embedding[:, -1].to(torch.float32).cpu().numpy()  # Convert back
            seq_emb_list.append(batch_seq_emb)

    seq_emb = np.concatenate(seq_emb_list, axis=0)  
    print(f"seq_embed shape: {seq_emb.shape}")

    # output embeddings 
    write_embed_to_binary(seq_emb, output_path)



def run(config_file=None, extra_args=[], custom_args=None):

    # configurations initialization
    config = Config(config_file_list=config_file)

    device = torch.device("cuda")
    config['device'] = device
    if len(extra_args):
        for i in range(0, len(extra_args), 2):
            key = extra_args[i][2:]
            if key in ["item_encoding", "seq_encoding"]: 
                continue  
            value = extra_args[i + 1]
            try:
                if '[' in value or '{' in value:
                    # added for text_keys? 
                    if "\\" in value: 
                        value = value.replace("\\", "")
                    value = json.loads(value)
                    if isinstance(value, dict):
                        for k, v in value.items():
                            value[k] = convert_str(v)
                    else:
                        value = [convert_str(x) for x in value]
                else:
                    value = convert_str(value)
                if '.' in key:
                    k1, k2 = key.split('.')
                    config[k1][k2] = value
                else:
                    config[key] = value
            except:
                raise ValueError(f"{key} {value} invalid")

    init_seed(config['seed'], config['reproducibility'])
    
    # get model 
    print("getting model with no dataload...")
    model = get_model(config['model'])(config, None).to(device)
    trainer = Trainer(config, model) # use for easy mixed precision setup

    # load ckpt 
    ckpt_path = os.path.join(config['checkpoint_dir'], 'pytorch_model.bin')
    if os.path.exists(ckpt_path):
        print(f"Found checkpoint at {ckpt_path}, attempting to load...")
        ckpt = torch.load(ckpt_path, map_location='cpu')
        # modify the word embedding -> get tops 
        msg = trainer.model.load_state_dict(ckpt, strict=False)
        print(f'Checkpoint loaded from {ckpt_path}')
        print(f'{msg.unexpected_keys = }')
        print(f'{msg.missing_keys = }')
    else:
        print("No checkpoint found. Starting training from scratch.")

    if config['strategy'] == 'deepspeed':
        print(f"Use deepspeed strategy")
        precision = config['precision'] if config['precision'] else '32'
        strategy = DeepSpeedStrategy(stage=config['stage'], precision=precision)
        lite = L.Fabric(accelerator='gpu', strategy=strategy, precision=precision, num_nodes=1)
        lite.launch()
        model, optimizer = lite.setup(trainer.model, trainer.optimizer)
    else:
        print(f"Use DDP strategy")
        precision = config['precision'] if config['precision'] else '32'
        strategy = DDPStrategy(find_unused_parameters=True)
        lite = L.Fabric(accelerator='gpu', strategy=strategy, precision=precision, num_nodes=1)
        lite.launch()
        model = lite.setup(traier.model)


    del trainer
    if optimizer: 
        del optimizer

    # encoding 
    if custom_args.item_encoding: 
        item_data = None
        item_encode(model, config, custom_args, custom_args.output_path, save_step=custom_args.save_step) 
    elif custom_args.seq_encoding:  
        seq_encode(model, config, custom_args, custom_args.seq_embed_output_path, device)  




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", nargs='+', type=str) # HLLM.yaml
    args, extra_args = parser.parse_known_args()

    custom_parser = argparse.ArgumentParser()
    custom_parser.add_argument("--id_map_path", type=str, default=None, help="cwid_id mapping")

    custom_parser.add_argument("--item_encoding", action="store_true", default=None) 
    custom_parser.add_argument("--seq_encoding", action="store_true", default=None) 
    custom_parser.add_argument("--seq_data_path", type=str, default=None, help="Sequence input file path")
    
    custom_parser.add_argument("--compute_seq_item_feature", action="store_true", default=None) 
    custom_parser.add_argument("--item_embed_path", type=str, default=None, help="Item embeddings path")
    custom_parser.add_argument("--feature_output_path", type=str, default=None, help="Path to store the sequence item embeddings")
    custom_parser.add_argument("--seq_embed_output_path", type=str, default=None, help="Path to store the sequence item embeddings")

    custom_parser.add_argument("--best_model_path", type=str, help="Path of the best checkpoint")
    custom_parser.add_argument("--output_path", type=str, help="Path to store output embeddings")
    custom_parser.add_argument("--batch_size", type=int, default=32, help="Set batch size")
    custom_parser.add_argument("--dataset_number_of_shards", type=int, default=1, help="Number of encoding shards")
    custom_parser.add_argument("--dataset_shard_index", type=int, default=0, help="Index of current shard")
    custom_parser.add_argument("--save_step", type=int, default=50, help="Number of batches to perform checkpointing")
    custom_parser.add_argument("--num_workers", type=int, default=1, help="Number of CPUs available for data processing")
    custom_args, _ = custom_parser.parse_known_args()

    config_file = args.config_file

    # compute sequence feature only -> no GPU, large RAM overhead 
    if custom_args.compute_seq_item_feature: 
        print(f"preparing seq features to {custom_args.feature_output_path}")
        prepare_seq_feature(config_file, custom_args, custom_args.feature_output_path)
        exit(0) 


    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank) # single GPU per shard 
    dist.init_process_group(backend='nccl')

    run(config_file=config_file, extra_args=extra_args, custom_args=custom_args)
