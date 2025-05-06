import argparse 
import os
import pickle 

from tqdm import tqdm 

import numpy as np
import torch
from torch.utils.data import SequentialSampler, DataLoader
from transformers import T5Tokenizer

from src.model import TASTEModel
from utils.data_loader import load_data, load_item_data, SequenceDataset, ItemDataset, ItemDataset_ClueWeb22, load_data_clueweb

from utils.rec_metrics import get_metrics_dict
from utils.util import set_randomseed, init_logger

 
from torch.utils.data import Dataset


def seq_encode(model, test_seqs_dataloader, output_path, device):

    print("***** Running Sequence Encoding *****")
    model.eval()
    model = model.module if hasattr(model, "module") else model

    # store the embeddings and ids 
    seq_emb_list = []

    with torch.no_grad():
        for i, batch in tqdm(enumerate(test_seqs_dataloader), total=len(test_seqs_dataloader)):

            seq_inputs = batch["seq_ids"].to(device)
            seq_masks = batch["seq_masks"].to(device)
            _, seq_emb = model(seq_inputs, seq_masks)
            seq_emb_list.append(seq_emb.cpu().numpy())

    seq_emb_list = np.concatenate(seq_emb_list, 0)

    # output embeddings 
    with open(output_path, 'wb') as f:
        pickle.dump(seq_emb_list, f)

    print('***** Finish Sequence Encoding *****')


def item_encode(model, test_item_dataloader, output_path, device, save_step=50):

    print("***** Running Item Encoding *****")
    model.eval()
    model = model.module if hasattr(model, "module") else model

    # store the embeddings and ids 
    item_emb_list, id_list= [], []

    batch_output_path = output_path + ".temp" 
    if os.path.exists(batch_output_path): 
        print("loading checkpoing...")
        with open(batch_output_path, 'rb') as f:
            item_emb_list, id_list = pickle.load(f) 
            starting_id = id_list[-1]
            start = False 
    else: 
        start = True 

    with torch.no_grad():
        for i, batch in tqdm(enumerate(test_item_dataloader), total=len(test_item_dataloader)):

            # skip completed batches 
            if not start: 
                # continue from next batch  
                if starting_id == batch["ids"][-1]:  
                    start = True
                continue
            id_list.extend(batch["ids"])
            item_inputs = batch["item_ids"].to(device)
            item_masks = batch["item_masks"].to(device)
            _,item_emb = model(item_inputs, item_masks)
            item_emb_list.append(item_emb.cpu().numpy())

            # preemption support: save per 50 batches 
            if (len(id_list) // len(batch["ids"])) % save_step == 0: 
                with open(batch_output_path, 'wb') as f:
                    pickle.dump((item_emb_list, id_list), f)
                print("saving checkpoint...")

    item_emb_list = np.concatenate(item_emb_list, 0)

    # check if all data elements for this shard present 
    print("sanity checking...")
    assert len(id_list) == len(item_emb_list)

    # output embeddings 
    with open(output_path, 'wb') as f:
        pickle.dump((item_emb_list, id_list), f)

    # remove checkpoint if needed 
    if os.path.exists(batch_output_path): 
        os.remove(batch_output_path)

    print('***** Finish Item Encoding *****')



def main():

    parser = argparse.ArgumentParser(description="Additional parser over encoding")
    parser.add_argument("--id_map_path", type=str, help="cwid_id mapping")
    parser.add_argument("--clueweb_path", type=str, default="/data/datasets/clueweb22/ClueWeb22_B", help="positive of clueweb dataset")
    parser.add_argument("--item_encoding", action="store_true", help="Item encoding")
    parser.add_argument("--seq_encoding", action="store_true", help="Sequence encoding")
    parser.add_argument("--seq_data_path", type=str, default=None, help="Sequence input file path")
    parser.add_argument("--item_size", type=int, default=32, help="Set item token length")
    parser.add_argument("--seq_size", type=int, default=256, help="Set seq token length")
    parser.add_argument("--num_passage", type=int, default=2, help="Number of parts for attention sparsity")
    parser.add_argument("--split_num", type=int, default=243, help="Where to split after the prompt")
    parser.add_argument("--best_model_path", type=str, help="Path of the best TASTE checkpoint")
    parser.add_argument("--output_path", type=str, help="Path to store output embeddings")
    parser.add_argument("--batch_size", type=int, default=32, help="Set batch size")
    parser.add_argument("--dataset_number_of_shards", type=int, default=1, help="Number of encoding shards")
    parser.add_argument("--dataset_shard_index", type=int, default=0, help="Index of current shard")
    parser.add_argument("--save_step", type=int, default=50, help="Number of batches to perform checkpointing")
    parser.add_argument("--seed", type=int, default=0, help="Index of current shard")
    args = parser.parse_args()

    # utils
    set_randomseed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load models
    tokenizer = T5Tokenizer.from_pretrained(args.best_model_path)
    model_class = TASTEModel
    model = model_class.from_pretrained(args.best_model_path)
    model.to(device)

    assert args.item_encoding or args.seq_encoding, "Either one of item_encoding or seq_encoding should be set. "

    if args.item_encoding: 
    
        # encode the items (ClueWeb items)
        test_item_dataset = ItemDataset_ClueWeb22(
            args=args,
            tokenizer=tokenizer, 
        )
        test_item_sampler = SequentialSampler(test_item_dataset)
        test_item_dataloader = DataLoader(
            test_item_dataset,
            sampler=test_item_sampler,
            batch_size=args.batch_size,
            drop_last=False,
            num_workers=0,
            collate_fn=test_item_dataset.collect_fn
        )
        
        item_encode(model, test_item_dataloader, args.output_path, device, save_step=args.save_step)
    
    elif args.seq_encoding: 

        test_data = load_data_clueweb(args.seq_data_path, args.id_map_path, args.clueweb_path)
        test_seq_dataset = SequenceDataset(test_data, tokenizer, args) # test_data[0]: ("id: 309 title: fds, id: 34 title: ",target_id)
        test_seq_sampler = SequentialSampler(test_seq_dataset)
        test_seq_dataloader = DataLoader(
            test_seq_dataset,
            sampler=test_seq_sampler,
            batch_size=args.batch_size,
            drop_last=False,
            num_workers=0,
            collate_fn=test_seq_dataset.collect_fn
        )

        seq_encode(model, test_seq_dataloader, args.output_path, device)
    
        


if __name__ == '__main__':
    main()