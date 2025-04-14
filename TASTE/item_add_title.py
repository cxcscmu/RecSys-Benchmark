import argparse 
import numpy as np
import pandas as pd


parser = ArgumentParser()
parser.add_argument('--item_path', type=str)
parser.add_argument('--source_item_path', type=str)
parser.add_argument('--title_item_output_path', type=str)
args = parser.parse_args()


with open(item_path, "r", encoding="utf-8") as f:
    head = f.readline()[:-1]
    columns = []
    usecols = []
    dtype = {}
    for field_type in head.split("\t"):
        field, ftype = field_type.split(":")
        columns.append(field)
        usecols.append(field_type)
        dtype[field_type] = np.float64 if ftype == np.float32 else str

df = pd.read_csv(
    args.item_path,
    delimiter="\t",
    usecols=usecols,
    dtype=dtype,
    encoding="utf-8",
    engine="python",
)

# get items ids
required_iids = df['item_id:token'].tolist()

# load titles
source_item_df = pd.read_json(args.source_item_path, lines=True)
iids = source_item_df['parent_asin'].tolist()
titles = source_item_df['title'].tolist()
id_title_map = {}
for i in range(len(iids)): 
    id_title_map[iids[i]] = titles[i]

# get the iid in the official item file 
ordered_titles = []
for iid in required_iids: 
    ordered_titles.append(id_title_map[iid])


df['title:token'] = ordered_titles
df.rename(columns={'categories:token_seq': 'categories:token'}, inplace=True)
breakpoint()
df.to_csv(args.title_item_output_path, sep='\t', index=False, na_rep='None') 


    