import os 
import re


from collections import defaultdict 
import pandas as pd 
from datetime import datetime



def to_unix_timestamp(datetime_str):
    # Parse the full datetime string
    dt = datetime.strptime(datetime_str, "%m/%d/%Y %I:%M:%S %p")
    # Convert to Unix timestamp
    return int(dt.timestamp())

def extract_ids_with_1(input_str):
    # Match N followed by digits, ending in -1
    return re.findall(r'(N\d+)-1', input_str)


def smart_wrap(s):
    if '"' in s and "'" in s:
        # If both quotes exist, escape double quotes and wrap with double quotes
        return '"' + s.replace('"', '\\"') + '"'
    elif '"' in s:
        return "'" + s + "'"
    elif "'" in s:
        return '"' + s + '"'
    else:
        return s





source_dir = "/data/group_data/cx_group/REC/data/mind_small/source"
split_dir = ["train", "dev"]


# #### Item Data 
item_dict = {}

for split in split_dir: 
    news_path = os.path.join(source_dir, split, "news.tsv")
    with open(news_path, 'r') as f:
        for line in f: 
            parts = line.strip().split('\t')
            item_id = parts[0].strip()
            if item_id in item_dict: 
                continue 
            # join category by , 
            category = parts[1].strip()
            title = parts[3].strip()
            abstract = parts[4].strip()
            item_dict[item_id] = (category, title, abstract)


print("# News: ", len(item_dict)) # # News:  65238

with open("/data/group_data/cx_group/REC/data/mind_small/mind_small.item", 'w') as f:
    f.write("item_id:token\tcategories:token\ttitle:token\tabstract:token\n")
    for item_id in item_dict:
        categories, title, abstract = item_dict[item_id]
        categories = smart_wrap(categories)
        title = smart_wrap(title)
        abstract = smart_wrap(abstract)
        line = f"{item_id}\t{categories}\t{title}\t{abstract}\n"
        f.write(line) 


# #### Interaction Data
inter_dict = defaultdict(list)

for split in split_dir: 
    inter_path = os.path.join(source_dir, split, "behaviors.tsv")
    with open(inter_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            user_id = parts[1].strip()
            if user_id in inter_dict: 
                continue 
            timestamp = to_unix_timestamp(
                datetime_str=parts[2].strip()
            )
            # prev history are in train 
            histories = parts[3].strip().split()
            for hist in histories: 
                inter_dict[user_id].append((hist.strip(), timestamp))
            # last timestamp validation 
            target_id = extract_ids_with_1(parts[4].strip())[0]
            inter_dict[user_id].append((target_id, timestamp))

print("# Users: ", len(inter_dict)) # # Users:  94057

with open("/data/group_data/cx_group/REC/data/mind_small/mind_small.inter", 'w') as f:
    f.write("user_id:token\titem_id:token\trating:float\ttimestamp:float\n")
    for user_id in inter_dict:
        for row in inter_dict[user_id]: 
            item_id, timestamp = row 
            f.write(f"{user_id}\t{item_id}\t1.0\t{timestamp}\n")


