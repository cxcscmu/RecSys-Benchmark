import argparse
import os

import pandas

import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument(
    "--source_dir", type=str, help="source directory of the benchmark official splits"
)
parser.add_argument(
    "--target_dir", type=str, help="target directory to output the formatted splits"
)

args, unknown_args = parser.parse_known_args()


col_map = {
    "user_id:FeatureType.TOKEN": "user_id",
    "item_id:FeatureType.TOKEN": "item_id",
    "timestamp:FeatureType.FLOAT": "timestamp",
}


# interaction data
for split in ["train", "valid", "test"]:
    source_path = os.path.join(args.source_dir, f"{split}.inter")
    target_path = os.path.join(args.target_dir, f"{split}_interactions.csv")
    # read df and transform
    df = pd.read_csv(source_path, sep="\t")
    df = df[col_map.keys()]
    df = df.rename(columns=col_map)
    df.to_csv(target_path, sep=",", index=False)
    print(f"transformed {source_path} to {target_path}")

# item datas
source_path = os.path.join(args.source_dir, "item")
target_path = os.path.join(args.target_dir, "item_details.csv")
# read df and transform
df = pd.read_csv(source_path, sep="\t")
# construct the item column map
item_cols = df.columns.tolist()
item_col_map = {}
for item in item_cols:
    item_col_map[item] = item.split(":")[0]
df = df.rename(columns=item_col_map)
df.to_csv(target_path, sep=",", index=False)
print(f"transformed {source_path} to {target_path}")
