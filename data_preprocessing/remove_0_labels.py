import os
import pandas as pd

base_dir = "/data/group_data/cx_group/REC/data"
source_dir = "mind_small_train"
target_dir = "mind_small_train_pos"

base_dir = "/data/group_data/cx_group/REC/data"
source_file = os.path.join(base_dir, source_dir, f"{source_dir}.inter")

# remove the 
df = pd.read_csv(source_file, sep="\t")
df = df[df["label:float"] != 0]


target_file = os.path.join(base_dir, target_dir, f"{target_dir}.inter")
df.to_csv(target_file, index=False, sep="\t")