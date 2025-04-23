import os
import numpy as np
from tqdm import tqdm

from recbole.data import data_preparation
from recbole.config import Config
from recbole.data import create_dataset
from recbole.utils import init_seed

def export_raw_format(dataset, inter_dict, output_file):
    """Export interaction data to a tab-separated file in raw format"""
    with open(output_file, 'w', encoding='utf-8') as f:
        field_types = {}
        for field in inter_dict.keys():
            if field in dataset.field2type:
                field_types[field] = dataset.field2type[field]
            else:
                field_types[field] = 'token'
        
        header = [f"{field}:{field_types.get(field, 'token')}" for field in inter_dict.keys() 
                  if not field.endswith('_list') and field != 'item_length']
        f.write('\t'.join(header) + '\n')
        
        for i in tqdm(range(len(inter_dict['user_id'])), desc=f"Writing {os.path.basename(output_file)}"):
            row = []
            for field in header:
                field_name = field.split(':')[0]
                if field_name in inter_dict:
                    row.append(str(inter_dict[field_name][i]))
            f.write('\t'.join(row) + '\n')
    print(f"Saved: {output_file}")


def export_amazon_item_file(dataset, output_file):
    """Export Amazon-style item file with mapped token fields and raw float fields."""
    item_feat = dataset.item_feat.numpy()

    # Extract fields
    item_id = item_feat['item_id']
    title_id = item_feat['title']
    category_id = item_feat['categories']
    avg_rating = item_feat['average_rating']
    rating_num = item_feat['rating_number']
    price = item_feat['price']

    # Token mappings
    title_map = dataset.field2id_token['title']
    category_map = dataset.field2id_token['categories']

    # Open file and write header
    with open(output_file, 'w', encoding='utf-8') as writer:
        writer.write('item_id:token\ttitle:token\tcategories:token\taverage_rating:float\trating_number:float\tprice:float\n')

        for i in range(len(item_id)):
            iid = int(item_id[i])  # transformed item_id
            tid = int(title_id[i])
            cid = int(category_id[i])

            title_str = title_map[tid] if tid < len(title_map) else '[UNK]'
            category_str = category_map[cid] if cid < len(category_map) else '[UNK]'

            writer.write(f"{iid}\t{title_str}\t{category_str}\t{avg_rating[i]:.3f}\t{rating_num[i]:.3f}\t{price[i]:.2f}\n")

    print(f"Finished writing: {output_file}")

def export_ml_item_file(dataset, output_file):
    """Export ML-1M item file in raw format with proper genre formatting."""
    item_feat = dataset.item_feat.numpy()

    # Extract each field
    item_id = item_feat['item_id']
    title_id = item_feat['movie_title']
    year_id = item_feat['release_year']
    genre_id_list = item_feat['genre']

    # Token mappings
    title_map = dataset.field2id_token['movie_title']
    year_map = dataset.field2id_token['release_year']
    genre_map = dataset.field2id_token['genre']

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("item_id:token\tmovie_title:token\trelease_year:token\tgenre:token_seq\n")

        for i in range(len(item_id)):
            iid = int(item_id[i])
            tid = int(title_id[i])
            yid = int(year_id[i])
            gid_list = genre_id_list[i]

            title_str = title_map[tid] if tid < len(title_map) else '[UNK]'
            year_str = year_map[yid] if yid < len(year_map) else '[UNK]'
            genre_tokens = [genre_map[gid] for gid in gid_list if gid < len(genre_map) and genre_map[gid] != '[PAD]']
            genre_str = ' '.join(genre_tokens)

            f.write(f"{iid}\t{title_str}\t{year_str}\t{genre_str}\n")

    print(f"Finished writing: {output_file}")


def export_user_file(dataset, output_file):
    """Export user data to a tab-separated file with proper token mapping"""
    if dataset.user_feat is None or len(dataset.user_feat) == 0:
        print("No user features to export.")
        return

    with open(output_file, 'w', encoding='utf-8') as f:
        user_fields = [field for field in dataset.field2type.keys() 
                      if field in dataset.user_feat]
        
        header = [f"{field}:{dataset.field2type[field]}" for field in user_fields]
        f.write('\t'.join(header) + '\n')
        
        user_feat_numpy = dict(dataset.user_feat.numpy())
        
        token_mappings = {}
        for field in user_fields:
            if field in dataset.field2id_token:
                token_mappings[field] = dataset.field2id_token[field]
        
        for i in tqdm(range(len(dataset.user_feat)), desc="Writing user file"):
            row = []
            for field in user_fields:
                if field in user_feat_numpy:
                    value = user_feat_numpy[field][i]
                    
                    if field in token_mappings and isinstance(value, (int, np.integer)):
                        token_id = int(value)
                        if token_id < len(token_mappings[field]):
                            value = token_mappings[field][token_id]
                    
                    row.append(str(value))
                else:
                    row.append("")
                    
            f.write('\t'.join(row) + '\n')
    print(f"Saved: {output_file}")

def export_dataset_raw_format(model, dataset_name, output_dir, config_file_list=None, config_dict=None):
    """Export dataset in raw RecBole format"""
    config = Config(model=model, dataset=dataset_name, config_file_list=config_file_list, config_dict=config_dict)
    init_seed(config['seed'], config['reproducibility'])
    
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    print("Dataset preparation done.")
    
    os.makedirs(output_dir, exist_ok=True)

    train_dict = dict(train_data.dataset.inter_feat.numpy())
    valid_dict = dict(valid_data.dataset.inter_feat.numpy())
    test_dict = dict(test_data.dataset.inter_feat.numpy())
    
    export_raw_format(dataset, train_dict, os.path.join(output_dir, 'train.inter'))
    export_raw_format(dataset, valid_dict, os.path.join(output_dir, 'valid.inter'))
    export_raw_format(dataset, test_dict, os.path.join(output_dir, 'test.inter'))

    if 'amzn' in dataset_name:
        export_amazon_item_file(dataset, os.path.join(output_dir, 'item'))
    else:
        export_ml_item_file(dataset, os.path.join(output_dir, 'item'))
    export_user_file(dataset, os.path.join(output_dir, 'user'))
    
    print("All files exported.")

if __name__ == "__main__":
    export_dataset_raw_format(
        model='NeuMF',
        dataset_name='amzn-toys',
        output_dir='output/amzn-toys',
        config_file_list=['config_amzn.yaml']
    )

