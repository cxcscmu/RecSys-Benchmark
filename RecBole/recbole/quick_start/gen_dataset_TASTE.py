"""
We directly obtain the data set processed by Recbole from the DIF-SR project:https://github.com/AIM-SE/DIF-SR.
Here is a code example for obtaining the data. If necessary, please run it in the DIF-SR project.
We have provided the processed dataset.
########################
"""


# @Time   : 2020/10/6
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn

"""
recbole.gen_dataset_TASTE
########################
"""
import os
import logging
from logging import getLogger

import torch
import pickle

from recbole.config import Config
from recbole.data import create_dataset, data_preparation, save_split_dataloaders, load_split_dataloaders
from recbole.utils import init_logger, get_model, get_trainer, init_seed, set_color

def write_data_amzn(dataset, train_dict, valid_dict, test_dict, output_dir=None): 

    # item.txt
    item_feat = dataset.item_feat.numpy()
    item_id = item_feat['item_id']
    title_id = item_feat['title']
    category_id = item_feat['categories']
    title = dataset.field2id_token['title']
    category = dataset.field2id_token['categories']
    
    out_path = os.path.join(output_dir, "item.txt") if output_dir else "item.txt"

    writer = open(out_path, 'w', encoding='utf-8')
    writer.write('%s\t%s\t%s\n' % ('item_id', 'item_name','categories'))
    i = 0
    for id, tid,cid in zip(item_id, title_id,category_id):
        id = int(id)
        tid = int(tid)
        cid = int(cid)
        name = str(title[tid])
        cate = str(category[cid])
        writer.write('%d\t%s\t%s\n' % (id, name,cate))
        i += 1
    print('------------------finish---------------')
    print(i)


    # train.txt
    writer = open(os.path.join(output_dir, 'train.txt'), 'w', encoding='utf-8')
    i = 0
    writer.write('%s\t%s\t%s\n' % ('user_id','seq','target'))
    for user_id,seq_list,target in zip(train_dict['user_id'],train_dict['item_id_list'],train_dict['item_id']):
        uid = int(user_id)
        writer.write('%d\t' % uid)
        for id in seq_list:
            writer.write('%d\t' % int(id))
        tid = int(target)
        writer.write('%d\n' % tid)
        i += 1
    print('------------------finish---------------')
    print(i)

    # valid.txt
    writer = open(os.path.join(output_dir, 'valid.txt'), 'w', encoding='utf-8')
    i = 0
    writer.write('%s\t%s\t%s\n' % ('user_id', 'seq', 'target'))
    for user_id, seq_list, target in zip(valid_dict['user_id'], valid_dict['item_id_list'], valid_dict['item_id']):
        uid = int(user_id)
        writer.write('%d\t' % uid)
        for id in seq_list:
            writer.write('%d\t' % int(id))
        tid = int(target)
        writer.write('%d\n' % tid)
        i += 1
    print('------------------finish---------------')
    print(i)

    # test.txt
    writer = open(os.path.join(output_dir, 'test.txt'), 'w', encoding='utf-8')
    i = 0
    writer.write('%s\t%s\t%s\n' % ('user_id', 'seq', 'target'))
    for user_id, seq_list, target in zip(test_dict['user_id'], test_dict['item_id_list'], test_dict['item_id']):
        uid = int(user_id)
        writer.write('%d\t' % uid)
        for id in seq_list:
            writer.write('%d\t' % int(id))
        tid = int(target)
        writer.write('%d\n' % tid)
        i += 1
    print('------------------finish---------------')
    print(i)


def write_data_ml(dataset, train_dict, valid_dict, test_dict, output_dir=None): 
        
    # item.txt
    item_feat = dataset.item_feat.numpy()
    item_id = item_feat["item_id"]
    title_id = item_feat["movie_title"]
    title = dataset.field2id_token["movie_title"]
    out_path = os.path.join(output_dir, "item.txt") if output_dir else "item.txt"
    writer = open(out_path, "w", encoding="utf-8")
    writer.write(
        "%s\t%s\t%s\t%s\n"
        % ("item_id", "item_name", "release_year", "categories")  # \t%s, "price",
    )
    i = 0
    for id, tid in zip(item_id, title_id):
        id = int(id)
        tid = int(tid)
        name = str(title[tid])
        writer.write("%d\t%s\n" % (id, name))  # %.2f\t, p
        i += 1
    print("------------------finish---------------")
    print(i)

    # train.txt
    out_path = os.path.join(output_dir, "train.txt") if output_dir else "train.txt"
    writer = open(out_path, "w", encoding="utf-8")
    i = 0
    writer.write("%s\t%s\t%s\n" % ("user_id", "seq", "target"))
    for user_id, seq_list, target in zip(
        train_dict["user_id"], train_dict["item_id_list"], train_dict["item_id"]
    ):
        uid = int(user_id)
        writer.write("%d\t" % uid)
        for id in seq_list:
            writer.write("%d\t" % int(id))
        tid = int(target)
        writer.write("%d\n" % tid)
        i += 1
    print("------------------finish---------------")
    print(i)

    # valid.txt
    out_path = os.path.join(output_dir, "valid.txt") if output_dir else "valid.txt"
    writer = open(out_path, "w", encoding="utf-8")
    i = 0
    writer.write("%s\t%s\t%s\n" % ("user_id", "seq", "target"))
    for user_id, seq_list, target in zip(
        valid_dict["user_id"], valid_dict["item_id_list"], valid_dict["item_id"]
    ):
        uid = int(user_id)
        writer.write("%d\t" % uid)
        for id in seq_list:
            writer.write("%d\t" % int(id))
        tid = int(target)
        writer.write("%d\n" % tid)
        i += 1
    print("------------------finish---------------")
    print(i)

    # test.txt
    out_path = os.path.join(output_dir, "test.txt") if output_dir else "test.txt"
    writer = open(out_path, "w", encoding="utf-8")
    i = 0
    writer.write("%s\t%s\t%s\n" % ("user_id", "seq", "target"))
    for user_id, seq_list, target in zip(
        test_dict["user_id"], test_dict["item_id_list"], test_dict["item_id"]
    ):
        uid = int(user_id)
        writer.write("%d\t" % uid)
        for id in seq_list:
            writer.write("%d\t" % int(id))
        tid = int(target)
        writer.write("%d\n" % tid)
        i += 1
    print("------------------finish---------------")
    print(i)

    """
    Finish data processing.
    ########################
    """


def run_recbole_dataprocess(model=None, dataset_name=None, exp_name=None, config_file_list=None, config_dict=None, saved=True):
    r""" A fast running api, which includes the complete process of
    training and testing a model on a specified dataset

    Args:
        model (str, optional): Model name. Defaults to ``None``.
        dataset_name (str, optional): Dataset name. Defaults to ``None``.
        config_file_list (list, optional): Config files used to modify experiment parameters. Defaults to ``None``.
        config_dict (dict, optional): Parameters dictionary used to modify experiment parameters. Defaults to ``None``.
        saved (bool, optional): Whether to save the model. Defaults to ``True``.
    """
    # configurations initialization
    config = Config(model=model, dataset=dataset_name, config_file_list=config_file_list, config_dict=config_dict)
    init_seed(config['seed'], config['reproducibility'])
    # logger initialization
    init_logger(config)
    logger = getLogger()

    logger.info(config)

    # dataset filtering
    dataset = create_dataset(config)

    if config['save_dataset']:
        dataset.save()

    logger.info(dataset)

    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)

    train_dict = train_data.dataset.inter_feat.numpy()
    valid_dict = valid_data.dataset.inter_feat.numpy()
    test_dict = test_data.dataset.inter_feat.numpy()

    """
    The following is the code for data processing.
    ########################
    """

    if 'amzn' in dataset_name: 
        write_data = write_data_amzn 
    elif 'ml' in dataset_name: 
        write_data = write_data_ml 

    if write_data: 
        write_data(dataset, train_dict, valid_dict, test_dict, exp_name)