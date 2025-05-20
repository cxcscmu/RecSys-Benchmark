# @Time   : 2020/10/6, 2022/7/18
# @Author : Shanlei Mu, Lei Wang
# @Email  : slmu@ruc.edu.cn, zxcptss@gmail.com

# UPDATE:
# @Time   : 2022/7/8, 2022/07/10, 2022/07/13, 2023/2/11
# @Author : Zhen Tian, Junjie Zhang, Gaowei Zhang
# @Email  : chenyuwuxinn@gmail.com, zjj001128@163.com, zgw15630559577@163.com

"""
recbole.quick_start
########################
"""
import logging
import sys
import os
import torch.distributed as dist
from collections.abc import MutableMapping
from logging import getLogger

from ray import tune

from recbole.config import Config
from recbole.data import (
    create_dataset,
    data_preparation,
)
from recbole.data.transform import construct_transform
from recbole.utils import (
    init_logger,
    get_model,
    get_trainer,
    init_seed,
    set_color,
    get_flops,
    get_environment,
)


def run(
    model,
    dataset,
    exp_name, 
    config_file_list=None,
    config_dict=None,
    saved=True,
    nproc=1,
    world_size=-1,
    ip="localhost",
    port="5678",
    group_offset=0,
):
    print("entering run")
    if nproc == 1 and world_size <= 0:
        res = run_recbole(
            model=model,
            dataset_name=dataset,
            exp_name=exp_name, 
            config_file_list=config_file_list,
            config_dict=config_dict,
            saved=saved,
        )
    else:
        if world_size == -1:
            world_size = nproc
        import torch.multiprocessing as mp

        # Refer to https://discuss.pytorch.org/t/problems-with-torch-multiprocess-spawn-and-simplequeue/69674/2
        # https://discuss.pytorch.org/t/return-from-mp-spawn/94302/2
        queue = mp.get_context("spawn").SimpleQueue()

        config_dict = config_dict or {}
        config_dict.update(
            {
                "world_size": world_size,
                "ip": ip,
                "port": port,
                "nproc": nproc,
                "offset": group_offset,
            }
        )
        kwargs = {
            "config_dict": config_dict,
            "queue": queue,
        }

        mp.spawn(
            run_recboles,
            args=(model, dataset, exp_name, config_file_list, kwargs),
            nprocs=nproc,
            join=True,
        )

        # Normally, there should be only one item in the queue
        res = None if queue.empty() else queue.get()
    return res


def run_recbole(
    model=None,
    dataset_name=None,
    exp_name=None, 
    config_file_list=None,
    config_dict=None,
    saved=True,
    queue=None,
):
    r"""A fast running api, which includes the complete process of
    training and testing a model on a specified dataset

    Args:
        model (str, optional): Model name. Defaults to ``None``.
        dataset (str, optional): Dataset name. Defaults to ``None``.
        config_file_list (list, optional): Config files used to modify experiment parameters. Defaults to ``None``.
        config_dict (dict, optional): Parameters dictionary used to modify experiment parameters. Defaults to ``None``.
        saved (bool, optional): Whether to save the model. Defaults to ``True``.
        queue (torch.multiprocessing.Queue, optional): The queue used to pass the result to the main process. Defaults to ``None``.
    """
    # configurations initialization
    print("entering run_recbole")
    config = Config(
        model=model,
        dataset=dataset_name,
        exp_name=exp_name, 
        config_file_list=config_file_list,
        config_dict=config_dict,
    )
    init_seed(config["seed"], config["reproducibility"])
    # logger initialization
    init_logger(config)
    logger = getLogger()
    logger.info(sys.argv)
    logger.info(config)

    # dataset filtering
    dataset = create_dataset(config)
    logger.info(dataset)

    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)

    train_dict = train_data.dataset.inter_feat.numpy()
    valid_dict = valid_data.dataset.inter_feat.numpy()
    test_dict = test_data.dataset.inter_feat.numpy()

    if 'amzn' in dataset_name: 
        write_data = write_data_amzn 
    elif 'ml' in dataset_name: 
        write_data = write_data_ml
    
    output_dir = f"/data/user_data/bolinw/HSTU/tmp/{dataset_name}/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if write_data: 
        write_data(dataset, train_dict, valid_dict, test_dict, output_dir)

    ############################
    # print ml-1m information
    ############################
    # item_feat = dataset.item_feat.numpy()
    # item_id = item_feat["item_id"]
    # title_id = item_feat["movie_title"]
    # user_feat = dataset.user_feat.numpy()
    # title = dataset.field2id_token["movie_title"]

    # print("dataset: ", dataset)
    # print("item_feat: ", item_feat)
    # print("user_feat: ", user_feat)
    # print("dataset attr: ", dir(dataset))
    # print("dataset.field2id_token: ", dataset.field2id_token)
    # print("train_dict: ", train_dict)

    ###########################
    # print amzn information
    ###########################
    # item_feat = dataset.item_feat.numpy()

    # print("dataset: ", dataset)
    # print("item_feat: ", item_feat)
    # print("dataset attr: ", dir(dataset))
    # print("dataset.field2id_token: ", dataset.field2id_token)
    # print("train_dict: ", train_dict)

def write_data_amzn(dataset, train_dict, valid_dict, test_dict, output_dir=None): 

    user_id_list = dataset.field2id_token['user_id']
    item_id_list = dataset.field2id_token['item_id']

    # train_ratings.csv
    out_path = os.path.join(output_dir, "train_ratings.csv") if output_dir else "train_ratings.csv"
    writer = open(out_path, 'w', encoding='utf-8')
    i = 0
    for user_id, item_id, rating, timestamp in zip (
        train_dict["user_id"], train_dict["item_id"], train_dict["rating"], train_dict["timestamp"]
    ):
        uid = user_id_list[int(user_id)]
        id = item_id_list[int(item_id)]
        rt = int(rating)
        time = int(timestamp)
        writer.write(f"{uid},{id},{rt},{time}\n")
        i += 1
    print("------------------finish---------------")
    print(i)

    # valid_ratings.csv
    out_path = os.path.join(output_dir, "valid_ratings.csv") if output_dir else "valid_ratings.csv"
    writer = open(out_path, 'w', encoding='utf-8')
    i = 0
    for user_id, item_id, rating, timestamp in zip (
        valid_dict["user_id"], valid_dict["item_id"], valid_dict["rating"], valid_dict["timestamp"]
    ):
        uid = user_id_list[int(user_id)]
        id = item_id_list[int(item_id)]
        rt = int(rating)
        time = int(timestamp)
        writer.write(f"{uid},{id},{rt},{time}\n")
        i += 1
    print("------------------finish---------------")
    print(i)

    # test_ratings.csv
    out_path = os.path.join(output_dir, "test_ratings.csv") if output_dir else "test_ratings.csv"
    writer = open(out_path, 'w', encoding='utf-8')
    i = 0
    for user_id, item_id, rating, timestamp in zip (
        test_dict["user_id"], test_dict["item_id"], test_dict["rating"], test_dict["timestamp"]
    ):
        uid = user_id_list[int(user_id)]
        id = item_id_list[int(item_id)]
        rt = int(rating)
        time = int(timestamp)
        writer.write(f"{uid},{id},{rt},{time}\n")
        i += 1
    print("------------------finish---------------")
    print(i)


def write_data_ml(dataset, train_dict, valid_dict, test_dict, output_dir=None): 
        
    # movies.dat
    item_feat = dataset.item_feat.numpy()
    item_id = item_feat["item_id"]
    title_id = item_feat["movie_title"]
    year_id = item_feat["release_year"]
    genre_id = item_feat["genre"]

    title = dataset.field2id_token["movie_title"]
    year = dataset.field2id_token["release_year"]
    genre = dataset.field2id_token["genre"]

    out_path = os.path.join(output_dir, "movies.dat") if output_dir else "movies.dat"
    writer = open(out_path, "w", encoding="utf-8")
    # writer.write(
    #     "%s\t%s\t%s\t%s\n"
    #     % ("item_id", "item_name", "release_year", "categories")  # \t%s, "price",
    # )
    i = 0
    for id, tid, yid, gids in zip(item_id[1:], title_id[1:], year_id[1:], genre_id[1:]):
        id = int(id)
        tid = int(tid)

        name = str(title[tid])
        release_year = year[int(yid)]

        genres_list = []
        for gid in gids:
            gid = int(gid)
            if gid != 0:
                genres_list.append(str(genre[gid]))
        genres = "|".join(genres_list)

        writer.write(f"{id}::{name} ({release_year})::{genres}\n")  # %.2f\t, p
        i += 1
    print("------------------finish---------------")
    print(i)

    # users.dat
    user_feat = dataset.user_feat.numpy()
    user_id = user_feat["user_id"]
    age_id = user_feat["gender"]
    gender_id = user_feat["age"]
    occupation_id = user_feat["occupation"]
    zip_code_id = user_feat["zip_code"]

    age = dataset.field2id_token["gender"]
    gender = dataset.field2id_token["age"]
    occupation = dataset.field2id_token["occupation"]
    zip_code = dataset.field2id_token["zip_code"]

    out_path = os.path.join(output_dir, "users.dat") if output_dir else "users.dat"
    writer = open(out_path, "w", encoding="utf-8")
    i = 0
    for uid, aid, gid, oid, zid in zip(user_id[1:], age_id[1:], gender_id[1:], occupation_id[1:], zip_code_id[1:]):
        uid = int(uid)
        aid = int(aid)
        gid = int(gid)
        oid = int(oid)
        zid = int(zid)

        ag = age[aid]
        gend = gender[gid]
        occup = occupation[oid]
        zipc = zip_code[zid]

        writer.write(f"{uid}::{gend}::{ag}::{occup}::{zipc}\n")  # %.2f\t, p
        i += 1
    print("------------------finish---------------")
    print(i)

    # train_ratings.dat
    out_path = os.path.join(output_dir, "train_ratings.dat") if output_dir else "train_ratings.dat"
    writer = open(out_path, "w", encoding="utf-8")
    i = 0
    # writer.write("%s\t%s\t%s\n" % ("user_id", "seq", "target"))
    for user_id, item_id, rating, timestamp in zip (
        train_dict["user_id"], train_dict["item_id"], train_dict["rating"], train_dict["timestamp"]
    ):
        uid = int(user_id)
        id = int(item_id)
        rt = int(rating)
        time = int(timestamp)
        writer.write(f"{uid}::{id}::{rt}::{time}\n")
        i += 1
    print("------------------finish---------------")
    print(i)

    # valid_ratings.dat
    out_path = os.path.join(output_dir, "valid_ratings.dat") if output_dir else "valid_ratings.dat"
    writer = open(out_path, "w", encoding="utf-8")
    i = 0
    # writer.write("%s\t%s\t%s\n" % ("user_id", "seq", "target"))
    for user_id, item_id, rating, timestamp in zip (
        valid_dict["user_id"], valid_dict["item_id"], valid_dict["rating"], valid_dict["timestamp"]
    ):
        uid = int(user_id)
        id = int(item_id)
        rt = int(rating)
        time = int(timestamp)
        writer.write(f"{uid}::{id}::{rt}::{time}\n")
        i += 1
    print("------------------finish---------------")
    print(i)

    # test_ratings.dat
    out_path = os.path.join(output_dir, "test_ratings.dat") if output_dir else "test_ratings.dat"
    writer = open(out_path, "w", encoding="utf-8")
    i = 0
    # writer.write("%s\t%s\t%s\n" % ("user_id", "seq", "target"))
    for user_id, item_id, rating, timestamp in zip (
        test_dict["user_id"], test_dict["item_id"], test_dict["rating"], test_dict["timestamp"]
    ):
        uid = int(user_id)
        id = int(item_id)
        rt = int(rating)
        time = int(timestamp)
        writer.write(f"{uid}::{id}::{rt}::{time}\n")
        i += 1
    print("------------------finish---------------")
    print(i)

    """
    Finish data processing.
    ########################
    """

def run_recboles(rank, *args):
    kwargs = args[-1]
    if not isinstance(kwargs, MutableMapping):
        raise ValueError(
            f"The last argument of run_recboles should be a dict, but got {type(kwargs)}"
        )
    kwargs["config_dict"] = kwargs.get("config_dict", {})
    kwargs["config_dict"]["local_rank"] = rank
    run_recbole(
        *args[:4],
        **kwargs,
    )


def objective_function(config_dict=None, config_file_list=None, saved=True):
    r"""The default objective_function used in HyperTuning

    Args:
        config_dict (dict, optional): Parameters dictionary used to modify experiment parameters. Defaults to ``None``.
        config_file_list (list, optional): Config files used to modify experiment parameters. Defaults to ``None``.
        saved (bool, optional): Whether to save the model. Defaults to ``True``.
    """

    config = Config(config_dict=config_dict, config_file_list=config_file_list)
    init_seed(config["seed"], config["reproducibility"])
    logger = getLogger()
    for hdlr in logger.handlers[:]:  # remove all old handlers
        logger.removeHandler(hdlr)
    init_logger(config)
    logging.basicConfig(level=logging.ERROR)
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    init_seed(config["seed"], config["reproducibility"])
    model_name = config["model"]
    model = get_model(model_name)(config, train_data._dataset).to(config["device"])
    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, verbose=False, saved=saved
    )
    test_result = trainer.evaluate(test_data, load_best_model=saved)

    tune.report(**test_result)
    return {
        "model": model_name,
        "best_valid_score": best_valid_score,
        "valid_score_bigger": config["valid_metric_bigger"],
        "best_valid_result": best_valid_result,
        "test_result": test_result,
    }


def load_data_and_model(model_file):
    r"""Load filtered dataset, split dataloaders and saved model.

    Args:
        model_file (str): The path of saved model file.

    Returns:
        tuple:
            - config (Config): An instance object of Config, which record parameter information in :attr:`model_file`.
            - model (AbstractRecommender): The model load from :attr:`model_file`.
            - dataset (Dataset): The filtered dataset.
            - train_data (AbstractDataLoader): The dataloader for training.
            - valid_data (AbstractDataLoader): The dataloader for validation.
            - test_data (AbstractDataLoader): The dataloader for testing.
    """
    import torch

    checkpoint = torch.load(model_file)
    config = checkpoint["config"]
    init_seed(config["seed"], config["reproducibility"])
    init_logger(config)
    logger = getLogger()
    logger.info(config)

    dataset = create_dataset(config)
    logger.info(dataset)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    init_seed(config["seed"], config["reproducibility"])
    model = get_model(config["model"])(config, train_data._dataset).to(config["device"])
    model.load_state_dict(checkpoint["state_dict"])
    model.load_other_parameter(checkpoint.get("other_parameter"))

    return config, model, dataset, train_data, valid_data, test_data
