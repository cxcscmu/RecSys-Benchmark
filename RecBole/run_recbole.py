# @Time   : 2020/7/20
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn

# UPDATE
# @Time   : 2022/7/8, 2020/10/3, 2020/10/1
# @Author : Zhen Tian, Yupeng Hou, Zihan Lin
# @Email  : chenyuwuxinn@gmail.com, houyupeng@ruc.edu.cn, zhlin@ruc.edu.cn

import argparse

from recbole.quick_start import run
from recbole.quick_start import run_recbole_dataprocess
# from recbole.gen_dataset_TASTE import run_recbole_dataprocess

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, help="name of models")
    parser.add_argument(
        "--dataset", "-d", type=str, help="name of datasets"
    )
    parser.add_argument(
        "--data_preprocess", action ='store_true', help="whether or not to go to the data preprocess pipeline"
    )
    parser.add_argument(
        "--exp_name", type=str, default=None, help="name of experiment run"
    )
    parser.add_argument("--config_files", type=str, default=None, help="config files")
    parser.add_argument(
        "--nproc", type=int, default=1, help="the number of process in this group"
    )
    parser.add_argument(
        "--ip", type=str, default="localhost", help="the ip of master node"
    )
    parser.add_argument(
        "--port", type=str, default="5678", help="the port of master node"
    )
    parser.add_argument(
        "--world_size", type=int, default=-1, help="total number of jobs"
    )
    parser.add_argument(
        "--group_offset",
        type=int,
        default=0,
        help="the global rank offset of this group",
    )

    args, _ = parser.parse_known_args()

    config_file_list = (
        args.config_files.strip().split(" ") if args.config_files else None
    )


    if args.data_preprocess: 
        run_recbole_dataprocess(
            args.model,
            args.dataset,
            args.exp_name, 
            config_file_list=config_file_list,
            saved=False, 
        )

    else: 
        run(
            args.model,
            args.dataset,
            args.exp_name, 
            config_file_list=config_file_list,
            nproc=args.nproc,
            world_size=args.world_size,
            ip=args.ip,
            port=args.port,
            group_offset=args.group_offset,
        )
