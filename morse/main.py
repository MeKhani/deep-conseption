import argparse
# from utils import init_dir, set_seed, get_num_rel
import random
import numpy as np
import torch
import dgl
import pickle
from meta_trainer import MetaTrainer
# from post_trainer import PostTrainer
import os
from subgraph import gen_subgraph_datasets
from pre_process import data2pkl
def set_seed(seed):
    dgl.seed(seed)
    dgl.random.seed(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
def init_dir(args):
    # state
    if not os.path.exists(args.state_dir):
        os.makedirs(args.state_dir)

    # tensorboard log
    if not os.path.exists(args.tb_log_dir):
        os.makedirs(args.tb_log_dir)

    # logging
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
def get_num_rel(args):
    data = pickle.load(open(args.data_path, 'rb'))
    num_rel = len(np.unique(np.array(data['train_graph']['train'])[:, 1]))

    return num_rel

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--data_name', default='fb237_v1')
    parser.add_argument('--data_name', default='nell_v1')

    parser.add_argument('--name', default='fb237_v1_transe', type=str)

    parser.add_argument('--step', default='meta_train', type=str, choices=['meta_train', 'fine_tune'])
    parser.add_argument('--metatrain_state', default='./state/fb237_v1_transe/fb237_v1_transe.best', type=str)

    parser.add_argument('--state_dir', '-state_dir', default='../dataset/morse/state', type=str)
    parser.add_argument('--log_dir', '-log_dir', default='../dtatset/morse/log', type=str)
    parser.add_argument('--tb_log_dir', '-tb_log_dir', default='../dataset/morse/tb_log', type=str)

    # params for subgraph
    parser.add_argument('--num_train_subgraph', default=10000)
    parser.add_argument('--num_valid_subgraph', default=200)
    parser.add_argument('--num_sample_for_estimate_size', default=50)
    parser.add_argument('--rw_0', default=10, type=int)
    parser.add_argument('--rw_1', default=10, type=int)
    parser.add_argument('--rw_2', default=5, type=int)
    parser.add_argument('--num_sample_cand', default=5, type=int)

    # params for meta-train
    parser.add_argument('--metatrain_num_neg', default=32)
    parser.add_argument('--metatrain_num_epoch', default=10)
    parser.add_argument('--metatrain_bs', default=64, type=int)
    parser.add_argument('--metatrain_lr', default=0.01, type=float)
    parser.add_argument('--metatrain_check_per_step', default=10, type=int)
    parser.add_argument('--indtest_eval_bs', default=512, type=int)

    # params for fine-tune
    parser.add_argument('--posttrain_num_neg', default=64, type=int)
    parser.add_argument('--posttrain_bs', default=512, type=int)
    parser.add_argument('--posttrain_lr', default=0.001, type=int)
    parser.add_argument('--posttrain_num_epoch', default=10, type=int)
    parser.add_argument('--posttrain_check_per_epoch', default=1, type=int)

    # params for R-GCN
    parser.add_argument('--num_layers', default=3, type=int)
    parser.add_argument('--num_bases', default=4, type=int)
    parser.add_argument('--emb_dim', default=32, type=int)

    # params for KGE
    parser.add_argument('--kge', default='TransE', type=str, choices=['TransE', 'DistMult', 'ComplEx', 'RotatE'])
    parser.add_argument('--gamma', default=10, type=float)
    parser.add_argument('--adv_temp', default=1, type=float)

    parser.add_argument('--gpu', default='cpu', type=str)
    parser.add_argument('--seed', default=1234, type=int)

    args = parser.parse_args()
    init_dir(args)

    args.ent_dim = args.emb_dim
    args.rel_dim = args.emb_dim
    if args.kge in ['ComplEx', 'RotatE']:
        args.ent_dim = args.emb_dim * 2
    if args.kge in ['ComplEx']:
        args.rel_dim = args.emb_dim * 2
    # print(f"entities dimention {args.ent_dim} ")

    # specify the paths for original data and subgraph db
    args.data_path = f'./dataset/{args.data_name}.pkl'
    args.db_path = f'./dataset/{args.data_name}_subgraph'

    # load original data and make index
    if not os.path.exists(args.data_path):
        data2pkl(args.data_name)

    if not os.path.exists(args.db_path):
        gen_subgraph_datasets(args)

    args.num_rel = get_num_rel(args)

    set_seed(args.seed)
    
    if args.step == 'meta_train':
        meta_trainer = MetaTrainer(args)
        meta_trainer.train()