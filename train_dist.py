import argparse
import socket
import time
from contextlib import contextmanager

import dgl
import dgl.nn.pytorch as dglnn

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm

def main(args):
    print(socket.gethostname(), "Initializing DGL dist")
    dgl.distributed.initialize(args.ip_config)
    if not args.standalone:
        print(socket.gethostname(), "Initializing DGL process group")
        th.distributed.init_process_group(backend=args.backend)
    print(socket.gethostname(), "Initializing DistGraph")
    g = dgl.distributed.DistGraph(args.graph_name, part_config=args.part_config)
    print(socket.gethostname(), "rank:", g.rank())

    ##############----------------------------------------------------------
    from dgl.distributed.dist_tensor import DistTensor
    import dgl.backend as _F
    test_tensor = DistTensor((g.number_of_nodes(), ), _F.float32, "test")
    if g.rank() == 1:
        print(g.rank(), ": ", test_tensor.kvstore._data_store[test_tensor._name][0])
    g.barrier()
    if g.rank() == 0:
        print(test_tensor.kvstore._data_store.keys())
        test_mem = test_tensor.kvstore._data_store[test_tensor._name][0] = 1
    g.barrier()
    if g.rank() == 1:
        print(g.rank(), ": ", test_tensor.kvstore._data_store[test_tensor._name][0])
    g.barrier()
    ##############----------------------------------------------------------
    # run(args, device, data)
    print("parent ends")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GCN")
    parser.add_argument("--graph_name", type=str, help="graph name")
    parser.add_argument("--id", type=int, help="the partition id")
    parser.add_argument(
        "--ip_config", type=str, help="The file for IP configuration"
    )
    parser.add_argument(
        "--part_config", type=str, help="The path to the partition config file"
    )
    parser.add_argument(
        "--n_classes", type=int, default=0, help="the number of classes"
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="gloo",
        help="pytorch distributed backend",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=-1,
        help="the number of GPU device. Use -1 for CPU training",
    )
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--num_hidden", type=int, default=16)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--fan_out", type=str, default="10,25")
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--batch_size_eval", type=int, default=100000)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument(
        "--local_rank", type=int, help="get rank of the process"
    )
    parser.add_argument(
        "--standalone", action="store_true", help="run in the standalone mode"
    )
    parser.add_argument(
        "--pad-data",
        default=False,
        action="store_true",
        help="Pad train nid to the same length across machine, to ensure num "
        "of batches to be the same.",
    )
    args = parser.parse_args()

    print(args)
    main(args)
