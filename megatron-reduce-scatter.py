#
# Copyright IBM Corp. 2024
# SPDX-License-Identifier: MIT
#

import os
import sys
import torch
import torch.distributed as dist
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--tensor_parallel", type=int, default=1)
parser.add_argument("-p", "--pipeline_parallel", type=int, default=1)
parser.add_argument("-c", "--communicator", choices=["data", "model", "pipeline"], default="data")
parser.add_argument("-m", "--multiplier", type=int, default=1)
parser.add_argument("-o", "--order", choices=["tdp", "tpd"], default="tdp")

args = parser.parse_args()
tp_size = args.tensor_parallel
pp_size = args.pipeline_parallel
communicator = args.communicator
multiplier = args.multiplier
order = args.order

local_rank = int(os.environ["LOCAL_RANK"])
world_rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])

local_size = torch.cuda.device_count()

torch.cuda.set_device(local_rank)

dist.init_process_group("nccl")

if world_rank == 0:
    print("NCCL version : ", torch.cuda.nccl.version(), file=sys.stderr)

_PIPELINE_MODEL_PARALLEL_GROUP = None
_MODEL_PARALLEL_GROUP = None
_DATA_PARALLEL_GROUP = None

dp_size = world_size // (tp_size*pp_size)
mp_size = tp_size * pp_size

if world_rank == 0:
    print("tp_size = ", tp_size, file=sys.stderr)
    print("pp_size = ", pp_size, file=sys.stderr)
    print("dp size = ", dp_size, file=sys.stderr)
    print("mp_size = ", mp_size, file=sys.stderr)
    print(" ", file=sys.stderr)

mynode = world_rank // local_size

num_tp_groups = world_size // tp_size
num_pp_groups = world_size // pp_size
num_dp_groups = world_size // dp_size
num_mp_groups = dp_size

if world_rank == 0:
    print("number of data     parallel groups = ", num_dp_groups, file=sys.stderr)
    print("number of pipeline parallel groups = ", num_pp_groups, file=sys.stderr)
    print("number of tensor   parallel groups = ", num_tp_groups, file=sys.stderr)
    print("number of model    parallel groups = ", num_mp_groups, file=sys.stderr)
    print(" ", file=sys.stderr)

# the tp_group_ranks are always sequential
tp_group_ranks = []
for g in range(num_tp_groups):
    ranks = range(g*tp_size, (g + 1)*tp_size)
    tp_group_ranks.append(list(ranks))

if world_rank == 0:
    print("tp_group_ranks:", file=sys.stderr)
    for g in range(num_tp_groups):
        print(tp_group_ranks[g], file=sys.stderr)
    print(" ", file=sys.stderr)

dp_group_ranks = []
for p in range(pp_size):
    for t in range(tp_size):
        ranks = []
        for d in range(dp_size):
            if order == "tdp":
                myrank = t + d*tp_size + p*tp_size*dp_size
            else:
                myrank = t + p*tp_size + d*tp_size*pp_size
            ranks.append(myrank)
        dp_group_ranks.append(list(ranks))
        group = torch.distributed.new_group(ranks)
        if world_rank in ranks:
            _DATA_PARALLEL_GROUP = group

if world_rank == 0:
    print("dp_group_ranks:", file=sys.stderr)
    for g in range(num_dp_groups):
        print(dp_group_ranks[g], file=sys.stderr)
    print(" ", file=sys.stderr)

pp_group_ranks = []
for d in range(dp_size):
    for t in range(tp_size):
        ranks = []
        for p in range(pp_size):
            if order == "tdp":
                myrank = t + d*tp_size + p*tp_size*dp_size
            else:
                myrank = t + p*tp_size + d*tp_size*pp_size
            ranks.append(myrank)
        pp_group_ranks.append(list(ranks))
        group = torch.distributed.new_group(ranks)
        if world_rank in ranks:
            _PIPELINE_MODEL_PARALLEL_GROUP = group

if world_rank == 0:
    print("pp_group_ranks:", file=sys.stderr)
    for g in range(num_pp_groups):
        print(pp_group_ranks[g], file=sys.stderr)
    print(" ", file=sys.stderr)

mp_group_ranks = []
for d in range(dp_size):
    ranks = []
    for p in range(pp_size):
        for t in range(tp_size):
            if order == "tdp":
                myrank = t + d*tp_size + p*tp_size*dp_size
            else:
                myrank = t + p*tp_size + d*tp_size*pp_size
            ranks.append(myrank)
    mp_group_ranks.append(list(ranks))
    group = torch.distributed.new_group(ranks)
    if world_rank in ranks:
        _MODEL_PARALLEL_GROUP = group
            
if world_rank == 0:
    print("mp_group_ranks:", file=sys.stderr)
    for g in range(num_mp_groups):
        print(mp_group_ranks[g], file=sys.stderr)
    print(" ", file=sys.stderr)

if communicator == "data":
    mygroup = _DATA_PARALLEL_GROUP
    group_size = dp_size
elif communicator == "pipeline":
    mygroup = _PIPELINE_MODEL_PARALLEL_GROUP
    group_size = pp_size
elif communicator == "model":
    mygroup = _MODEL_PARALLEL_GROUP
    group_size = mp_size

if world_rank == 0:
    print("using communicator = ", communicator, "; order =", order, "; group size = ", group_size, file=sys.stderr)
    print(" ", file=sys.stderr)

group_rank = dist.get_rank(group=mygroup)

if  group_rank == 0:
    if order == 'tdp':
        filename = "world_rank." + str(world_rank) + ".tdp.txt"
    else:
        filename = "world_rank." + str(world_rank) + ".tpd.txt"
    outfile = open(filename, 'w')

###############################################################################################

nMB = 10000.0
npts = int(nMB*1.0e6/4.0)

Tensor = torch.rand(npts, device='cuda')

group_is_in_node = 0
if group_rank == 0:
    if mynode == 0:
        group_is_in_node = 1

NodeTensor  = torch.tensor([[group_is_in_node]], dtype=torch.int, device='cuda')
dist.all_reduce(NodeTensor, op=dist.ReduceOp.SUM, group=None)
torch.cuda.synchronize()

NodeTensor = NodeTensor.cpu()

groups_per_node = int(NodeTensor[0])

if world_rank == 0:
    print("groups_per_node = ", groups_per_node, file=sys.stderr)
    print(" ", file=sys.stderr)


if world_rank == 0:
    print(" size(MB)   tavg(usec)    tmin(usec)    tmax(usec)  avgbw(GB/sec)  maxbw(GB/sec)  minbw(GB/sec)", file=sys.stderr)

for nMB in [0.10,0.12,0.15,0.20,0.32,0.40,0.50,0.64,0.80,1.00,1.25,1.50,2.00,3.16,4.00,5.00,6.40,8.00,\
            10.0,12.5,15.0,20.0,31.6,40.0,50.0,64.0,80.0,100.0,125.0,160.0,200.0,250.0,316.0,400.0,500.0,640.0,800.0,\
            1000.0,1250.0,1600.0,2000.0,2500.0,3160.0,4000.0,5000.0,6400.0,8000.0]:

    dist.barrier(group=None)

    if nMB < 10.0:
        maxiter = 100*multiplier
    elif nMB < 512.0:
        maxiter = 20*multiplier
    elif nMB < 2000.0:
        maxiter = 10*multiplier
    else:
        maxiter = 5*multiplier

    nglobal = int(nMB*1.0e6/4.0)
    nlocal  = int((nglobal + 1)/group_size)
    nglobal = nlocal*group_size

    Input  = torch.rand(nglobal, device='cuda')
    Output = torch.rand(nlocal,  device='cuda')
    torch.cuda.synchronize()

    # launch two calls outside the timing loop
    dist.reduce_scatter_tensor(Output, Input, group=mygroup)
    torch.cuda.synchronize()
    dist.reduce_scatter_tensor(Output, Input, group=mygroup)
    torch.cuda.synchronize()

    tbeg = time.perf_counter()
    t1 = tbeg
    tmin = 1.0e30
    tmax = 0.0
    tsum = 0.0

    for i in range(maxiter):
        dist.reduce_scatter_tensor(Output, Input, group=mygroup)
        torch.cuda.synchronize()
        tsum = tsum + (time.perf_counter() - t1)
        dist.barrier(group=None)
        t2 = time.perf_counter()
        if (t2 - t1) < tmin:
            tmin = (t2 - t1)
        if (t2 - t1) > tmax:
            tmax = (t2 - t1)
        t1 = t2

    torch.cuda.synchronize()
    tend = time.perf_counter()

    elapsed = tend - tbeg
    tavg = elapsed / maxiter

    tsum = tsum / maxiter

    del Output
    del Input
    torch.cuda.synchronize()

    factor = groups_per_node

    avgbw = factor*4.0e-9*nglobal*((group_size - 1)/group_size)/tavg
    maxbw = factor*4.0e-9*nglobal*((group_size - 1)/group_size)/tmin
    minbw = factor*4.0e-9*nglobal*((group_size - 1)/group_size)/tmax


    if world_rank == 0:
        print("{:8.2f}".format(nMB), "  ", "{:7.1f}".format(tavg*1.0e6), "      ", "{:7.1f}".format(tmin*1.0e6), "      ", "{:7.1f}".format(tmax*1.0e6), \
              "     ", "{:7.2f}".format(avgbw), "      ", "{:7.2f}".format(maxbw), "      ", "{:7.2f}".format(minbw), file=sys.stderr)

    if group_rank == 0:
        print("world_rank ", world_rank, " reports avg time = ", "{:8.3f}".format(tsum), " msec for array size ", "{:6.1f}".format(nMB), file=outfile)
        outfile.flush()

if group_rank == 0:
    outfile.close()

dist.destroy_process_group()
