#
# Copyright IBM Corp. 2024
# SPDX-License-Identifier: MIT
#

import os
import sys
import torch
import torch.distributed as dist
import time
import numpy as np
import argparse

# optional args : -i iterations   and  -s array size (in MBytes)
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--iterations", type=int, default=5000)
parser.add_argument("-s", "--size", type=int, default=500)

args = parser.parse_args()
maxiter = args.iterations
nMB = args.size

local_rank = int(os.environ["LOCAL_RANK"])
rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])

torch.cuda.set_device(local_rank)

dist.init_process_group("nccl")

if rank == 0:
    print("world size = ", world_size, " NCCL version = ", torch.cuda.nccl.version(), file=sys.stderr)

torch.manual_seed(1235911);

npts = int(nMB*1.0e6/4.0)

Tensor = torch.rand(npts, device='cuda')
torch.cuda.synchronize()

if rank == 0:
    print("size(MB)   avgbw(GB/sec)   maxbw(GB/sec)     minbw(GB/sec)", file=sys.stderr)

mytimes = np.empty(maxiter, dtype=float)

npts = int(nMB*1.0e6/4.0)
nm1 = int(npts - 1)

# launch two calls outside the timing loop
dist.all_reduce(Tensor[0:nm1], op=dist.ReduceOp.SUM)
torch.cuda.synchronize()
dist.all_reduce(Tensor[0:nm1], op=dist.ReduceOp.SUM)
torch.cuda.synchronize()

tbeg = time.perf_counter()
t1 = tbeg
tmin = 1.0e30
tmax = 0.0

for i in range(maxiter):
    dist.all_reduce(Tensor[0:nm1], op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()
    t2 = time.perf_counter()
    if (t2 - t1) < tmin:
        tmin = (t2 - t1)
    if (t2 - t1) > tmax:
        tmax = (t2 - t1)
    mytimes[i] = t2 - t1
    t1 = t2

torch.cuda.synchronize()
tend = time.perf_counter()

elapsed = tend - tbeg

avg_bandwidth = 4.0*2.0e-9*maxiter*npts*((world_size - 1)/world_size)/elapsed
max_bandwidth = 4.0*2.0e-9*npts*((world_size - 1)/world_size)/tmin
min_bandwidth = 4.0*2.0e-9*npts*((world_size - 1)/world_size)/tmax

nglobal = maxiter*world_size

gputimes = torch.from_numpy(mytimes).float().cuda()
alltimes = torch.rand(nglobal, dtype=torch.float, device='cuda')
torch.cuda.synchronize()

dist.all_gather_into_tensor(alltimes, gputimes)
torch.cuda.synchronize()

alltimes = alltimes.cpu()
torch.cuda.synchronize()

if rank == 0:
    print("{:7.1f}".format(nMB), "    ", "{:6.1f}".format(avg_bandwidth), "       ", "{:6.1f}".format(max_bandwidth), "        ", "{:6.1f}".format(min_bandwidth), file=sys.stderr)


if rank == 0:
    outfile = open("times.txt", "w")
    for i in range(nglobal):
        print(alltimes[i].numpy(), file=outfile)
