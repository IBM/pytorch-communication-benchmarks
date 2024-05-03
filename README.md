# pytorch communication benchmarks 
This repository contains simple benchmarks for the most common collective communication calls that are used
in AI training jobs.  These include allreduce, allgather, and reduce-scatter operations.  The benchmarks
are implemented in python scripts that use native PyTorch calls for the collective communication routines.
This makes the tests portable to a variety of systems using any of the backends supported by PyTorch.
It is expected that in most cases users will choose the "nccl" backend, and so that is specified in the
python scripts, but this can be adjusted if desired.  The benchmarks will use the communication library that
is built into your PyTorch distribution, and jobs are launched using your preferred torch.distributed()
launch mechanism.  This ensures that the benchmarks reflect the performance that can be achieved in your
training jobs.

The main benchmark codes are allreduce-loop.py, allgather-loop.py, and reduce-scatter-loop.py.  These codes
loop through a list of global array dimensions and print the average, min, and max bandwidths in units of
GB/sec (10^9 bytes per second) for some number of iterations.  The default number of iterations varies with 
the size of the array, and was chosen to make the tests run quickly on current high-end hardware.  One can
add a command-line option " -m multiplier " to multiply the iteration counts by a constant factor.  The
range of global array dimensions is hard-coded to 8 MB - 10000 MB, which covers the sizes that are commonly
encountered in AI training jobs.  The list of array sizes is close to evenly spaced on a log scale, with
10 data points per decade.  This provides finer granularity than typical power-of-two increments, and is 
suited for plotting the results with a log scale for the X-axis.  These three codes, allreduce-loop.py, 
allgather-loop.py, and reduce-scatter-loop.py have a single communicating group, containing all of the 
torch.distributed() workers.

Job launching is discussed in more detail later, but launches using mpirun, for example, are:

mpirun -np 512 helper.sh python allreduce-loop.py <br />
mpirun -np 512 helper.sh python allgather-loop.py <br />
mpirun -np 512 helper.sh python reduce-scatter-loop.py <br />

where a helper script, helper.sh, is used to set the variables that are needed for the torch.distributed()
environment : MASTER_ADDR, MASTER_PORT, RANK, LOCAL_RANK, WORLD_SIZE.  

Some AI training frameworks support multiple dimensions of parallelization, and the communication patterns 
for these frameworks are more complex.  This repository contains a megatron-allreduce.py script that uses
three levels of parallelization, with tensor, pipeline, and data-parallel groups, as in Megatron-LM.  In this
case one of the main collective communication calls is allreduce within each data-parallel group, and there
will be many independent and possibly concurrent calls to allreduce for the different data-parallel groups.
The number of participants in each data-parallel group is the world size divided by the product of the 
tensor-parallel and pipeline-parallel dimensions.  This can be much smaller than the world size, and that can
improve scaling by reducing the effect of latency.  Since there are multiple independent communicating groups,
we add barrier synchronization and take the effective communication time to be the time from when all groups
start until the last group finishes.  It is also of interest to determine whether each group takes about the
same time, or if one or more groups is causing a delay.  To check on this, the megatron-allreduce.py code
has the group leader for each group report separately on the time spent in allreduce calls.  To launch the
job, one needs to specify the dimensions for tensor and pipeline parallelism :

mpirun -np 512 helper.sh python megatron-allreduce.py -t 4 -p 8 <br />

where the -t option is for tensor-parallel and the -p option is for pipeline-parallel.  For the megatron code,
world_size = tp_size * dp_size * pp_size.  Communication for tensor-parallelism is very fine-grained, 
so that is normally limited to within a node.  The default ordering of ranks is "tensor, data, pipeline"  
but the code also supports an alternative "tensor, pipeline, data" ordering, by adding : --order tpd.

## Launching Jobs

We recommend launching these PyTorch communication benchmarks using the same method that you use for AI 
training jobs.  Our experience has been that the latency measured for smaller global array dimensions can
be sensitive to process affinity, and not all launch methods provide sufficient affinity control.  When
using torchrun, we have found that setting environment variable NCCL_IGNORE_CPU_AFFINITY=1 can help for the
smaller array dimensions.  This variable lets the NCCL library manage affinity, typically by setting a CPU
mask that matches the socket affinity for the GPU assigned to each worker.  If you launch jobs with mpirun 
and a hostfile, MASTER_ADDR should be the hostname where rank 0 is assigned.  We use helper scripts
with lines like the ones listed here : 

source /path/to/conda.env <br />

if [ -z "$OMPI_COMM_WORLD_RANK" ]; then <br />
  \# for MPICH <br />
  let local_size=$MPI_LOCALNRANKS <br />
  let local_rank=$MPI_LOCALRANKID <br />
  let world_size=$PMI_SIZE <br />
  let world_rank=$PMI_RANK <br />
else <br />
  \# for OpenMPI <br />
  let local_size=$OMPI_COMM_WORLD_LOCAL_SIZE <br />
  let local_rank=$OMPI_COMM_WORLD_LOCAL_RANK <br />
  let world_size=$OMPI_COMM_WORLD_SIZE <br />
  let world_rank=$OMPI_COMM_WORLD_RANK <br />
fi <br />

export MASTER_ADDR=yourHost <br />
export MASTER_PORT=29400 <br />
export WORLD_SIZE=$world_size <br />
export RANK=$world_rank <br />
export LOCAL_RANK=$local_rank <br />

exec "$@" <br />

The MPI environment variables above work with current versions of the most popular MPI implementations.
The last line of the helper script execs the arguments that follow.  A sample helper script is included.

When using a job scheduler like slurm or LSF, the list of hosts is generally not known in advance, so the
helper script needs to set MASTER_ADDR at runtime.  For slurm the usual method is : 

head_node=$(/path/to/scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1) <br />
export MASTER_ADDR=$head_node <br />

FOR LSF one can get the host name from $LSF_MCPU_HOSTS : 

head_node=`echo $LSB_MCPU_HOSTS | awk '{print $3}'` <br />
or <br />
head_node=`echo $LSB_MCPU_HOSTS | awk '{print $1}'` <br />
export MASTER_ADDR=$head_node <br />

A helper script can be useful for many other purposes, such as setting NCCL environment variables, setting
process affinity, or flexibly enabling profiling control.

work in progress ...

## License

If you would like to see the detailed LICENSE click [here](LICENSE).
