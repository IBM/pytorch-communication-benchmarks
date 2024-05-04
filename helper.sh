#!/bin/bash
#--------------------------------------------------------------------------------
# syntax : srun ./helper.sh your.exe [args]
#--------------------------------------------------------------------------------

source /path/to/conda.env

cpus_per_node=96
declare -a list=(`seq 0 95`)

head_node=$(/path/to/scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)

let world_size=$SLURM_NTASKS
let world_rank=$SLURM_PROCID
let local_size=$SLURM_NTASKS_PER_NODE
let local_rank=$world_rank%$local_size

export MASTER_ADDR=$head_node
export MASTER_PORT=56789
export WORLD_SIZE=$world_size
export RANK=$world_rank
export LOCAL_RANK=$local_rank

# divide available cpu slots evenly
let cpus_per_rank=$cpus_per_node/$local_size

let ndx=$local_rank*$cpus_per_rank
let start_cpu=${list[$ndx]}
let pdx=$ndx+$cpus_per_rank-1
let stop_cpu=${list[$pdx]}

#-------------------------------------------------
# set an affinity mask for each rank
#-------------------------------------------------
printf -v command "taskset -c %d-%d "  $start_cpu $stop_cpu

#------------------------------------
# exec the command and following args
#------------------------------------
exec $command "$@"

