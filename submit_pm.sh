#!/bin/bash 
#SBATCH -C gpu 
#SBATCH -A nstaff
#SBATCH --ntasks-per-node 4
#SBATCH --cpus-per-task 32
#SBATCH --gpus-per-node 4
#SBATCH --time=06:00:00
#SBATCH --image=nersc/pytorch:ngc-23.04-v0
#SBATCH --module=gpu,nccl-2.15
#SBATCH -J vit-era5
#SBATCH -o %x-%j.out

DATADIR=/pscratch/sd/s/shas1693/data/sc23_tutorial_data/
LOGDIR=${SCRATCH}/sc23-dl-tutorial/logs
mkdir -p ${LOGDIR}
args="${@}"

export FI_MR_CACHE_MONITOR=userfaultfd
export NCCL_NET_GDR_LEVEL=PHB
export HDF5_USE_FILE_LOCKING=FALSE

# Profiling
if [ "${ENABLE_PROFILING:-0}" -eq 1 ]; then
    echo "Enabling profiling..."
    NSYS_ARGS="--trace=cuda,cublas,nvtx --kill none -c cudaProfilerApi -f true"
    NSYS_OUTPUT=${PROFILE_OUTPUT:-"profile"}
    export PROFILE_CMD="nsys profile $NSYS_ARGS -o $NSYS_OUTPUT"
fi

export MASTER_ADDR=$(hostname)

# Reversing order of GPUs to match default CPU affinities from Slurm
export CUDA_VISIBLE_DEVICES=3,2,1,0

set -x
srun -u shifter -V ${DATADIR}:/data -V ${LOGDIR}:/logs \
    bash -c "
    source nccl-2.17.sh
    source export_DDP_vars.sh
    ${PROFILE_CMD} python train.py ${args}
    "